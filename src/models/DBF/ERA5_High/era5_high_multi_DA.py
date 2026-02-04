"""
Multi-run data assimilation experiments for DBF on high-resolution ERA5.
"""

import os
import sys
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Tuple

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

# Add src directory to path
current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
sys.path.append(src_directory)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_config")

from src.models.DBF.ERA5.era5_model import ERA5Decoder
from src.models.DBF.ERA5_High.era5_high_trainer import (
    ERA5IOONetwork,
    ObservationMask,
    SpectralKoopmanOperator,
    create_probe_mask,
)
from src.utils.Dataset import ERA5HighDataset
from src.models.DBF.ERA5_High.era5_high_DA import (
    dbf_gaussian_update,
    decode_pairs,
    initial_latent_state,
)


@dataclass(frozen=True)
class AssimilationEvent:
    at: int
    win: int
    obs_offsets: List[int]


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _validate_obs_offsets(obs_offsets: List[int], win: int, at: int) -> None:
    for offset in obs_offsets:
        if not isinstance(offset, int):
            raise ValueError(
                f"Invalid obs_offsets for event at {at}: all offsets must be int."
            )
        if offset < 0 or offset >= win:
            raise ValueError(
                f"Invalid obs_offsets for event at {at}: offset {offset} not in [0, {win - 1}]."
            )


def build_event_map_default(
    observation_schedule: List[int],
    da_window: int,
    window_length: int,
) -> Dict[int, AssimilationEvent]:
    event_map: Dict[int, AssimilationEvent] = {}
    for step in observation_schedule:
        if not isinstance(step, int):
            print(f"Warning: observation schedule entry {step} is not int, skipping.")
            continue
        if step < 0 or step >= window_length:
            print(
                f"Warning: observation step {step} outside [0, {window_length - 1}], skipping."
            )
            continue
        if step + da_window - 1 > window_length - 1:
            print(
                f"Warning: observation step {step} with window {da_window} "
                f"exceeds window_length {window_length}, skipping."
            )
            continue
        obs_offsets = list(range(da_window))
        event_map[step] = AssimilationEvent(at=step, win=da_window, obs_offsets=obs_offsets)
    return event_map


def load_event_map_from_yaml(
    config_path: str,
    *,
    require_sorted_offsets: bool = True,
    require_unique_offsets: bool = True,
    require_nonempty_offsets: bool = True,
    require_zero_offset: bool = True,
    strict_win_tail: bool = False,
) -> Tuple[int, Dict[int, AssimilationEvent]]:
    """
    Load custom DA schedule from YAML and return:
      - online_nsteps (int): window_length for online assimilation loop
      - event_map (Dict[int, AssimilationEvent]): {at_step: event}

    YAML format:
      online_nsteps: <int>
      events:
        - at: <int>
          win: <int>
          obs_offsets: [<int>, ...]

    Robustness features:
      - Clear validation errors with event index and at
      - obs_offsets: non-empty, sorted (optional), unique (optional), include 0 (optional)
      - win and bounds checks
      - optional strict check that max(obs_offsets) == win - 1
    """
    if yaml is None:
        raise ImportError("please pip install pyyaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Custom DA config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise ValueError("YAML config root must be a mapping/dict.")

    if "online_nsteps" not in config:
        raise ValueError("Missing required field 'online_nsteps' in YAML config.")
    if "events" not in config:
        raise ValueError("Missing required field 'events' in YAML config.")

    online_nsteps = config["online_nsteps"]
    if not isinstance(online_nsteps, int) or online_nsteps < 1:
        raise ValueError("'online_nsteps' must be a positive integer.")

    events = config["events"]
    if not isinstance(events, list):
        raise ValueError("'events' must be a list of event mappings.")

    event_map: Dict[int, AssimilationEvent] = {}

    for idx, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(
                f"Event #{idx} must be a mapping with keys 'at', 'win', 'obs_offsets'."
            )

        missing = [k for k in ("at", "win", "obs_offsets") if k not in event]
        if missing:
            raise ValueError(
                f"Event #{idx} is missing required fields: {missing}."
            )

        at = event["at"]
        win = event["win"]
        obs_offsets = event["obs_offsets"]

        # --- validate 'at'
        if not isinstance(at, int):
            raise ValueError(
                f"Event #{idx} 'at' must be int, got {type(at).__name__}."
            )
        if at < 0 or at > online_nsteps - 1:
            raise ValueError(
                f"Event #{idx} at={at} out of bounds [0, {online_nsteps - 1}]."
            )
        if at in event_map:
            raise ValueError(f"Duplicate event 'at' value: {at}.")

        # --- validate 'win'
        if not isinstance(win, int) or win < 1:
            raise ValueError(f"Event at {at} 'win' must be int >= 1, got {win}.")
        if at + win - 1 > online_nsteps - 1:
            raise ValueError(
                f"Event at {at} with win={win} exceeds online_nsteps={online_nsteps} "
                f"(needs at+win-1 <= {online_nsteps - 1})."
            )

        # --- validate 'obs_offsets'
        if not isinstance(obs_offsets, list):
            raise ValueError(f"Event at {at}: 'obs_offsets' must be a list.")

        if require_nonempty_offsets and len(obs_offsets) == 0:
            raise ValueError(f"Event at {at}: 'obs_offsets' cannot be empty.")

        # element-wise validation (int and range)
        _validate_obs_offsets(obs_offsets, win, at)

        # enforce sorted/unique if desired
        if require_unique_offsets and len(set(obs_offsets)) != len(obs_offsets):
            raise ValueError(f"Event at {at}: 'obs_offsets' contains duplicates: {obs_offsets}")

        if require_sorted_offsets and obs_offsets != sorted(obs_offsets):
            raise ValueError(
                f"Event at {at}: 'obs_offsets' must be sorted ascending. Got {obs_offsets}."
            )

        if require_zero_offset and 0 not in obs_offsets:
            raise ValueError(
                f"Event at {at}: 'obs_offsets' must include 0 (window start). Got {obs_offsets}."
            )

        event_map[at] = AssimilationEvent(at=at, win=win, obs_offsets=obs_offsets)

    return online_nsteps, event_map


def build_obs_stack(
    sparse_observations: torch.Tensor,
    step: int,
    obs_offsets: List[int],
) -> Tuple[torch.Tensor, List[int], List[int]]:
    obs_stack = torch.stack([sparse_observations[step + off] for off in obs_offsets])
    observation_time_steps = list(obs_offsets)
    gaps = [
        observation_time_steps[i + 1] - observation_time_steps[i]
        for i in range(len(observation_time_steps) - 1)
    ]
    return obs_stack, observation_time_steps, gaps


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def set_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_denorm(x: torch.Tensor, dataset: ERA5HighDataset) -> torch.Tensor:
    """Denormalize ERA5 High tensors on CPU."""
    if isinstance(x, torch.Tensor):
        x_cpu = x.detach().cpu()
        min_val = dataset.min.reshape(-1, 1, 1)
        max_val = dataset.max.reshape(-1, 1, 1)
        return (x_cpu * (max_val - min_val) + min_val).cpu()
    return x


def compute_metrics(
    da_states: torch.Tensor,
    noda_states: torch.Tensor,
    groundtruth: torch.Tensor,
    dataset: ERA5HighDataset,
    start_offset: int = 1,
) -> Dict[str, np.ndarray]:
    """Compute per-step, per-channel metrics for one assimilation run."""
    mse = []
    rrmse = []
    ssim_scores = []

    T = da_states.shape[0]
    assert start_offset + T <= groundtruth.shape[0], "groundtruth length mismatch"
    for step in range(T):
        target = groundtruth[step + start_offset]  # assimilation starts from index 1
        da = safe_denorm(da_states[step], dataset)
        noda = safe_denorm(noda_states[step], dataset)

        step_mse = []
        step_rrmse = []
        step_ssim = []

        for c in range(target.shape[0]):
            diff_da = (da[c] - target[c]) ** 2
            diff_noda = (noda[c] - target[c]) ** 2

            mse_da_c = diff_da.mean().item()
            mse_noda_c = diff_noda.mean().item()

            rrmse_da_c = (diff_da.sum() / (target[c] ** 2).sum()).sqrt().item()
            rrmse_noda_c = (diff_noda.sum() / (target[c] ** 2).sum()).sqrt().item()

            data_range_c = target[c].max().item() - target[c].min().item()
            if data_range_c > 0:
                ssim_da_c = ssim(target[c].numpy(), da[c].numpy(), data_range=data_range_c)
                ssim_noda_c = ssim(target[c].numpy(), noda[c].numpy(), data_range=data_range_c)
            else:
                ssim_da_c = 1.0
                ssim_noda_c = 1.0

            step_mse.append((mse_da_c, mse_noda_c))
            step_rrmse.append((rrmse_da_c, rrmse_noda_c))
            step_ssim.append((ssim_da_c, ssim_noda_c))

        mse.append(step_mse)
        rrmse.append(step_rrmse)
        ssim_scores.append(step_ssim)

    return {
        "mse": np.array(mse),
        "rrmse": np.array(rrmse),
        "ssim": np.array(ssim_scores),
    }


def _forward_steps(
    koopman: SpectralKoopmanOperator,
    mu: torch.Tensor,
    cov: torch.Tensor,
    steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    for _ in range(steps):
        mu, cov = koopman.predict(mu, cov)
    return mu, cov


def run_multi_continuous_da_experiment(
    obs_ratio: float = 0.15,
    obs_noise_std: float = 0.05,
    observation_schedule: list = [0, 10, 20],
    observation_variance: float | None = None,
    mode: str = "default",
    da_window: int = 4,
    window_length: int = 30,
    num_runs: int = 5,
    early_stop_config: Tuple[int, float] = (100, 1e-3),
    start_T: int = 0,
    da_start_step: int = 1,
    model_name: str = "DBF",
    checkpoint_name: str = "best_model.pt",
    use_rho: bool = True,
    save_prefix: str | None = None,
    config_path: str = "configs/DA/demo_config.yaml",
):
    """Run repeated DA experiments and collect mean/std statistics."""

    set_seed(42)
    device = set_device()
    print(f"Using device: {device}")

    if mode not in {"default", "custom"}:
        raise ValueError("mode must be 'default' or 'custom'.")

    if mode == "custom":
        real_config_path = os.path.join(_repo_root(), config_path)
        window_length, event_map = load_event_map_from_yaml(real_config_path)
        print(f"Loaded custom DA config from {real_config_path}")
    else:
        if da_window < 2:
            raise ValueError("da_window must be >= 2 for default mode.")
        event_map = build_event_map_default(
            observation_schedule=observation_schedule,
            da_window=da_window,
            window_length=window_length,
        )

    ckpt_path = os.path.join("../../../../results", model_name, "ERA5_High", checkpoint_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    cfg = checkpoint.get("config", {})
    latent_dim = int(cfg.get("latent_dim", 512))
    init_cov = float(cfg.get("init_cov", 1.0))
    cov_epsilon = float(cfg.get("cov_epsilon", 1e-6))
    rho_clip = float(cfg.get("rho_clip", 0.2))
    process_noise_init = float(cfg.get("process_noise_init", -2.0))
    hidden_dims = cfg.get("ioo_hidden_dims", [1024, 1024])
    mask_seed = int(cfg.get("mask_seed", 1024))

    # Dataset slice
    forward_step = 12
    dataset = ERA5HighDataset(
        data_path="../../../../data/ERA5_high/raw_data/weatherbench_test.h5",
        seq_length=forward_step,
        min_path="../../../../data/ERA5_high/raw_data/era5high_240x121_min.npy",
        max_path="../../../../data/ERA5_high/raw_data/era5high_240x121_max.npy",
    )

    total_frames = window_length + da_start_step
    raw_data = dataset.data[start_T : start_T + total_frames, ...]
    groundtruth = torch.tensor(raw_data, dtype=torch.float32).permute(0, 3, 1, 2)
    normalized_groundtruth = dataset.normalize(groundtruth)
    print(f"Ground truth slice shape: {groundtruth.shape}")

    # Observation mask
    if "mask" in checkpoint:
        observation_mask = ObservationMask(checkpoint["mask"].bool())
        print("Loaded observation mask from checkpoint.")
    else:
        print("Checkpoint missing mask; regenerating from obs_ratio and seed.")
        mask_tensor = create_probe_mask(
            channels=dataset.C,
            height=dataset.H,
            width=dataset.W,
            rate=obs_ratio,
            seed=mask_seed,
        )
        observation_mask = ObservationMask(mask_tensor)
    observation_mask = observation_mask.to(device)
    print(f"Observation dimension: {observation_mask.obs_dim}")

    decoder = ERA5Decoder(dim_z=latent_dim).to(device)
    decoder.load_state_dict(checkpoint["decoder"])
    decoder.eval()

    ioo = ERA5IOONetwork(
        obs_dim=observation_mask.obs_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
    ).to(device)
    ioo.load_state_dict(checkpoint["ioo"])
    ioo.eval()

    koopman = SpectralKoopmanOperator(
        latent_dim=latent_dim,
        rho_clip=rho_clip,
        process_noise_init=process_noise_init,
    ).to(device)
    koopman.load_state_dict(checkpoint["koopman"])
    koopman.eval()

    num_pairs = koopman.num_pairs
    print(f"Latent spectral pairs: {num_pairs}")

    normalized_gt_device = normalized_groundtruth.to(device)
    obs_all = observation_mask.sample(normalized_gt_device.unsqueeze(0))  # [1, T, obs_dim]
    obs_all_clean = obs_all.detach()

    run_metrics = {"mse": [], "rrmse": [], "ssim": []}
    run_times = []
    run_iterations = []
    first_run_states = None

    cov_rho = init_cov * torch.eye(2, device=device).view(1, 1, 2, 2)
    mu_rho = torch.zeros(1, num_pairs, 2, device=device)

    for run_idx in range(num_runs):
        print(f"\nStarting assimilation run {run_idx + 1}/{num_runs}")
        set_seed(42 + run_idx)

        obs_all = obs_all_clean.clone()
        if obs_noise_std and obs_noise_std > 0:
            obs_all = obs_all + torch.randn_like(obs_all) * obs_noise_std

        sparse_observations = []
        for idx in range(window_length):
            sparse_observations.append(obs_all[:, da_start_step + idx, :].squeeze(0))
        sparse_observations = torch.stack(sparse_observations).to(device)

        mu_prev, cov_prev = initial_latent_state(
            batch_size=1,
            num_pairs=num_pairs,
            device=device,
            init_cov=init_cov,
        )
        init_obs = obs_all[:, 0, :].to(device)
        mu_init_flat, sigma2_init_flat = ioo(init_obs)
        mu_init_pairs = mu_init_flat.view(1, num_pairs, 2)
        cov_init = torch.diag_embed(sigma2_init_flat.view(1, num_pairs, 2))
        mu_prev, cov_prev = dbf_gaussian_update(
            mu_prior=mu_prev,
            cov_prior=cov_prev,
            mu_r=mu_init_pairs,
            cov_r=cov_init,
            mu_rho=mu_rho,
            cov_rho=cov_rho,
            epsilon=cov_epsilon,
            use_rho=use_rho,
        )
        mu_prev, cov_prev = _forward_steps(koopman, mu_prev, cov_prev, da_start_step)

        mu_noda_prev = mu_prev.clone()
        cov_noda_prev = cov_prev.clone()

        da_states = []
        noda_states = []
        run_start = perf_counter()

        for step in range(window_length):
            if step in event_map:
                event = event_map[step]
                obs_offsets = event.obs_offsets
                obs_stack, observation_time_steps, gaps = build_obs_stack(
                    sparse_observations, step, obs_offsets
                )
                if obs_stack.shape[0] != len(obs_offsets):
                    raise RuntimeError(
                        f"Observation stack length mismatch at step {step}."
                    )

                # Sequential updates across the window: forecast to each offset, then update.
                # This converts the windowed likelihood into time-ordered DBF updates.
                mu_work, cov_work = mu_prev, cov_prev
                for idx, obs in enumerate(obs_stack):
                    if idx > 0:
                        mu_work, cov_work = _forward_steps(koopman, mu_work, cov_work, gaps[idx - 1])
                    mu_obs_flat, sigma2_obs_flat = ioo(obs.unsqueeze(0))
                    mu_obs_pairs = mu_obs_flat.view(1, num_pairs, 2)
                    cov_obs = torch.diag_embed(sigma2_obs_flat.view(1, num_pairs, 2))
                    mu_work, cov_work = dbf_gaussian_update(
                        mu_prior=mu_work,
                        cov_prior=cov_work,
                        mu_r=mu_obs_pairs,
                        cov_r=cov_obs,
                        mu_rho=mu_rho,
                        cov_rho=cov_rho,
                        epsilon=cov_epsilon,
                        use_rho=use_rho,
                    )
                mu_post, cov_post = mu_work, cov_work
            else:
                mu_post, cov_post = mu_prev, cov_prev

            decoded_da = decode_pairs(decoder, mu_post).squeeze(0).detach().cpu()
            da_states.append(decoded_da)

            mu_prev, cov_prev = koopman.predict(mu_post, cov_post)

            mu_noda_prev, cov_noda_prev = koopman.predict(mu_noda_prev, cov_noda_prev)
            decoded_noda = decode_pairs(decoder, mu_noda_prev).squeeze(0).detach().cpu()
            noda_states.append(decoded_noda)

        da_stack = torch.stack(da_states)
        noda_stack = torch.stack(noda_states)

        if first_run_states is None:
            first_run_states = da_stack.clone()

        metrics = compute_metrics(da_stack, noda_stack, groundtruth, dataset, start_offset=da_start_step)
        for key in run_metrics:
            run_metrics[key].append(metrics[key])

        run_times.append(perf_counter() - run_start)
        run_iterations.append(0)
        print(f"Run {run_idx + 1} completed.")

    save_dir = "../../../../results/DBF/ERA5_High/DA"
    os.makedirs(save_dir, exist_ok=True)

    def prefixed(name: str) -> str:
        return f"{save_prefix}{name}" if save_prefix else name

    def _as_numpy(value):
        if value is None:
            return np.array(None, dtype=object)
        return np.array(value)

    # Save one run's DA states
    if first_run_states is not None:
        np.save(
            os.path.join(save_dir, prefixed("multi.npy")),
            safe_denorm(first_run_states, dataset).numpy(),
        )
        print(f"Saved sample DA trajectory to {os.path.join(save_dir, prefixed('multi.npy'))}")

    metrics_meanstd = {}
    for key in run_metrics:
        metric_array = np.stack(run_metrics[key], axis=0)  # (runs, steps, channels, 2)
        metrics_meanstd[f"{key}_mean"] = metric_array.mean(axis=0)
        metrics_meanstd[f"{key}_std"] = metric_array.std(axis=0)

    np.savez(
        os.path.join(save_dir, prefixed("multi_meanstd.npz")),
        **metrics_meanstd,
        steps=np.arange(da_start_step, da_start_step + window_length),
        metrics=["MSE", "RRMSE", "SSIM"],
    )
    print(
        f"Saved mean/std metrics to {os.path.join(save_dir, prefixed('multi_meanstd.npz'))}"
    )

    # Overall stats
    overall_stats = {}
    for key in ["mse", "rrmse", "ssim"]:
        run_values = [m.mean() for m in run_metrics[key]]
        overall_stats[f"{key}_mean"] = float(np.mean(run_values))
        overall_stats[f"{key}_std"] = float(np.std(run_values))
        print(
            f"{key.upper()} mean over runs: {overall_stats[f'{key}_mean']:.6f}, std: {overall_stats[f'{key}_std']:.6f}"
        )

    print(f"Average assimilation time: {np.mean(run_times):.2f}s over {num_runs} runs")

    time_info = {
        "assimilation_time": run_times,
        "assimilation_time_mean": float(np.mean(run_times)),
        "assimilation_time_std": float(np.std(run_times)),
        "iteration_counts": run_iterations,
        "iteration_count_mean": float(np.mean(run_iterations)),
        "iteration_count_std": float(np.std(run_iterations)),
    }
    time_info_path = os.path.join(save_dir, prefixed("time_info.npz"))
    np.savez(
        time_info_path,
        assimilation_time=_as_numpy(run_times),
        assimilation_time_mean=time_info["assimilation_time_mean"],
        assimilation_time_std=time_info["assimilation_time_std"],
        iteration_counts=_as_numpy(run_iterations),
        iteration_count_mean=time_info["iteration_count_mean"],
        iteration_count_std=time_info["iteration_count_std"],
    )
    print(f"Saved time info to {time_info_path}")
    return time_info


if __name__ == "__main__":
    run_multi_continuous_da_experiment()
