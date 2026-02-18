"""
Multi-run data assimilation experiments for DBF on Cylinder.
"""

import os
import sys
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Tuple

import numpy as np
import torch

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
sys.path.append(src_directory)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_config")

from src.models.DBF.Cylinder.cylinder_model import CylinderDecoder
from src.models.DBF.Cylinder.cylinder_trainer import (
    CylinderIOONetwork,
    ObservationMask,
    SpectralKoopmanOperator,
    create_probe_mask,
)
from src.utils.Dataset import CylinderDynamicsDataset
from src.models.DBF.Cylinder.cylinder_DA import (
    dbf_gaussian_update,
    decode_pairs,
    initial_latent_state,
    compute_metrics,
    safe_denorm,
)


@dataclass(frozen=True)
class AssimilationEvent:
    at: int
    win: int
    obs_offsets: List[int]


def _validate_obs_offsets(obs_offsets: List[int], win: int, at: int) -> None:
    for offset in obs_offsets:
        if not isinstance(offset, int):
            raise ValueError(f"Invalid obs_offsets for event at {at}: all offsets must be int.")
        if offset < 0 or offset >= win:
            raise ValueError(f"Invalid obs_offsets for event at {at}: offset {offset} not in [0, {win - 1}].")


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
            print(f"Warning: observation step {step} outside [0, {window_length - 1}], skipping.")
            continue
        if step + da_window - 1 > window_length - 1:
            print(
                f"Warning: observation step {step} with window {da_window} "
                f"exceeds window_length {window_length}, skipping."
            )
            continue
        event_map[step] = AssimilationEvent(at=step, win=da_window, obs_offsets=list(range(da_window)))
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
            raise ValueError(f"Event #{idx} must be a mapping with keys 'at', 'win', 'obs_offsets'.")

        missing = [k for k in ("at", "win", "obs_offsets") if k not in event]
        if missing:
            raise ValueError(f"Event #{idx} is missing required fields: {missing}.")

        at, win, obs_offsets = event["at"], event["win"], event["obs_offsets"]

        if not isinstance(at, int):
            raise ValueError(f"Event #{idx} 'at' must be int, got {type(at).__name__}.")
        if at < 0 or at > online_nsteps - 1:
            raise ValueError(f"Event #{idx} at={at} out of bounds [0, {online_nsteps - 1}].")
        if at in event_map:
            raise ValueError(f"Duplicate event 'at' value: {at}.")

        if not isinstance(win, int) or win < 1:
            raise ValueError(f"Event at {at} 'win' must be int >= 1, got {win}.")
        if at + win - 1 > online_nsteps - 1:
            raise ValueError(
                f"Event at {at} with win={win} exceeds online_nsteps={online_nsteps} "
                f"(needs at+win-1 <= {online_nsteps - 1})."
            )

        if not isinstance(obs_offsets, list):
            raise ValueError(f"Event at {at} 'obs_offsets' must be list, got {type(obs_offsets).__name__}.")
        if require_nonempty_offsets and len(obs_offsets) == 0:
            raise ValueError(f"Event at {at} has empty obs_offsets; at least one offset is required.")

        _validate_obs_offsets(obs_offsets, win, at)

        if require_sorted_offsets and obs_offsets != sorted(obs_offsets):
            raise ValueError(f"Event at {at} obs_offsets must be sorted ascending. Got {obs_offsets}.")
        if require_unique_offsets and len(set(obs_offsets)) != len(obs_offsets):
            raise ValueError(f"Event at {at} obs_offsets must be unique. Got {obs_offsets}.")
        if require_zero_offset and 0 not in obs_offsets:
            raise ValueError(f"Event at {at} obs_offsets must include 0. Got {obs_offsets}.")
        if strict_win_tail and max(obs_offsets) != win - 1:
            raise ValueError(
                f"Event at {at} strict tail check failed: max(obs_offsets)={max(obs_offsets)} != win-1={win - 1}."
            )

        event_map[at] = AssimilationEvent(at=at, win=win, obs_offsets=obs_offsets)

    return online_nsteps, event_map


def _forward_steps(koopman: SpectralKoopmanOperator, mu: torch.Tensor, cov: torch.Tensor, n_steps: int):
    if n_steps < 0:
        raise ValueError(f"n_steps must be >= 0, got {n_steps}")
    for _ in range(n_steps):
        mu, cov = koopman.predict(mu, cov)
    return mu, cov


def build_obs_stack(
    sparse_observations: torch.Tensor,
    current_step: int,
    obs_offsets: List[int],
) -> Tuple[torch.Tensor, List[int], List[int]]:
    time_indices = [current_step + off for off in obs_offsets]
    observation_time_steps = list(time_indices)
    gaps = []
    for i in range(1, len(observation_time_steps)):
        gaps.append(observation_time_steps[i] - observation_time_steps[i - 1])
    obs_stack = sparse_observations[time_indices]
    return obs_stack, observation_time_steps, gaps


@torch.no_grad()
def run_multi_continuous_da_experiment(
    obs_ratio: float = 0.15,
    obs_noise_std: float = 0.05,
    observation_schedule: List[int] = [0, 10, 20],
    observation_variance: float | None = None,
    window_length: int = 30,
    num_runs: int = 5,
    da_start_step: int = 1,
    start_T: int = 700,
    sample_idx: int = 3,
    model_name: str = "DBF",
    checkpoint_name: str = "best_model.pt",
    use_rho: bool = True,
    da_window: int = 1,
    custom_da_config: str | None = None,
    save_prefix: str | None = None,
):
    del observation_variance

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if custom_da_config is not None:
        loaded_window_length, event_map = load_event_map_from_yaml(custom_da_config)
        window_length = loaded_window_length
    else:
        event_map = build_event_map_default(observation_schedule, da_window, window_length)

    if da_start_step < 0:
        raise ValueError("da_start_step must be >= 0")

    ckpt_path = os.path.join(f"../../../../results/{model_name}/Cylinder", checkpoint_name)
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

    forward_step = 12
    cyl_train_dataset = CylinderDynamicsDataset(
        data_path="../../../../data/cylinder/cylinder_train_data.npy",
        seq_length=forward_step,
        mean=None,
        std=None,
    )
    dataset = CylinderDynamicsDataset(
        data_path="../../../../data/cylinder/cylinder_val_data.npy",
        seq_length=forward_step,
        mean=cyl_train_dataset.mean,
        std=cyl_train_dataset.std,
    )

    total_frames = da_start_step + window_length + 1
    if start_T + total_frames > dataset.data.shape[1]:
        raise ValueError("Requested window exceeds available cylinder sequence length.")

    raw_data = dataset.data[sample_idx, start_T : start_T + total_frames, ...]
    groundtruth = torch.tensor(raw_data, dtype=torch.float32)
    normalized_groundtruth = dataset.normalize(groundtruth)
    print(f"Ground truth slice shape: {groundtruth.shape}")

    if "mask" in checkpoint:
        observation_mask = ObservationMask(checkpoint["mask"].bool())
        print("Loaded observation mask from checkpoint.")
    else:
        print("Checkpoint missing mask; regenerating from obs_ratio and seed.")
        mask_tensor = create_probe_mask(
            channels=dataset.channel,
            height=dataset.H,
            width=dataset.W,
            rate=obs_ratio,
            seed=mask_seed,
        )
        observation_mask = ObservationMask(mask_tensor)
    observation_mask = observation_mask.to(device)

    decoder = CylinderDecoder(dim_z=latent_dim).to(device)
    decoder.load_state_dict(checkpoint["decoder"])
    decoder.eval()

    ioo = CylinderIOONetwork(
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

    normalized_gt_device = normalized_groundtruth.to(device)
    obs_all = observation_mask.sample(normalized_gt_device.unsqueeze(0))
    obs_all_clean = obs_all.detach()[0]

    sparse_observations_clean = obs_all_clean[da_start_step : da_start_step + window_length]

    run_metrics = {"mse": [], "rrmse": [], "ssim": []}
    run_times: List[float] = []
    run_iterations = []
    first_run_states = None

    cov_rho = init_cov * torch.eye(2, device=device).view(1, 1, 2, 2)
    mu_rho = torch.zeros(1, num_pairs, 2, device=device)

    for run_idx in range(num_runs):
        print(f"\nStarting assimilation run {run_idx + 1}/{num_runs}")
        torch.manual_seed(42 + run_idx)
        np.random.seed(42 + run_idx)

        sparse_observations = sparse_observations_clean.clone()
        if obs_noise_std and obs_noise_std > 0:
            sparse_observations = sparse_observations + torch.randn_like(sparse_observations) * obs_noise_std

        mu_prev, cov_prev = initial_latent_state(batch_size=1, num_pairs=num_pairs, device=device, init_cov=init_cov)

        init_obs = obs_all_clean[0].unsqueeze(0)
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
                obs_stack, _, gaps = build_obs_stack(sparse_observations, step, event.obs_offsets)

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

    save_dir = "../../../../results/DBF/Cylinder/DA"
    os.makedirs(save_dir, exist_ok=True)

    def prefixed(name: str) -> str:
        return f"{save_prefix}{name}" if save_prefix else name

    def _as_numpy(value):
        if value is None:
            return np.array(None, dtype=object)
        return np.array(value)

    if first_run_states is not None:
        np.save(
            os.path.join(save_dir, prefixed("multi.npy")),
            safe_denorm(first_run_states, dataset).numpy(),
        )
        print(f"Saved sample DA trajectory to {os.path.join(save_dir, prefixed('multi.npy'))}")

    metrics_meanstd = {}
    for key in run_metrics:
        metric_array = np.stack(run_metrics[key], axis=0)
        metrics_meanstd[f"{key}_mean"] = metric_array.mean(axis=0)
        metrics_meanstd[f"{key}_std"] = metric_array.std(axis=0)

    np.savez(
        os.path.join(save_dir, prefixed("multi_meanstd.npz")),
        **metrics_meanstd,
        steps=np.arange(da_start_step, da_start_step + window_length),
        metrics=["MSE", "RRMSE", "SSIM"],
    )
    print(f"Saved mean/std metrics to {os.path.join(save_dir, prefixed('multi_meanstd.npz'))}")

    for key in ["mse", "rrmse", "ssim"]:
        run_values = [m.mean() for m in run_metrics[key]]
        print(
            f"{key.upper()} mean over runs: {float(np.mean(run_values)):.6f}, std: {float(np.std(run_values)):.6f}"
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
