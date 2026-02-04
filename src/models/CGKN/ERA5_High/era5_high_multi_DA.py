"""
Multi-run data assimilation experiments for CGKN on ERA5 high-resolution data.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torchdiffeq
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

from src.utils.Dataset import ERA5HighDataset
from src.models.CGKN.ERA5.era5_model import ERA5Decoder, ERA5Encoder, ERA5_settings
from src.models.CGKN.ERA5_High.era5_high_DA import (
    CGFilter,
    CGN,
    CGKN_ODE,
    ProbeSampler,
    make_probe_coords_from_ratio,
    set_device,
    set_seed,
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


def safe_denorm(x: torch.Tensor, dataset: ERA5HighDataset) -> torch.Tensor:
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
    mse = []
    rrmse = []
    ssim_scores = []

    T = da_states.shape[0]
    assert start_offset + T <= groundtruth.shape[0], "groundtruth length mismatch"
    for step in range(T):
        target = groundtruth[step + start_offset]
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
    model_name: str = "CGKN",
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

    results_dir = Path(f"../../../../results/{model_name}/ERA5_High")
    ckpt_paths = {
        "encoder": results_dir / "stage2_encoder.pt",
        "decoder": results_dir / "stage2_decoder.pt",
        "cgn": results_dir / "stage2_cgn.pt",
    }
    missing = [name for name, path in ckpt_paths.items() if not path.exists()]
    if missing:
        missing_list = ", ".join(str(ckpt_paths[name]) for name in missing)
        raise FileNotFoundError(f"Missing checkpoint(s): {missing_list}")

    expected_dim_z = ERA5_settings["state_feature_dim"][-1]
    encoder = ERA5Encoder(dim_z=expected_dim_z).to(device)
    decoder = ERA5Decoder(dim_z=expected_dim_z).to(device)
    encoder.load_state_dict(torch.load(ckpt_paths["encoder"], map_location=device, weights_only=True))
    decoder.load_state_dict(torch.load(ckpt_paths["decoder"], map_location=device, weights_only=True))

    sigma_path = results_dir / "sigma_hat.npy"
    if not sigma_path.exists():
        raise FileNotFoundError(f"sigma_hat.npy not found at {sigma_path}")
    sigma_hat = torch.tensor(np.load(sigma_path), dtype=torch.float32)

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

    probe_layout = "random"
    probe_seed = 42
    probe_min_spacing = 4
    use_channels: Sequence[int] = (0, 1)
    dt = 0.001

    H, W = int(dataset.H), int(dataset.W)
    coords_path = results_dir / "probe_coords.npy"
    if coords_path.exists():
        coords = np.load(coords_path).tolist()
        print(f"Loaded probe coordinates from {coords_path}")
    else:
        coords = make_probe_coords_from_ratio(
            H, W, obs_ratio, layout=probe_layout, seed=probe_seed, min_spacing=probe_min_spacing
        )
        print(f"Probe coordinates not found at {coords_path}, generated {len(coords)} points instead.")
    probe_sampler = ProbeSampler(coords, use_channels)
    dim_u1 = probe_sampler.dim_u1
    print(f"[OBS] dim_u1 = {dim_u1}  (points={len(coords)}, channels={len(use_channels)})")

    latent_dim = sigma_hat.numel() - dim_u1
    assert latent_dim > 0, "sigma_hat length must exceed dim_u1 to include latent components"
    if latent_dim != expected_dim_z:
        raise ValueError(f"latent dim from sigma_hat ({latent_dim}) does not match expected {expected_dim_z}")
    if sigma_hat.numel() != dim_u1 + latent_dim:
        raise ValueError(
            f"sigma_hat length {sigma_hat.numel()} does not match expected dim_u1+dim_z ({dim_u1 + latent_dim})."
        )

    cgn = CGN(dim_u1=dim_u1, dim_z=latent_dim, hidden=128).to(device)
    cgn.load_state_dict(torch.load(ckpt_paths["cgn"], map_location=device, weights_only=True))
    ode_func = CGKN_ODE(cgn).to(device)

    u1_full = probe_sampler.sample(normalized_groundtruth.unsqueeze(0))
    assert u1_full.shape[1] == total_frames, "Probe sampling length mismatch"
    u1_gt = u1_full[0]  # [T, dim_u1]

    # Build update mask from event windows (continuous DA).
    update_mask = torch.zeros(total_frames, dtype=torch.bool)
    for step, event in event_map.items():
        for offset in event.obs_offsets:
            time_idx = da_start_step + step + offset
            if 0 <= time_idx < total_frames:
                update_mask[time_idx] = True
            else:
                print(
                    f"Warning: event at {step} with offset {offset} maps to {time_idx}, "
                    "outside available range, skipping."
                )

    v0_enc = encoder(normalized_groundtruth[:1].unsqueeze(0).to(device))[:, 0, :]  # [1, dim_z]
    mu0 = v0_enc.squeeze(0).detach().unsqueeze(-1)  # [dim_z, 1]
    R0 = 1e-4 * torch.eye(latent_dim, device=device)

    run_metrics = {"mse": [], "rrmse": [], "ssim": []}
    run_times: List[float] = []
    run_iterations = []
    first_run_states = None

    update_mask = update_mask.to(device)

    for run_idx in range(num_runs):
        print(f"\nStarting assimilation run {run_idx + 1}/{num_runs}")
        set_seed(42 + run_idx)

        u1_obs = torch.full_like(u1_gt, float("nan"))
        update_indices = update_mask.nonzero(as_tuple=True)[0].tolist()
        for step in update_indices:
            obs = u1_gt[step].clone()
            if obs_noise_std > 0:
                obs = obs + torch.randn_like(obs) * obs_noise_std
            u1_obs[step] = obs

        if update_mask[0]:
            u1_init = u1_obs[0].clone()
        else:
            u1_init = u1_gt[0].clone()

        obs_series = u1_obs.unsqueeze(-1).to(device)  # [T, dim_u1, 1]

        run_start = perf_counter()
        obs_floor_std = 0.02
        obs_var = (max(obs_noise_std, obs_floor_std) ** 2) if observation_variance is None else observation_variance

        mu_post, _, _ = CGFilter(
            cgn,
            sigma_hat.to(device),
            u1_init.to(device),
            obs_series,
            update_mask,
            mu0,
            R0,
            dt,
            observation_variance=obs_var,
        )
        mu_v = mu_post.squeeze(-1)
        da_fields = decoder(mu_v.unsqueeze(0)).squeeze(0).detach().cpu()
        da_stack = da_fields[da_start_step : da_start_step + window_length]
        run_times.append(perf_counter() - run_start)
        run_iterations.append(0)

        tspan = torch.linspace(0.0, (total_frames - 1) * dt, total_frames, device=device)
        v0 = torch.zeros(1, latent_dim, device=device)
        uext0 = torch.cat([u1_init.to(device).unsqueeze(0), v0], dim=-1)
        uext_pred = torchdiffeq.odeint(ode_func, uext0, tspan, method="rk4", options={"step_size": dt})
        uext_pred = uext_pred.transpose(0, 1)
        v_pred = uext_pred[:, :, dim_u1:]
        noda_fields = decoder(v_pred).squeeze(0).detach().cpu()
        noda_stack = noda_fields[da_start_step : da_start_step + window_length]

        if first_run_states is None:
            first_run_states = da_stack.clone()

        metrics = compute_metrics(da_stack, noda_stack, groundtruth, dataset, start_offset=da_start_step)
        for key in run_metrics:
            run_metrics[key].append(metrics[key])

    save_dir = "../../../../results/CGKN/ERA5_High/DA"
    os.makedirs(save_dir, exist_ok=True)

    def prefixed(name: str) -> str:
        return f"{save_prefix}{name}" if save_prefix else name

    def _as_numpy(value):
        if value is None:
            return np.array(None, dtype=object)
        return np.array(value)

    if first_run_states is not None:
        np.save(os.path.join(save_dir, prefixed("multi.npy")), safe_denorm(first_run_states, dataset).numpy())
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
            f"{key.upper()} mean over runs: {float(np.mean(run_values)):.6f}, "
            f"std: {float(np.std(run_values)):.6f}"
        )

    if run_times:
        print(f"Average assimilation time: {np.mean(run_times):.2f}s over {num_runs} runs")

    time_info = {
        "assimilation_time": run_times,
        "assimilation_time_mean": float(np.mean(run_times)) if run_times else 0.0,
        "assimilation_time_std": float(np.std(run_times)) if run_times else 0.0,
        "iteration_counts": run_iterations,
        "iteration_count_mean": float(np.mean(run_iterations)) if run_iterations else 0.0,
        "iteration_count_std": float(np.std(run_iterations)) if run_iterations else 0.0,
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
