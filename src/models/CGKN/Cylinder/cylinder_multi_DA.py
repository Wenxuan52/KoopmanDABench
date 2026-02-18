"""
Multi-run continuous data assimilation experiments for CGKN on Cylinder.
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

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

# Add src directory to path
current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
sys.path.append(src_directory)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_config")

from src.utils.Dataset import CylinderDynamicsDataset
from src.models.CGKN.Cylinder.cylinder_model import (
    CYLINDER_settings,
    CylinderDecoder,
    CylinderEncoder,
)
from src.models.CGKN.Cylinder.cylinder_DA import (
    CGFilter,
    CGN,
    CGKN_ODE,
    ProbeSampler,
    compute_metrics,
    make_probe_coords_from_ratio,
    safe_denorm,
    set_device,
    set_seed,
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
                f"Warning: observation step {step} with window {da_window} exceeds window_length {window_length}, skipping."
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

        at = event["at"]
        win = event["win"]
        obs_offsets = event["obs_offsets"]

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
            raise ValueError(
                f"Event at {at} obs_offsets must be sorted ascending. Got {obs_offsets}."
            )
        if require_unique_offsets and len(set(obs_offsets)) != len(obs_offsets):
            raise ValueError(
                f"Event at {at} obs_offsets must be unique. Got {obs_offsets}."
            )
        if require_zero_offset and 0 not in obs_offsets:
            raise ValueError(
                f"Event at {at} obs_offsets must include 0 (current-step observation). Got {obs_offsets}."
            )
        if strict_win_tail and max(obs_offsets) != win - 1:
            raise ValueError(
                f"Event at {at} strict tail check failed: max(obs_offsets)={max(obs_offsets)} != win-1={win - 1}."
            )

        event_map[at] = AssimilationEvent(at=at, win=win, obs_offsets=obs_offsets)

    return online_nsteps, event_map


@torch.no_grad()
def run_multi_continuous_da_experiment(
    obs_ratio: float = 0.15,
    obs_noise_std: float = 0.05,
    observation_schedule: list = [0, 10, 20],
    observation_variance: float | None = None,
    window_length: int = 30,
    num_runs: int = 5,
    da_start_step: int = 1,
    start_T: int = 700,
    sample_idx: int = 3,
    model_name: str = "CGKN",
    ckpt_prefix: str = "stage2",
    probe_layout: str = "random",
    probe_seed: int = 42,
    probe_min_spacing: int = 4,
    use_channels: Sequence[int] = (0, 1),
    dt: float = 0.001,
    da_window: int = 1,
    custom_da_config: str | None = None,
    save_prefix: str | None = None,
):
    set_seed(42)
    device = set_device()
    print(f"Using device: {device}")

    if custom_da_config is not None:
        loaded_window_length, event_map = load_event_map_from_yaml(custom_da_config)
        window_length = loaded_window_length
    else:
        event_map = build_event_map_default(observation_schedule, da_window, window_length)

    if da_start_step < 0:
        raise ValueError("da_start_step must be >= 0")

    results_dir = Path(f"../../../../results/{model_name}/Cylinder")
    ckpt_paths = {
        "encoder": results_dir / f"{ckpt_prefix}_encoder.pt",
        "decoder": results_dir / f"{ckpt_prefix}_decoder.pt",
        "cgn": results_dir / f"{ckpt_prefix}_cgn.pt",
    }
    missing = [name for name, path in ckpt_paths.items() if not path.exists()]
    if missing:
        missing_list = ", ".join(str(ckpt_paths[name]) for name in missing)
        raise FileNotFoundError(f"Missing checkpoint(s): {missing_list}")

    expected_dim_z = CYLINDER_settings["state_feature_dim"][-1]
    encoder = CylinderEncoder(dim_z=expected_dim_z).to(device)
    decoder = CylinderDecoder(dim_z=expected_dim_z).to(device)
    encoder.load_state_dict(torch.load(ckpt_paths["encoder"], map_location=device, weights_only=True))
    decoder.load_state_dict(torch.load(ckpt_paths["decoder"], map_location=device, weights_only=True))

    sigma_path = results_dir / "sigma_hat.npy"
    if not sigma_path.exists():
        raise FileNotFoundError(f"sigma_hat.npy not found at {sigma_path}")
    sigma_hat = torch.tensor(np.load(sigma_path), dtype=torch.float32)

    forward_step = 12
    cyl_train_dataset = CylinderDynamicsDataset(
        data_path="../../../../data/cylinder/cylinder_train_data.npy",
        seq_length=forward_step,
        mean=None,
        std=None,
    )
    cyl_val_dataset = CylinderDynamicsDataset(
        data_path="../../../../data/cylinder/cylinder_val_data.npy",
        seq_length=forward_step,
        mean=cyl_train_dataset.mean,
        std=cyl_train_dataset.std,
    )

    total_frames = da_start_step + window_length + 1
    if start_T + total_frames > cyl_val_dataset.data.shape[1]:
        raise ValueError("Requested window exceeds available cylinder sequence length.")

    raw_data = cyl_val_dataset.data[sample_idx, start_T : start_T + total_frames, ...]
    groundtruth = torch.tensor(raw_data, dtype=torch.float32)
    normalized_groundtruth = cyl_val_dataset.normalize(groundtruth)
    print(f"Ground truth slice shape: {groundtruth.shape}")

    H, W = int(cyl_val_dataset.H), int(cyl_val_dataset.W)
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

    cgn = CGN(dim_u1=dim_u1, dim_z=latent_dim, hidden=128).to(device)
    cgn.load_state_dict(torch.load(ckpt_paths["cgn"], map_location=device, weights_only=True))
    ode_func = CGKN_ODE(cgn).to(device)

    u1_full = probe_sampler.sample(normalized_groundtruth.unsqueeze(0))
    assert u1_full.shape[1] == total_frames, "Probe sampling length mismatch"
    u1_gt = u1_full[0]

    update_mask = torch.zeros(total_frames, dtype=torch.bool)
    for step, event in event_map.items():
        for offset in event.obs_offsets:
            time_idx = da_start_step + step + offset
            if 0 <= time_idx < total_frames:
                update_mask[time_idx] = True
            else:
                print(
                    f"Warning: event at {step} with offset {offset} maps to {time_idx}, outside range, skipping."
                )

    v0_enc = encoder(normalized_groundtruth[:1].unsqueeze(0).to(device))[:, 0, :]
    mu0 = v0_enc.squeeze(0).detach().unsqueeze(-1)
    R0 = 1e-4 * torch.eye(latent_dim, device=device)

    run_metrics = {"mse": [], "rrmse": [], "ssim": []}
    run_times: List[float] = []
    run_iterations = []
    first_run_states = None
    first_run_original_states = None

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

        obs_series = u1_obs.unsqueeze(-1).to(device)

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
            first_run_original_states = noda_stack.clone()

        metrics = compute_metrics(
            da_stack,
            noda_stack,
            groundtruth,
            cyl_val_dataset,
            start_offset=da_start_step,
        )
        for key in run_metrics:
            run_metrics[key].append(metrics[key])

    save_dir = "../../../../results/CGKN/Cylinder/DA"
    os.makedirs(save_dir, exist_ok=True)

    def prefixed(name: str) -> str:
        return f"{save_prefix}{name}" if save_prefix else name

    def _as_numpy(value):
        if value is None:
            return np.array(None, dtype=object)
        return np.array(value)

    if first_run_states is not None:
        np.save(os.path.join(save_dir, prefixed("multi.npy")), safe_denorm(first_run_states, cyl_val_dataset).numpy())
        print(f"Saved sample DA trajectory to {os.path.join(save_dir, prefixed('multi.npy'))}")

    if first_run_original_states is not None:
        np.save(
            os.path.join(save_dir, prefixed("multi_original.npy")),
            safe_denorm(first_run_original_states, cyl_val_dataset).numpy(),
        )
        print(f"Saved sample NoDA trajectory to {os.path.join(save_dir, prefixed('multi_original.npy'))}")

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
