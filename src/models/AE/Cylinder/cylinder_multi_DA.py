"""
Multi-run data assimilation experiments for AE on Cylinder.
"""

import os
import sys
from dataclasses import dataclass
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

from src.models.AE.Cylinder.cylinder_model import CYLINDER_C_FORWARD
from src.models.AE.dabase import (
    MLPDAExecutor,
    UnifiedDynamicSparseObservationHandler,
    set_device,
    set_seed,
)
from src.utils.Dataset import CylinderDynamicsDataset


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
            raise ValueError(f"Event #{idx} is missing required fields: {missing}.")

        at = event["at"]
        win = event["win"]
        obs_offsets = event["obs_offsets"]

        # --- validate 'at'
        if not isinstance(at, int):
            raise ValueError(f"Event #{idx} 'at' must be int, got {type(at).__name__}.")
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

        _validate_obs_offsets(obs_offsets, win, at)

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

        if strict_win_tail and max(obs_offsets) != win - 1:
            raise ValueError(
                f"Event at {at}: max(obs_offsets) must be win-1 when strict_win_tail=True."
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


def build_event_R(
    obs_handler: UnifiedDynamicSparseObservationHandler,
    observation_variance: float | None,
) -> torch.Tensor:
    """Return per-time-step observation covariance (torchda applies it per time index)."""
    return obs_handler.create_block_R_matrix(base_variance=observation_variance)


def safe_denorm(x: torch.Tensor, dataset: CylinderDynamicsDataset) -> torch.Tensor:
    """Denormalize Cylinder tensors on CPU."""
    if isinstance(x, torch.Tensor):
        x_cpu = x.detach().cpu()
        mean = dataset.mean.reshape(1, -1, 1, 1)
        std = dataset.std.reshape(1, -1, 1, 1)
        return (x_cpu * std + mean).cpu()
    return x


def compute_metrics(
    da_states: torch.Tensor,
    noda_states: torch.Tensor,
    groundtruth: torch.Tensor,
    dataset: CylinderDynamicsDataset,
    start_offset: int = 1,
) -> Dict[str, np.ndarray]:
    """Compute per-step, per-channel metrics for one assimilation run."""
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
    sample_idx: int = 0,
    model_name: str = "AE",
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

    # Load forward model
    forward_model = CYLINDER_C_FORWARD().to(device)
    forward_model.load_state_dict(
        torch.load(
            f"../../../../results/{model_name}/Cylinder/4loss_model/forward_model.pt",
            weights_only=True,
            map_location=device,
        )
    )
    forward_model.eval()
    print("Forward model loaded.")

    # Load dataset and slice sequence
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

    total_frames = window_length + da_start_step
    if start_T + total_frames > cyl_val_dataset.data.shape[1]:
        raise ValueError("Requested window exceeds available cylinder sequence length.")

    raw_data = cyl_val_dataset.data[sample_idx, start_T : start_T + total_frames, ...]
    groundtruth = torch.tensor(raw_data, dtype=torch.float32)
    normalized_groundtruth = cyl_val_dataset.normalize(groundtruth)
    print(f"Ground truth slice shape: {groundtruth.shape}")

    print(f"Model will latent forward {da_start_step} step before DA")

    # Observations (fixed positions, fixed ratio) at specific steps
    obs_handler = UnifiedDynamicSparseObservationHandler(
        max_obs_ratio=obs_ratio,
        min_obs_ratio=obs_ratio,
        seed=42,
        noise_std=obs_noise_std,
        fixed_valid_mask=True,
    )
    sample_shape = normalized_groundtruth[1].shape
    obs_handler.generate_unified_observations(sample_shape, list(range(window_length)))

    latent_dim = int(forward_model.hidden_dim)
    B = torch.eye(latent_dim, device=device)
    executor = MLPDAExecutor(
        forward_model=forward_model,
        obs_handler=obs_handler,
        device=device,
        early_stop=early_stop_config,
    )

    run_metrics = {"mse": [], "rrmse": [], "ssim": []}
    run_times = []
    run_iterations = []
    first_run_states = None

    for run_idx in range(num_runs):
        print(f"\nStarting assimilation run {run_idx + 1}/{num_runs}")
        set_seed(42 + run_idx)

        sparse_observations = []
        for idx in range(window_length):
            sparse = obs_handler.apply_unified_observation(
                normalized_groundtruth[da_start_step + idx],
                idx,
                add_noise=True,
            )
            sparse_observations.append(sparse)
        sparse_observations = torch.stack(sparse_observations).to(device)

        da_states = []
        noda_states = []
        total_da_time = 0.0
        total_iterations = 0

        x0 = normalized_groundtruth[0].to(device).unsqueeze(0)
        z_background = forward_model.K_S(x0)  # z0
        for _ in range(da_start_step):
            z_background = forward_model.latent_forward(z_background)
        noda_background = z_background.clone()

        for step in range(window_length):
            if step in event_map:
                event = event_map[step]
                obs_offsets = event.obs_offsets
                obs_stack, observation_time_steps, gaps = build_obs_stack(
                    sparse_observations, step, obs_offsets
                )
                if obs_stack.shape[0] != len(obs_offsets):
                    raise RuntimeError(f"Observation stack length mismatch at step {step}.")
                R = build_event_R(
                    obs_handler=obs_handler,
                    observation_variance=observation_variance,
                ).to(device)
                background_state = z_background.ravel()

                z_assimilated, intermediates, elapsed = executor.assimilate_step(
                    observations=obs_stack,
                    background_state=background_state,
                    observation_time_idx=step,
                    observation_time_steps=observation_time_steps,
                    gaps=gaps,
                    B=B,
                    R=R,
                )

                total_da_time += elapsed
                if intermediates:
                    final_cost = intermediates.get("J", [None])[-1]
                    if final_cost is not None:
                        print(f"Step {step + 1}: final cost {final_cost}")
                    if "J" in intermediates:
                        total_iterations += len(intermediates["J"])

                z_assimilated = (
                    z_assimilated if z_assimilated.ndim > 1 else z_assimilated.unsqueeze(0)
                )
                decoded_assim = (
                    executor.decode_latent(z_assimilated).squeeze(0).detach().cpu()
                )
                da_states.append(decoded_assim)
                z_background = forward_model.latent_forward(z_assimilated)
            else:
                decoded_background = (
                    executor.decode_latent(z_background).squeeze(0).detach().cpu()
                )
                da_states.append(decoded_background)
                z_background = forward_model.latent_forward(z_background)

            noda_decoded = executor.decode_latent(noda_background).squeeze(0).detach().cpu()
            noda_states.append(noda_decoded)
            noda_background = forward_model.latent_forward(noda_background)

        da_stack = torch.stack(da_states)
        noda_stack = torch.stack(noda_states)

        if first_run_states is None:
            first_run_states = da_stack.clone()

        metrics = compute_metrics(
            da_stack,
            noda_stack,
            groundtruth,
            cyl_val_dataset,
            start_offset=da_start_step,
        )
        for key in run_metrics:
            run_metrics[key].append(metrics[key])

        run_times.append(total_da_time)
        run_iterations.append(total_iterations)
        print(f"Run {run_idx + 1} assimilation time: {total_da_time:.2f}s")

    save_dir = "../../../../results/AE/Cylinder/DA"
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
            safe_denorm(first_run_states, cyl_val_dataset).numpy(),
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
            f"{key.upper()} mean over runs: {overall_stats[f'{key}_mean']:.6f}, "
            f"std: {overall_stats[f'{key}_std']:.6f}"
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
