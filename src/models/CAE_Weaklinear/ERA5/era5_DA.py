"""
Multi-run data assimilation experiments for CAE_Weaklinear on ERA5.
"""

import os
import sys
from typing import Dict, Tuple

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

# Add src directory to path
current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
sys.path.append(src_directory)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_config")

from src.models.CAE_Weaklinear.ERA5.era5_model import ERA5_C_FORWARD
from src.models.CAE_Weaklinear.dabase import (
    UnifiedDynamicSparseObservationHandler,
    WeaklinearDAExecutor,
    set_device,
    set_seed,
)
from src.utils.Dataset import ERA5Dataset


def safe_denorm(x: torch.Tensor, dataset: ERA5Dataset) -> torch.Tensor:
    """Denormalize ERA5 tensors on CPU."""
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
    dataset: ERA5Dataset,
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


def run_multi_da_experiment(
    obs_ratio: float = 0.15,
    obs_noise_std: float = 0.05,
    observation_schedule: list = [0, 10, 20],
    observation_variance: float | None = None,
    window_length: int = 30,
    num_runs: int = 5,
    early_stop_config: Tuple[int, float] = (100, 1e-3),
    max_iterations: int = 5000,
    start_T: int = 0,
    da_start_step: int = 1,
    model_name: str = "CAE_Weaklinear",
    save_prefix: str | None = None,
):
    """Run repeated DA experiments and collect mean/std statistics."""
    assert 0 <= da_start_step <= window_length, "da_start_step must in [0, window_length]"
    set_seed(42)
    device = set_device()
    print(f"Using device: {device}")

    # Load forward model
    forward_model = ERA5_C_FORWARD().to(device)
    forward_model.load_state_dict(
        torch.load(
            f"../../../../results/{model_name}/ERA5/3loss_model/forward_model.pt",
            weights_only=True,
            map_location=device,
        )
    )
    forward_model.eval()
    print("Forward model loaded.")

    # Load dataset and slice sequence
    forward_step = 12
    dataset = ERA5Dataset(
        data_path="../../../../data/ERA5/ERA5_data/test_seq_state.h5",
        seq_length=forward_step,
        min_path="../../../../data/ERA5/ERA5_data/min_val.npy",
        max_path="../../../../data/ERA5/ERA5_data/max_val.npy",
    )

    total_frames = window_length + da_start_step + 1
    raw_data = dataset.data[start_T : start_T + total_frames, ...]
    groundtruth = torch.tensor(raw_data, dtype=torch.float32).permute(0, 3, 1, 2)
    normalized_groundtruth = dataset.normalize(groundtruth)
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

    obs_steps = sorted(observation_schedule)
    assert all(0 <= t < window_length for t in obs_steps), "observation_schedule out of bound"
    gaps = [obs_steps[i + 1] - obs_steps[i] for i in range(len(obs_steps) - 1)]
    if len(obs_steps) == 1:
        gaps = None

    latent_dim = int(forward_model.hidden_dim)
    B = torch.eye(latent_dim, device=device)
    R = obs_handler.create_block_R_matrix(base_variance=observation_variance).to(device)

    executor = WeaklinearDAExecutor(
        forward_model=forward_model,
        obs_handler=obs_handler,
        device=device,
        early_stop=early_stop_config,
        max_iterations=max_iterations,
    )

    run_metrics = {"mse": [], "rrmse": [], "ssim": []}
    run_times = []
    run_iterations = []
    first_run_states = None
    first_run_original_states = None

    for run_idx in range(num_runs):
        print(f"\nStarting assimilation run {run_idx + 1}/{num_runs}")
        set_seed(42 + run_idx)

        # ===== 1) 背景（t=1） =====
        x0 = normalized_groundtruth[0].to(device).unsqueeze(0)
        z = forward_model.K_S(x0)  # z0
        for _ in range(da_start_step):
            z = forward_model.latent_forward(z)  # rollout da_start_step 步
        z_start_background = z
        background_state = z_start_background.ravel()

        # ===== 2) 生成本 run 的观测序列（只取 schedule 时刻）=====
        obs_list = []

        assert max(obs_steps) < window_length, "observation_schedule out of bounds"
        assert (
            da_start_step + max(obs_steps) < normalized_groundtruth.shape[0]
        ), "Observation index out of bounds"

        for t in obs_steps:
            y_t = obs_handler.apply_unified_observation(
                normalized_groundtruth[da_start_step + t].to(device),
                t,
                add_noise=True,
            )
            obs_list.append(y_t)
        observations = torch.stack(obs_list).to(device)

        # ===== 3) 整窗 4D-Var 同化（一次）=====
        z_assimilated, intermediates, elapsed = executor.assimilate_step(
            observations=observations,
            background_state=background_state,
            observation_time_idx=0,
            observation_time_steps=obs_steps,
            gaps=gaps,
            B=B,
            R=R,
        )

        run_times.append(float(elapsed))
        if intermediates and "J" in intermediates:
            run_iterations.append(len(intermediates["J"]))
            print(f"Final cost: {intermediates['J'][-1]}  | iters: {len(intermediates['J'])}")
        else:
            run_iterations.append(0)

        # ===== 4) 用同化后的 z1 roll out 得到 window_length 帧 DA 轨迹 =====
        if z_assimilated.ndim == 1:
            z = z_assimilated.unsqueeze(0)
        else:
            z = z_assimilated

        da_states = []
        for step in range(window_length):
            da_states.append(executor.decode_latent(z).squeeze(0).detach().cpu())
            z = forward_model.latent_forward(z)

        # ===== 5) NoDA 基线：从背景 z1_background roll out =====
        z = z_start_background.clone()
        noda_states = []
        for step in range(window_length):
            noda_states.append(executor.decode_latent(z).squeeze(0).detach().cpu())
            z = forward_model.latent_forward(z)

        da_stack = torch.stack(da_states)
        noda_stack = torch.stack(noda_states)

        if first_run_states is None:
            first_run_states = da_stack.clone()
            first_run_original_states = noda_stack.clone()

        metrics = compute_metrics(da_stack, noda_stack, groundtruth, dataset, start_offset=da_start_step)
        for key in run_metrics:
            run_metrics[key].append(metrics[key])

        print(f"Run {run_idx + 1} assimilation time: {elapsed:.2f}s")

    save_dir = f"../../../../results/{model_name}/ERA5/DA"
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

    if first_run_original_states is not None:
        np.save(
            os.path.join(save_dir, prefixed("multi_original.npy")),
            safe_denorm(first_run_original_states, dataset).numpy(),
        )
        print(f"Saved sample NoDA trajectory to {os.path.join(save_dir, prefixed('multi_original.npy'))}")

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
        print(f"{key.upper()} mean over runs: {overall_stats[f'{key}_mean']:.6f}, std: {overall_stats[f'{key}_std']:.6f}")

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
    run_multi_da_experiment()
