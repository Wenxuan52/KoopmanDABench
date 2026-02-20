"""Multi-run data assimilation experiments for DMD on Kolmogorov Flow."""

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

from src.models.DMD.base import TorchDMD
from src.models.DMD.dabase import (
    DMDDAExecutor,
    UnifiedDynamicSparseObservationHandler,
    set_device,
    set_seed,
)
from src.utils.Dataset import KolDynamicsDataset


def safe_denorm(x: torch.Tensor, dataset: KolDynamicsDataset) -> torch.Tensor:
    """Denormalize Kolmogorov tensors on CPU. Keep same ndim as input."""
    if not isinstance(x, torch.Tensor):
        return x

    x_cpu = x.detach().cpu()
    mean = torch.as_tensor(dataset.mean, dtype=x_cpu.dtype, device=x_cpu.device)
    std  = torch.as_tensor(dataset.std,  dtype=x_cpu.dtype, device=x_cpu.device)

    if x_cpu.ndim == 4:          # (B,C,H,W)
        mean = mean.view(1, -1, 1, 1)
        std  = std.view(1, -1, 1, 1)
    elif x_cpu.ndim == 3:        # (C,H,W)
        mean = mean.view(-1, 1, 1)
        std  = std.view(-1, 1, 1)
    else:
        raise ValueError(f"safe_denorm expects 3D or 4D tensor, got shape {tuple(x_cpu.shape)}")

    return x_cpu * std + mean


def compute_metrics(
    da_states: torch.Tensor,
    noda_states: torch.Tensor,
    groundtruth: torch.Tensor,
    dataset: KolDynamicsDataset,
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
    sample_idx: int = 0,
    model_name: str = "DMD",
    svd_rank: int | None = None,
    save_prefix: str | None = None,
):
    """Run repeated DA experiments and collect mean/std statistics."""
    assert 0 <= da_start_step <= window_length, "da_start_step must in [0, window_length]"
    set_seed(42)
    device = set_device()
    print(f"Using device: {device}")

    # Load DMD model
    dmd_model = TorchDMD(device=device)
    dmd_model.load_dmd(f"../../../../results/{model_name}/KMG/dmd_model.pth")
    if svd_rank is not None:
        print(f"Overriding DMD rank to {svd_rank} (original {dmd_model.svd_rank})")
        dmd_model.svd_rank = svd_rank
    latent_dim = dmd_model.modes.shape[1]
    print(f"DMD model loaded with latent dimension {latent_dim}")

    # Load dataset and slice sequence
    forward_step = 12
    kol_train_dataset = KolDynamicsDataset(
        data_path="../../../../data/kol/kolmogorov_train_data.npy",
        seq_length=forward_step,
        mean=None,
        std=None,
    )
    kol_val_dataset = KolDynamicsDataset(
        data_path="../../../../data/kol/kolmogorov_val_data.npy",
        seq_length=forward_step,
        mean=kol_train_dataset.mean,
        std=kol_train_dataset.std,
    )

    total_frames = window_length + da_start_step + 1
    if start_T + total_frames > kol_val_dataset.data.shape[1]:
        raise ValueError("Requested window exceeds available Kolmogorov sequence length.")

    raw_data = kol_val_dataset.data[sample_idx, start_T : start_T + total_frames, ...]
    groundtruth = torch.tensor(raw_data, dtype=torch.float32)
    normalized_groundtruth = kol_val_dataset.normalize(groundtruth)
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

    B = torch.eye(2 * latent_dim, device=device)
    R = obs_handler.create_block_R_matrix(base_variance=observation_variance).to(device)

    executor = DMDDAExecutor(
        dmd_model=dmd_model,
        obs_handler=obs_handler,
        device=device,
        image_shape=sample_shape,
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
        x0 = normalized_groundtruth[0].unsqueeze(0).to(device)

        print("x0 shape:", normalized_groundtruth[0].shape)
        print("numel:", normalized_groundtruth[0].numel())
        print("modes shape:", dmd_model.modes.shape)

        b = executor.encode_state(x0)
        for _ in range(da_start_step):
            b = executor.latent_forward(b).squeeze(0)
        b_start_background = b
        background_state = executor.complex_to_real(b_start_background).squeeze()

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
        b_current = executor.real_to_complex(z_assimilated).squeeze(0)
        da_states = []
        for step in range(window_length):
            da_states.append(executor.decode_latent(b_current).squeeze(0).detach().cpu())
            b_current = executor.latent_forward(b_current).squeeze(0)

        # ===== 5) NoDA 基线：从背景 z1_background roll out =====
        b_current = b_start_background.clone()
        noda_states = []
        for step in range(window_length):
            noda_states.append(executor.decode_latent(b_current).squeeze(0).detach().cpu())
            b_current = executor.latent_forward(b_current).squeeze(0)

        da_stack = torch.stack(da_states)
        noda_stack = torch.stack(noda_states)

        if first_run_states is None:
            first_run_states = da_stack.clone()
            first_run_original_states = noda_stack.clone()

        metrics = compute_metrics(
            da_stack, noda_stack, groundtruth, kol_val_dataset, start_offset=da_start_step
        )
        for key in run_metrics:
            run_metrics[key].append(metrics[key])

        print(f"Run {run_idx + 1} assimilation time: {elapsed:.2f}s")

    save_dir = f"../../../../results/{model_name}/KMG/DA"
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
            safe_denorm(first_run_states, kol_val_dataset).numpy(),
        )
        print(f"Saved sample DA trajectory to {os.path.join(save_dir, prefixed('multi.npy'))}")

    if first_run_original_states is not None:
        np.save(
            os.path.join(save_dir, prefixed("multi_original.npy")),
            safe_denorm(first_run_original_states, kol_val_dataset).numpy(),
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
    run_multi_da_experiment()
