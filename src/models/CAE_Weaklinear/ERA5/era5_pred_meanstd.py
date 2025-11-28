import os
import sys
import random
from typing import Dict

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from era5_model_FTF import ERA5_C_FORWARD

# Register project root
current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
sys.path.append(src_directory)

from src.utils.Dataset import ERA5Dataset


def compute_channel_metrics(gt: torch.Tensor, pred: torch.Tensor) -> Dict[str, np.ndarray]:
    """
    Compute per-channel metric sequences.
    gt, pred: tensors of shape (T, C, H, W)
    Returns numpy arrays with shape (C, T) for each metric.
    """
    gt_np = gt.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()

    num_steps, num_channels = gt_np.shape[0], gt_np.shape[1]
    mse = np.zeros((num_channels, num_steps), dtype=np.float64)
    rrmse = np.zeros_like(mse)
    ssim_vals = np.zeros_like(mse)

    for ch in range(num_channels):
        for t in range(num_steps):
            g = gt_np[t, ch]
            p = pred_np[t, ch]

            mse_val = np.mean((p - g) ** 2)
            mse[ch, t] = mse_val

            denom = np.sqrt(np.mean(g ** 2)) + 1e-8
            rrmse[ch, t] = np.sqrt(mse_val) / denom

            data_range = g.max() - g.min()
            if data_range < 1e-12:
                data_range = 1.0
            ssim_vals[ch, t] = ssim(g, p, data_range=data_range, channel_axis=None)

    return {"mse": mse, "rrmse": rrmse, "ssim": ssim_vals}


def rollout_prediction(forward_model: ERA5_C_FORWARD, initial_state: torch.Tensor, n_steps: int) -> torch.Tensor:
    """
    Roll forward the Koopman model for n_steps starting from a normalized initial state.
    initial_state: (C, H, W)
    returns tensor of shape (n_steps, C, H, W)
    """
    predictions = []
    current_state = initial_state.unsqueeze(0)

    with torch.no_grad():
        z_current = forward_model.K_S(current_state)
        for _ in range(n_steps):
            z_next = forward_model.latent_forward(z_current)
            next_state = forward_model.K_S_preimage(z_next)
            predictions.append(next_state)
            z_current = z_next

    return torch.cat(predictions, dim=0)


if __name__ == "__main__":
    prediction_steps = 50
    num_starts = 30
    start_min = 100
    seed = 42
    model_name = "CAE_Weaklinear"
    model_dir = f"../../../../results/{model_name}/ERA5/3loss_model"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Dataset loading mirrors era5_predictor for consistent normalization
    era5_test_dataset = ERA5Dataset(
        data_path="../../../../data/ERA5/ERA5_data/test_seq_state.h5",
        seq_length=12,
        min_path="../../../../data/ERA5/ERA5_data/min_val.npy",
        max_path="../../../../data/ERA5/ERA5_data/max_val.npy",
    )
    denorm = era5_test_dataset.denormalizer()
    raw_test_data = era5_test_dataset.data  # shape: (N, H, W, C)
    total_frames = raw_test_data.shape[0]

    max_start = total_frames - (prediction_steps + 1)
    if max_start <= start_min:
        raise ValueError("Not enough frames to sample the requested rollouts with the specified start_min.")

    start_candidates = range(start_min, max_start)
    if len(start_candidates) < num_starts:
        raise ValueError(f"Cannot sample {num_starts} unique start frames from range({start_min}, {max_start}).")

    start_frames = random.sample(start_candidates, num_starts)
    print(f"Selected start frames ({num_starts} total): {start_frames}")

    forward_model = ERA5_C_FORWARD()
    forward_model.load_state_dict(
        torch.load(os.path.join(model_dir, "forward_model.pt"), weights_only=True, map_location="cpu")
    )
    forward_model.eval()

    metric_storage = {"mse": [], "rrmse": [], "ssim": []}

    for start_frame in tqdm(start_frames, desc="Evaluating ERA5 rollouts"):
        # Initial state and ground truth are taken from raw (unnormalized) data
        initial_state = torch.tensor(raw_test_data[start_frame, ...], dtype=torch.float32).permute(2, 0, 1)
        groundtruth = torch.tensor(
            raw_test_data[start_frame + 1 : start_frame + 1 + prediction_steps, ...],
            dtype=torch.float32,
        ).permute(0, 3, 1, 2)

        normalize_initial = era5_test_dataset.normalize(initial_state.unsqueeze(0))[0]
        rollout = rollout_prediction(forward_model, normalize_initial, prediction_steps)
        de_rollout = denorm(rollout)

        metrics = compute_channel_metrics(groundtruth, de_rollout)
        for key in metric_storage:
            metric_storage[key].append(metrics[key])

    stacked_metrics = {k: np.stack(v, axis=0) for k, v in metric_storage.items()}  # (num_starts, C, T)

    channel_step_means = {k: np.mean(val, axis=0) for k, val in stacked_metrics.items()}
    channel_step_stds = {k: np.std(val, axis=0) for k, val in stacked_metrics.items()}

    channel_means = {k: np.mean(val, axis=(0, 2)) for k, val in stacked_metrics.items()}  # (C,)
    channel_stds = {k: np.std(val, axis=(0, 2)) for k, val in stacked_metrics.items()}

    save_dir = f"../../../../results/{model_name}/figures"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "metrics_era5_forward.npz")

    np.savez(
        save_path,
        all_mse=stacked_metrics["mse"],
        all_rrmse=stacked_metrics["rrmse"],
        all_ssim=stacked_metrics["ssim"],
        mse_mean_channel_step=channel_step_means["mse"],
        mse_std_channel_step=channel_step_stds["mse"],
        rrmse_mean_channel_step=channel_step_means["rrmse"],
        rrmse_std_channel_step=channel_step_stds["rrmse"],
        ssim_mean_channel_step=channel_step_means["ssim"],
        ssim_std_channel_step=channel_step_stds["ssim"],
        mse_mean_channel=channel_means["mse"],
        mse_std_channel=channel_stds["mse"],
        rrmse_mean_channel=channel_means["rrmse"],
        rrmse_std_channel=channel_stds["rrmse"],
        ssim_mean_channel=channel_means["ssim"],
        ssim_std_channel=channel_stds["ssim"],
        start_frames=np.array(start_frames),
    )

    channel_names = ["Geopotential", "Temperature", "Humidity", "Wind_u", "Wind_v"]
    print("\nPer-channel overall metrics (mean ± std across rollouts and steps):")
    for idx, name in enumerate(channel_names):
        print(
            f"{name:<12} | "
            f"MSE={channel_means['mse'][idx]:.6e}±{channel_stds['mse'][idx]:.2e}, "
            f"RRMSE={channel_means['rrmse'][idx]:.6e}±{channel_stds['rrmse'][idx]:.2e}, "
            f"SSIM={channel_means['ssim'][idx]:.6f}±{channel_stds['ssim'][idx]:.3f}"
        )

    print(f"\nSaved metrics to: {save_path}")
