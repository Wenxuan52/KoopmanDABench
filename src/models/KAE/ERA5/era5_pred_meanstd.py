import os
import sys
import random
from typing import Dict

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from era5_model import ERA5_C_FORWARD

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
    """Roll forward the Koopman model for n_steps starting from a normalized initial state."""
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
    # =========================================================
    #  NEW: predict 5, 10, 50 steps separately
    # =========================================================
    prediction_steps_list = [5, 10, 50]
    num_starts = 30
    start_min = 100
    seed = 42
    model_name = "KAE"
    model_dir = f"../../../../results/{model_name}/ERA5/3loss_model"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Dataset loading consistent normalization
    era5_test_dataset = ERA5Dataset(
        data_path="../../../../data/ERA5/ERA5_data/test_seq_state.h5",
        seq_length=12,
        min_path="../../../../data/ERA5/ERA5_data/min_val.npy",
        max_path="../../../../data/ERA5/ERA5_data/max_val.npy",
    )
    denorm = era5_test_dataset.denormalizer()
    raw_test_data = era5_test_dataset.data
    total_frames = raw_test_data.shape[0]

    forward_model = ERA5_C_FORWARD()
    forward_model.load_state_dict(
        torch.load(os.path.join(model_dir, "forward_model.pt"), weights_only=True, map_location="cpu")
    )
    forward_model.eval()

    save_dir = f"../../../../results/{model_name}/figures"
    os.makedirs(save_dir, exist_ok=True)

    channel_names = ["Geopotential", "Temperature", "Humidity", "Wind_u", "Wind_v"]

    # =========================================================
    #  Loop over 5, 10, 50
    # =========================================================
    for prediction_steps in prediction_steps_list:
        print(f"\n========== Evaluating {prediction_steps}-step rollouts ==========\n")

        # Pick start frames
        max_start = total_frames - (prediction_steps + 1)
        start_candidates = range(start_min, max_start)
        start_frames = random.sample(start_candidates, num_starts)
        print(f"Start frames: {start_frames}")

        metric_storage = {"mse": [], "ssim": []}

        for start_frame in tqdm(start_frames, desc=f"Rollout {prediction_steps} steps"):
            # Raw initial + future states
            initial_state = torch.tensor(raw_test_data[start_frame, ...], dtype=torch.float32).permute(2, 0, 1)
            groundtruth = torch.tensor(
                raw_test_data[start_frame + 1 : start_frame + 1 + prediction_steps, ...],
                dtype=torch.float32,
            ).permute(0, 3, 1, 2)

            # Normalize → rollout → de-normalize
            normalize_initial = era5_test_dataset.normalize(initial_state.unsqueeze(0))[0]
            rollout = rollout_prediction(forward_model, normalize_initial, prediction_steps)
            de_rollout = denorm(rollout)

            metrics = compute_channel_metrics(groundtruth, de_rollout)

            metric_storage["mse"].append(metrics["mse"])
            metric_storage["ssim"].append(metrics["ssim"])

        # Stack → shape: (num_starts, C, T)
        stacked_metrics = {k: np.stack(v, axis=0) for k, v in metric_storage.items()}

        # Mean over starts × steps
        channel_means = {k: np.mean(val, axis=(0, 2)) for k, val in stacked_metrics.items()}
        channel_stds = {k: np.std(val, axis=(0, 2)) for k, val in stacked_metrics.items()}

        # Save
        # save_path = os.path.join(save_dir, f"metrics_era5_forward_{prediction_steps}steps.npz")
        # np.savez(
        #     save_path,
        #     all_mse=stacked_metrics["mse"],
        #     all_ssim=stacked_metrics["ssim"],
        #     mse_mean_channel=channel_means["mse"],
        #     mse_std_channel=channel_stds["mse"],
        #     ssim_mean_channel=channel_means["ssim"],
        #     ssim_std_channel=channel_stds["ssim"],
        #     start_frames=np.array(start_frames),
        # )

        # Print summary
        print(f"\nPer-channel overall metrics for {prediction_steps} steps (mean ± std):")
        for idx, name in enumerate(channel_names):
            print(
                f"{name:<12} | "
                f"MSE={channel_means['mse'][idx]:.6e}±{channel_stds['mse'][idx]:.2e}, "
                f"SSIM={channel_means['ssim'][idx]:.6f}±{channel_stds['ssim'][idx]:.3f}"
            )

        # print(f"\nSaved: {save_path}")
