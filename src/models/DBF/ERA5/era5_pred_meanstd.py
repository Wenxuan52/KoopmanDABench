import os
import sys
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# Register repo root
current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
if src_directory not in sys.path:
    sys.path.append(src_directory)

from src.utils.Dataset import ERA5Dataset
from src.models.DBF.ERA5.era5_model import ERA5Decoder
from src.models.DBF.ERA5.era5_trainer import (
    ObservationMask,
    ERA5IOONetwork,
    SpectralKoopmanOperator,
)


def compute_channel_metrics(gt: torch.Tensor, pred: torch.Tensor) -> Dict[str, np.ndarray]:
    """
    Compute per-channel metric sequences for tensors shaped [T, C, H, W].
    Returns numpy arrays with shape (C, T) per metric.
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


def initial_latent_state(batch_size: int, num_pairs: int, device: torch.device, init_cov: float):
    mean = torch.zeros(batch_size, num_pairs, 2, device=device)
    base_cov = torch.eye(2, device=device).view(1, 1, 2, 2)
    cov = init_cov * base_cov.expand(batch_size, num_pairs, 2, 2).clone()
    return mean, cov


def gaussian_update(mu_prior, cov_prior, mu_obs, cov_obs, epsilon: float):
    eye = torch.eye(2, device=mu_prior.device).view(1, 1, 2, 2)
    cov_prior = cov_prior + epsilon * eye
    cov_obs = cov_obs + epsilon * eye
    cov_prior_inv = torch.linalg.inv(cov_prior)
    cov_obs_inv = torch.linalg.inv(cov_obs)
    cov_post_inv = cov_prior_inv + cov_obs_inv
    cov_post = torch.linalg.inv(cov_post_inv)
    fused_mean = torch.matmul(cov_prior_inv, mu_prior.unsqueeze(-1)) + torch.matmul(cov_obs_inv, mu_obs.unsqueeze(-1))
    mu_post = torch.matmul(cov_post, fused_mean).squeeze(-1)
    cov_post = 0.5 * (cov_post + cov_post.transpose(-1, -2))
    return mu_post, cov_post


def decode_pairs_sequence(decoder: ERA5Decoder, latent_pairs: torch.Tensor) -> torch.Tensor:
    """
    Convert latent pair sequence [B, T, num_pairs, 2] to field sequence via ERA5 decoder.
    """
    b, t, num_pairs, _ = latent_pairs.shape
    latent_flat = latent_pairs.reshape(b, t, num_pairs * 2)
    frames = []
    for step in range(t):
        frame = decoder.decoder(latent_flat[:, step, :])
        frames.append(frame.unsqueeze(1))
    return torch.cat(frames, dim=1)


@torch.no_grad()
def run_dbf_rollout(
    sequence: torch.Tensor,
    observation_mask: ObservationMask,
    ioo: ERA5IOONetwork,
    koopman: SpectralKoopmanOperator,
    decoder: ERA5Decoder,
    init_cov: float,
    cov_epsilon: float,
) -> torch.Tensor:
    """
    Perform DBF rollout (prediction only, no DA updates after initial frame).
    sequence: [T, C, H, W] normalized tensor.
    Returns rollout frames [T, C, H, W] in normalized space.
    """
    device = next(decoder.parameters()).device
    seq = sequence.unsqueeze(0).to(device)  # [1, T, C, H, W]
    obs_all = observation_mask.sample(seq)  # [1, T, obs_dim]

    batch_size, time_steps = seq.shape[0], seq.shape[1]
    num_pairs = koopman.num_pairs

    mu_prev, cov_prev = initial_latent_state(batch_size, num_pairs, device, init_cov)

    mu_obs_flat, sigma2_obs_flat = ioo(obs_all[:, 0, :])
    mu_obs_pairs = mu_obs_flat.view(batch_size, num_pairs, 2)
    cov_obs = torch.diag_embed(sigma2_obs_flat.view(batch_size, num_pairs, 2))
    mu_first, cov_first = gaussian_update(mu_prev, cov_prev, mu_obs_pairs, cov_obs, cov_epsilon)

    rollout_pairs = [mu_first]
    mu_roll_prev, cov_roll_prev = mu_first, cov_first

    for t in range(1, time_steps):
        mu_prior_roll, cov_prior_roll = koopman.predict(mu_roll_prev, cov_roll_prev)
        mu_roll_post, cov_roll_post = mu_prior_roll, cov_prior_roll
        rollout_pairs.append(mu_roll_post)
        mu_roll_prev, cov_roll_prev = mu_roll_post, cov_roll_post

    rollout_pairs = torch.stack(rollout_pairs, dim=1)  # [1, T, num_pairs, 2]
    rollout_frames = decode_pairs_sequence(decoder, rollout_pairs)
    return rollout_frames.squeeze(0)


if __name__ == "__main__":
    prediction_steps = 50
    num_starts = 30
    start_min = 100
    seed = 42

    model_dir = Path("../../../../results/DBF/ERA5")
    ckpt_path = model_dir / "best_model.pt"
    save_dir = Path("../../../../results/DBF/figures")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    cfg = checkpoint.get("config", {})
    latent_dim = int(cfg.get("latent_dim", 512))
    init_cov = float(cfg.get("init_cov", 1.0))
    cov_epsilon = float(cfg.get("cov_epsilon", 1e-6))
    rho_clip = float(cfg.get("rho_clip", 0.2))
    process_noise_init = float(cfg.get("process_noise_init", -2.0))
    hidden_dims = cfg.get("ioo_hidden_dims", [1024, 1024])

    observation_mask = ObservationMask(checkpoint["mask"].bool()).to(device)

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

    era5_test_dataset = ERA5Dataset(
        data_path="../../../../data/ERA5/ERA5_data/test_seq_state.h5",
        seq_length=12,
        min_path="../../../../data/ERA5/ERA5_data/min_val.npy",
        max_path="../../../../data/ERA5/ERA5_data/max_val.npy",
    )
    denorm = era5_test_dataset.denormalizer()
    raw_test_data = era5_test_dataset.data  # [N, H, W, C]
    total_frames = raw_test_data.shape[0]

    max_start = total_frames - (prediction_steps + 1)
    if max_start <= start_min:
        raise ValueError("Not enough frames to sample the requested rollouts with the specified start_min.")

    start_candidates = range(start_min, max_start)
    if len(start_candidates) < num_starts:
        raise ValueError(f"Cannot sample {num_starts} unique start frames from range({start_min}, {max_start}).")

    start_frames = random.sample(start_candidates, num_starts)
    print(f"Selected start frames ({num_starts} total): {start_frames}")

    metric_storage = {"mse": [], "rrmse": [], "ssim": []}

    for start_frame in tqdm(start_frames, desc="Evaluating DBF ERA5 rollouts"):
        groundtruth = torch.tensor(
            raw_test_data[start_frame + 1 : start_frame + 1 + prediction_steps, ...],
            dtype=torch.float32,
        ).permute(0, 3, 1, 2)

        sequence = torch.tensor(
            raw_test_data[start_frame : start_frame + prediction_steps, ...],
            dtype=torch.float32,
        ).permute(0, 3, 1, 2)
        normalized_seq = era5_test_dataset.normalize(sequence)

        rollout_norm = run_dbf_rollout(
            normalized_seq,
            observation_mask,
            ioo,
            koopman,
            decoder,
            init_cov,
            cov_epsilon,
        ).cpu()
        de_rollout = denorm(rollout_norm)

        metrics = compute_channel_metrics(groundtruth, de_rollout)
        for key in metric_storage:
            metric_storage[key].append(metrics[key])

    stacked_metrics = {k: np.stack(v, axis=0) for k, v in metric_storage.items()}  # (num_starts, C, T)

    channel_step_means = {k: np.mean(val, axis=0) for k, val in stacked_metrics.items()}
    channel_step_stds = {k: np.std(val, axis=0) for k, val in stacked_metrics.items()}

    channel_means = {k: np.mean(val, axis=(0, 2)) for k, val in stacked_metrics.items()}  # (C,)
    channel_stds = {k: np.std(val, axis=(0, 2)) for k, val in stacked_metrics.items()}

    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "metrics_era5_forward.npz"

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
