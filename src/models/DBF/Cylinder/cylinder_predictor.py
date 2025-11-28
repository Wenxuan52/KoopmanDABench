import argparse
import os
import sys
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
sys.path.append(src_directory)

from src.utils.Dataset import CylinderDynamicsDataset
from src.models.DBF.Cylinder.cylinder_model import CylinderDecoder
from src.models.DBF.Cylinder.cylinder_trainer import (
    ObservationMask,
    CylinderIOONetwork,
    SpectralKoopmanOperator,
)


def set_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def gaussian_update(mu_prior, cov_prior, mu_obs, cov_obs, epsilon: float):
    eye = torch.eye(2, device=mu_prior.device).view(1, 1, 2, 2)
    cov_prior = cov_prior + epsilon * eye
    cov_obs = cov_obs + epsilon * eye
    cov_prior_inv = torch.linalg.inv(cov_prior)
    cov_obs_inv = torch.linalg.inv(cov_obs)
    cov_post_inv = cov_prior_inv + cov_obs_inv
    cov_post = torch.linalg.inv(cov_post_inv)
    fused_mean = (
        torch.matmul(cov_prior_inv, mu_prior.unsqueeze(-1))
        + torch.matmul(cov_obs_inv, mu_obs.unsqueeze(-1))
    )
    mu_post = torch.matmul(cov_post, fused_mean).squeeze(-1)
    cov_post = 0.5 * (cov_post + cov_post.transpose(-1, -2))
    return mu_post, cov_post


def initial_latent_state(batch_size: int, num_pairs: int, device: torch.device, init_cov: float):
    mean = torch.zeros(batch_size, num_pairs, 2, device=device)
    base_cov = torch.eye(2, device=device).view(1, 1, 2, 2)
    cov = init_cov * base_cov.expand(batch_size, num_pairs, 2, 2).clone()
    return mean, cov


def decode_pairs_sequence(decoder: CylinderDecoder, latent_pairs: torch.Tensor) -> torch.Tensor:
    # latent_pairs: [B, T, num_pairs, 2]
    b, t, num_pairs, _ = latent_pairs.shape
    latent_flat = latent_pairs.reshape(b, t, num_pairs * 2)
    frames = []
    for step in range(t):
        frame = decoder.decoder(latent_flat[:, step, :])
        frames.append(frame.unsqueeze(1))
    return torch.cat(frames, dim=1)


def velocity_magnitude(field: torch.Tensor) -> torch.Tensor:
    """Compute velocity magnitude from 2-channel field -> shape [T,1,H,W]."""
    if field.dim() != 4:
        raise ValueError("Expected field tensor with shape [T,C,H,W]")
    if field.size(1) == 1:
        return field.abs()
    magnitude = field.pow(2).sum(dim=1, keepdim=True).sqrt()
    return magnitude


@torch.no_grad()
def run_dbf_sequence(
    sequence: torch.Tensor,
    observation_mask: ObservationMask,
    ioo: CylinderIOONetwork,
    koopman: SpectralKoopmanOperator,
    decoder: CylinderDecoder,
    init_cov: float,
    cov_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return rollout (prior) and DA (filtered) reconstructions for a single sequence."""
    device = next(decoder.parameters()).device
    seq = sequence.unsqueeze(0).to(device)  # [1, T, C, H, W]
    obs_all = observation_mask.sample(seq)

    batch_size, time_steps = seq.shape[0], seq.shape[1]
    num_pairs = koopman.num_pairs

    mu_prev, cov_prev = initial_latent_state(batch_size, num_pairs, device, init_cov)

    # Assimilate the first frame to initialise state
    mu_obs_flat, sigma2_obs_flat = ioo(obs_all[:, 0, :])
    mu_obs_pairs = mu_obs_flat.view(batch_size, num_pairs, 2)
    cov_obs = torch.diag_embed(sigma2_obs_flat.view(batch_size, num_pairs, 2))
    mu_first, cov_first = gaussian_update(mu_prev, cov_prev, mu_obs_pairs, cov_obs, cov_epsilon)

    rollout_pairs = [mu_first]
    rollout_covs = [cov_first]

    da_pairs = [mu_first]
    da_covs = [cov_first]

    mu_roll_prev, cov_roll_prev = mu_first, cov_first
    mu_da_prev, cov_da_prev = mu_first, cov_first

    for t in range(1, time_steps):
        mu_prior_roll, cov_prior_roll = koopman.predict(mu_roll_prev, cov_roll_prev)
        mu_prior_da, cov_prior_da = koopman.predict(mu_da_prev, cov_da_prev)

        # Rollout: no update with observations
        mu_roll_post, cov_roll_post = mu_prior_roll, cov_prior_roll

        # Data assimilation: fuse with current observation
        mu_obs_flat, sigma2_obs_flat = ioo(obs_all[:, t, :])
        mu_obs_pairs = mu_obs_flat.view(batch_size, num_pairs, 2)
        cov_obs = torch.diag_embed(sigma2_obs_flat.view(batch_size, num_pairs, 2))
        mu_da_post, cov_da_post = gaussian_update(
            mu_prior_da, cov_prior_da, mu_obs_pairs, cov_obs, cov_epsilon
        )

        rollout_pairs.append(mu_roll_post)
        rollout_covs.append(cov_roll_post)

        da_pairs.append(mu_da_post)
        da_covs.append(cov_da_post)

        mu_roll_prev, cov_roll_prev = mu_roll_post, cov_roll_post
        mu_da_prev, cov_da_prev = mu_da_post, cov_da_post

    rollout_pairs = torch.stack(rollout_pairs, dim=1)
    da_pairs = torch.stack(da_pairs, dim=1)

    rollout_frames = decode_pairs_sequence(decoder, rollout_pairs)
    da_frames = decode_pairs_sequence(decoder, da_pairs)

    return rollout_frames.squeeze(0), da_frames.squeeze(0)


def plot_sequence_with_error(
    groundtruth: np.ndarray,
    prediction: np.ndarray,
    steps: Sequence[int],
    save_path: str,
    title: str,
    time_labels: Sequence[int],
):
    channels = groundtruth.shape[1]
    num_cols = len(steps) * channels
    fig, axes = plt.subplots(3, num_cols, figsize=(4 * len(steps) * channels, 9))
    if num_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    vmin = min(groundtruth.min(), prediction.min())
    vmax = max(groundtruth.max(), prediction.max())
    error = np.abs(prediction - groundtruth)
    err_max = error.max() if error.size > 0 else 1.0

    for col_idx, step in enumerate(steps):
        for ch in range(channels):
            col = col_idx * channels + ch
            gt_img = groundtruth[step, ch]
            pred_img = prediction[step, ch]
            err_img = error[step, ch]

            ax_gt = axes[0, col]
            im_gt = ax_gt.imshow(gt_img, cmap="viridis", vmin=vmin, vmax=vmax)
            ax_gt.set_axis_off()
            if col == 0:
                ax_gt.set_ylabel("Ground Truth", fontsize=12)
            if ch == 0:
                ax_gt.set_title(f"t={time_labels[col_idx]}", fontsize=13)

            ax_pred = axes[1, col]
            im_pred = ax_pred.imshow(pred_img, cmap="viridis", vmin=vmin, vmax=vmax)
            ax_pred.set_axis_off()
            if col == 0:
                ax_pred.set_ylabel("Prediction", fontsize=12)

            ax_err = axes[2, col]
            im_err = ax_err.imshow(err_img, cmap="magma", vmin=0.0, vmax=err_max)
            ax_err.set_axis_off()
            if col == 0:
                ax_err.set_ylabel("|Error|", fontsize=12)

            axes[2, col].set_xlabel(f"t={time_labels[col_idx]}", fontsize=11)

            fig.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)
            fig.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)
            fig.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_mse_curve(
    groundtruth: torch.Tensor,
    rollout: torch.Tensor,
    da: torch.Tensor,
    save_path: str,
    start_step: int,
):
    gt_flat = groundtruth.view(groundtruth.shape[0], -1)
    rollout_flat = rollout.view(rollout.shape[0], -1)
    da_flat = da.view(da.shape[0], -1)

    mse_roll = torch.mean((rollout_flat - gt_flat) ** 2, dim=1).cpu().numpy()
    mse_da = torch.mean((da_flat - gt_flat) ** 2, dim=1).cpu().numpy()

    steps = start_step + np.arange(groundtruth.shape[0])

    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, mse_roll, label="Rollout MSE", linewidth=2)
    plt.plot(steps, mse_da, label="DA MSE", linewidth=2)
    plt.xlabel("Time Step")
    plt.ylabel("MSE")
    plt.title("MSE per Prediction Step")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cylinder DBF Predictor")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="../../../../results/DBF/Cylinder/best_model.pt",
        help="Path to trained DBF checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../../../../results/DBF/Cylinder/figures",
        help="Directory to save figures",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=3,
        help="Dataset sample index (default: 3)",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=700,
        help="Starting timestep within the sequence (default: 700)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=30,
        help="Number of prediction steps to evaluate (default: 30)",
    )
    parser.add_argument(
        "--plot-steps",
        type=str,
        default="1,5,10,15,25,29",
        help="Comma separated list of timesteps to visualise (relative or absolute)",
    )
    args = parser.parse_args()

    device = set_device()
    print(f"[INFO] Using device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    cfg = checkpoint.get("config", {})
    latent_dim = int(cfg.get("latent_dim", 512))
    seq_length = int(cfg.get("seq_length", 64))
    data_path = cfg.get(
        "val_data", "../../../../data/cylinder/cylinder_val_data.npy"
    )
    init_cov = float(cfg.get("init_cov", 1.0))
    cov_epsilon = float(cfg.get("cov_epsilon", 1e-6))

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = CylinderDynamicsDataset(
        data_path=data_path,
        seq_length=seq_length,
        mean=checkpoint["train_mean"],
        std=checkpoint["train_std"],
    )

    observation_mask = ObservationMask(checkpoint["mask"].bool()).to(device)

    decoder = CylinderDecoder(dim_z=latent_dim).to(device)
    decoder.load_state_dict(checkpoint["decoder"])
    decoder.eval()

    ioo = CylinderIOONetwork(
        obs_dim=observation_mask.obs_dim,
        latent_dim=latent_dim,
        hidden_dims=cfg.get("ioo_hidden_dims", [1024, 1024]),
    ).to(device)
    ioo.load_state_dict(checkpoint["ioo"])
    ioo.eval()

    koopman = SpectralKoopmanOperator(
        latent_dim=latent_dim,
        rho_clip=float(cfg.get("rho_clip", 0.2)),
        process_noise_init=float(cfg.get("process_noise_init", -2.0)),
    ).to(device)
    koopman.load_state_dict(checkpoint["koopman"])
    koopman.eval()

    sample_idx = max(0, min(args.sample_index, dataset.n_samples - 1))
    max_frames = dataset.frames
    start_step = max(0, min(args.start_step, max_frames - 1))
    horizon = max(1, args.horizon)
    if start_step + horizon > max_frames:
        horizon = max_frames - start_step
    print(
        f"[INFO] Using sample index: {sample_idx}, start timestep: {start_step}, horizon: {horizon}"
    )

    raw_sequence = torch.tensor(
        dataset.data[sample_idx, start_step : start_step + horizon], dtype=torch.float32
    )
    sequence = dataset.normalize(raw_sequence.clone())

    plot_tokens = [token.strip() for token in args.plot_steps.split(",") if token.strip()]
    if plot_tokens:
        plot_tokens = [int(token) for token in plot_tokens]
    else:
        plot_tokens = [1, 5, 10, 15, 25, 29]

    if max(plot_tokens) >= horizon:
        plot_steps = [max(0, min(token - start_step, horizon - 1)) for token in plot_tokens]
    else:
        plot_steps = [max(0, min(token, horizon - 1)) for token in plot_tokens]

    unique_steps = []
    time_labels = []
    for rel_step in plot_steps:
        if rel_step not in unique_steps:
            unique_steps.append(rel_step)
            time_labels.append(start_step + rel_step)
    plot_steps = unique_steps

    with torch.no_grad():
        rollout_norm, da_norm = run_dbf_sequence(
            sequence, observation_mask, ioo, koopman, decoder, init_cov, cov_epsilon
        )

    denormalize = dataset.denormalizer()
    gt_denorm = denormalize(sequence.cpu()).cpu()
    rollout_denorm = denormalize(rollout_norm.cpu()).cpu()
    da_denorm = denormalize(da_norm.cpu()).cpu()

    gt_mag = velocity_magnitude(gt_denorm)
    rollout_mag = velocity_magnitude(rollout_denorm)
    da_mag = velocity_magnitude(da_denorm)

    print(f"[INFO] Plotting steps (relative): {plot_steps}")
    print(f"[INFO] Plotting timesteps (absolute): {time_labels}")

    rollout_path = os.path.join(args.output_dir, "rollout_vs_groundtruth.png")
    plot_sequence_with_error(
        groundtruth=gt_mag.numpy(),
        prediction=rollout_mag.numpy(),
        steps=plot_steps,
        save_path=rollout_path,
        title="Rollout vs Ground Truth",
        time_labels=time_labels,
    )
    print(f"[INFO] Saved rollout comparison to {rollout_path}")

    da_path = os.path.join(args.output_dir, "da_vs_groundtruth.png")
    plot_sequence_with_error(
        groundtruth=gt_mag.numpy(),
        prediction=da_mag.numpy(),
        steps=plot_steps,
        save_path=da_path,
        title="Data Assimilation vs Ground Truth",
        time_labels=time_labels,
    )
    print(f"[INFO] Saved DA comparison to {da_path}")

    mse_curve_path = os.path.join(args.output_dir, "mse_rollout_vs_da.png")
    plot_mse_curve(gt_mag, rollout_mag, da_mag, mse_curve_path, start_step)
    print(f"[INFO] Saved MSE curve to {mse_curve_path}")


if __name__ == "__main__":
    main()
