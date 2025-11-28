"""CGKN Cylinder predictor and visualisation utilities.

This script mirrors the inference pipeline used in training and evaluation:
- load the Stage 2 encoder/decoder/CGN weights,
- run latent-space rollout using the learned ODE,
- run the analytic CGFilter data-assimilation step,
- denormalise predictions and export comparison figures.

Usage (defaults provide a reasonable demo run):

python cylinder_predictor.py \
    --val-index 3 \
    --start-frame 700 \
    --horizon 30

Figures are written to ``results/CGKN/figures``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchdiffeq import odeint


# ---------------------------------------------------------------------------
# Repo imports (dataset, models, CGKN components)
# ---------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.utils.Dataset import CylinderDynamicsDataset  # noqa: E402
from cylinder_model import CylinderEncoder, CylinderDecoder  # noqa: E402
from cylinder_train import (  # noqa: E402
    CGFilter,
    CGKN_ODE,
    CGN,
    ProbeSampler,
    clamp_sigma_sections,
    cfg,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_stage2_models(
    device: torch.device,
    dim_u1: int,
    checkpoint_dir: Path,
) -> Tuple[CylinderEncoder, CylinderDecoder, CGN, CGKN_ODE]:
    """Load Stage 2 encoder/decoder/CGN weights."""

    encoder = CylinderEncoder(dim_z=cfg.dim_z).to(device)
    decoder = CylinderDecoder(dim_z=cfg.dim_z).to(device)
    cgn = CGN(dim_u1=dim_u1, dim_z=cfg.dim_z, hidden=cfg.hidden).to(device)

    encoder_ckpt = checkpoint_dir / "stage2_encoder.pt"
    decoder_ckpt = checkpoint_dir / "stage2_decoder.pt"
    cgn_ckpt = checkpoint_dir / "stage2_cgn.pt"

    encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device, weights_only=True))
    decoder.load_state_dict(torch.load(decoder_ckpt, map_location=device, weights_only=True))
    cgn.load_state_dict(torch.load(cgn_ckpt, map_location=device, weights_only=True))

    encoder.eval()
    decoder.eval()
    cgn.eval()

    ode_func = CGKN_ODE(cgn).to(device)
    ode_func.eval()

    return encoder, decoder, cgn, ode_func


def velocity_magnitude(field: torch.Tensor) -> torch.Tensor:
    """Compute velocity magnitude from a [T,2,H,W] field."""

    u = field[:, 0]
    v = field[:, 1]
    mag = torch.sqrt(torch.clamp(u ** 2 + v ** 2, min=0.0))
    return mag.unsqueeze(1)


def select_time_indices(length: int, num: int = 4) -> Iterable[int]:
    """Evenly spaced time indices over the horizon."""

    if length <= num:
        return list(range(length))
    return [int(round(x)) for x in np.linspace(0, length - 1, num=num)]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_rows(
    sequences: Iterable[Tuple[str, np.ndarray, str, Tuple[float, float]]],
    time_indices: Iterable[int],
    save_path: Path,
    suptitle: str,
) -> None:
    """Plot rows of sequences for the selected time indices."""

    sequences = list(sequences)
    time_indices = list(time_indices)
    n_rows = len(sequences)
    n_cols = len(time_indices)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, (title, data, cmap, (vmin, vmax)) in enumerate(sequences):
        for col, t in enumerate(time_indices):
            ax = axes[row, col]
            im = ax.imshow(data[t, 0], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"t = {t}")
            if col == 0:
                ax.set_ylabel(title)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def rollout_prediction(
    encoder: CylinderEncoder,
    decoder: CylinderDecoder,
    ode_func: CGKN_ODE,
    probe_sampler: ProbeSampler,
    normalized_sequence: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Roll out the ODE in latent space and decode to fields."""

    device = normalized_sequence.device
    seq = normalized_sequence.unsqueeze(0)  # [1, T, C, H, W]

    with torch.no_grad():
        v_seq = encoder(seq)  # [1, T, dim_z]
        u1_seq = probe_sampler.sample(seq)  # [1, T, dim_u1]

        uext0 = torch.cat([u1_seq[:, 0, :], v_seq[:, 0, :]], dim=-1)  # [1, dim_u1+dim_z]
        horizon = normalized_sequence.shape[0]
        tspan = torch.linspace(0.0, (horizon - 1) * dt, horizon, device=device)
        uext_pred = odeint(ode_func, uext0, tspan, method="rk4", options={"step_size": dt}).transpose(0, 1)
        dim_u1 = u1_seq.shape[-1]
        v_rollout = uext_pred[:, :, dim_u1:]
        rollout = decoder(v_rollout).squeeze(0)
    return rollout


def data_assimilation_field(
    decoder: CylinderDecoder,
    cgn: CGN,
    probe_sampler: ProbeSampler,
    normalized_sequence: torch.Tensor,
    sigma: torch.Tensor,
    dt: float,
    device: torch.device,
) -> torch.Tensor:
    """Run CGFilter to obtain posterior-mean field reconstruction."""

    seq = normalized_sequence.unsqueeze(0)
    u1_series = probe_sampler.sample(seq)[0, :, :].unsqueeze(-1)  # [T, dim_u1, 1]
    dim_z = cfg.dim_z

    mu0 = torch.zeros(dim_z, 1, device=device)
    R0 = 1e-2 * torch.eye(dim_z, device=device)

    with torch.no_grad():
        mu_post, _ = CGFilter(cgn, sigma, u1_series, mu0, R0, dt)
        mu_v = mu_post.squeeze(-1).unsqueeze(0)
        field = decoder(mu_v).squeeze(0)
    return field


def plot_mse_curve(
    rollout_mse: np.ndarray,
    da_mse: np.ndarray,
    dt: float,
    save_path: Path,
    zoom_steps: int = 5,
) -> None:
    """Plot per-step MSE with an inset zoom for the DA spin-up region."""

    steps = np.arange(len(rollout_mse))
    times = steps * dt

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [2.5, 1.5]})

    # Full horizon view
    axes[0].plot(times, rollout_mse, label="Rollout", marker="o")
    axes[0].plot(times, da_mse, label="DA", marker="s")
    axes[0].set_ylabel("MSE (|u|)")
    axes[0].set_title("Per-step MSE of CGKN Rollout vs Data Assimilation")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # Zoom into early steps to show spin-up behaviour
    zoom_idx = min(len(times), zoom_steps)
    axes[1].plot(times[:zoom_idx], rollout_mse[:zoom_idx], marker="o")
    axes[1].plot(times[:zoom_idx], da_mse[:zoom_idx], marker="s")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("MSE (zoom)")
    axes[1].set_title("DA Spin-up (first {} steps)".format(zoom_idx))
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="CGKN Cylinder predictor")
    parser.add_argument("--train-data", type=str, default="../../../../data/cylinder/cylinder_train_data.npy")
    parser.add_argument("--val-data", type=str, default="../../../../data/cylinder/cylinder_val_data.npy")
    parser.add_argument("--checkpoint-dir", type=str, default="../../../../results/CGKN/Cylinder")
    parser.add_argument("--fig-dir", type=str, default="../../../../results/CGKN/figures")
    parser.add_argument("--probe-file", type=str, default="../../../../results/CGKN/Cylinder/probe_coords.npy")
    parser.add_argument("--sigma-file", type=str, default="../../../../results/CGKN/Cylinder/sigma_hat.npy")
    parser.add_argument("--val-index", type=int, default=3, help="Sample index in validation dataset")
    parser.add_argument("--start-frame", type=int, default=700, help="Start frame for the sequence")
    parser.add_argument("--horizon", type=int, default=30, help="Number of frames to visualise")
    parser.add_argument("--time-steps", type=int, default=4, help="Number of columns in figures")
    parser.add_argument("--warmup", type=int, default=0, help="Number of DA warm-up steps to skip in metrics")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    cfg.debug_stage2 = False  # silence training-time debug prints during inference

    # Load datasets (train for stats, val for evaluation)
    cyl_train = CylinderDynamicsDataset(data_path=args.train_data, seq_length=cfg.seq_length)
    cyl_val = CylinderDynamicsDataset(
        data_path=args.val_data,
        seq_length=cfg.seq_length,
        mean=cyl_train.mean,
        std=cyl_train.std,
    )

    total_frames = cyl_val.frames
    assert args.start_frame + args.horizon <= total_frames, "Requested horizon exceeds dataset length"

    mean = cyl_val.mean.view(1, -1, 1, 1).to(device)
    std = cyl_val.std.view(1, -1, 1, 1).to(device)

    raw_seq = torch.tensor(
        cyl_val.data[args.val_index, args.start_frame : args.start_frame + args.horizon],
        dtype=torch.float32,
        device=device,
    )
    normalized_seq = (raw_seq - mean) / std

    # Probe sampler
    coords = np.load(args.probe_file)
    coords = [tuple(map(int, entry)) for entry in coords.tolist()]
    channels = list(range(mean.shape[1]))  # assumes both velocity channels
    probe_sampler = ProbeSampler(coords, channels)

    # Sigma (clamped to config bounds)
    sigma_np = np.load(args.sigma_file)
    sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)
    sigma = clamp_sigma_sections(sigma, probe_sampler.dim_u1).to(device)

    # Load Stage 2 models
    encoder, decoder, cgn, ode_func = load_stage2_models(device, probe_sampler.dim_u1, Path(args.checkpoint_dir))

    # Predictions
    rollout_norm = rollout_prediction(encoder, decoder, ode_func, probe_sampler, normalized_seq, cfg.dt)
    da_norm = data_assimilation_field(decoder, cgn, probe_sampler, normalized_seq, sigma, cfg.dt, device)

    # Denormalise
    def denorm(x: torch.Tensor) -> torch.Tensor:
        return x * std + mean

    gt_field = denorm(normalized_seq).cpu()
    rollout_field = denorm(rollout_norm).cpu()
    da_field = denorm(da_norm).cpu()

    gt_mag = velocity_magnitude(gt_field)
    rollout_mag = velocity_magnitude(rollout_field)
    da_mag = velocity_magnitude(da_field)

    rollout_err = torch.abs(rollout_mag - gt_mag)
    da_err = torch.abs(da_mag - gt_mag)

    rollout_mse_full = torch.mean((rollout_mag - gt_mag) ** 2, dim=(1, 2, 3)).cpu().numpy()
    da_mse_full = torch.mean((da_mag - gt_mag) ** 2, dim=(1, 2, 3)).cpu().numpy()

    warmup = max(0, min(args.warmup, len(rollout_mse_full)))
    rollout_mse = rollout_mse_full[warmup:]
    da_mse = da_mse_full[warmup:]

    # Prepare plotting data
    time_indices = select_time_indices(args.horizon, num=args.time_steps)
    fig_dir = Path(args.fig_dir)
    ensure_dir(fig_dir)

    gt_mag_np = gt_mag.numpy()
    rollout_mag_np = rollout_mag.numpy()
    da_mag_np = da_mag.numpy()
    rollout_err_np = rollout_err.numpy()
    da_err_np = da_err.numpy()

    value_min = min(gt_mag_np.min(), rollout_mag_np.min())
    value_max = max(gt_mag_np.max(), rollout_mag_np.max())
    da_value_max = max(gt_mag_np.max(), da_mag_np.max())

    rollout_rows = [
        ("Ground Truth", gt_mag_np, "viridis", (value_min, value_max)),
        ("CGKN Rollout", rollout_mag_np, "viridis", (value_min, value_max)),
        ("|Error|", rollout_err_np, "magma", (rollout_err_np.min(), rollout_err_np.max())),
    ]
    plot_rows(rollout_rows, time_indices, fig_dir / "cyl_cgkn_rollout.png", "CGKN Rollout vs Ground Truth")

    da_rows = [
        ("Ground Truth", gt_mag_np, "viridis", (value_min, da_value_max)),
        ("CGKN DA", da_mag_np, "viridis", (value_min, da_value_max)),
        ("|Error|", da_err_np, "magma", (da_err_np.min(), da_err_np.max())),
    ]
    plot_rows(da_rows, time_indices, fig_dir / "cyl_cgkn_da.png", "CGFilter Data Assimilation")

    plot_mse_curve(rollout_mse, da_mse, cfg.dt, fig_dir / "cyl_cgkn_mse.png", zoom_steps=warmup if warmup > 0 else 5)

    print(f"Saved figures to {fig_dir.resolve()}")


if __name__ == "__main__":
    main()
