import os
import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from torchdiffeq import odeint
from tqdm import tqdm

current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
if src_directory not in sys.path:
    sys.path.append(src_directory)

from src.utils.Dataset import ERA5Dataset
from src.models.CGKN.ERA5.era5_model import ERA5Encoder, ERA5Decoder
from src.models.CGKN.ERA5.era5_train import (  # noqa: E402
    ProbeSampler,
    CGN,
    CGKN_ODE,
    cfg,
)


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


def load_stage2_models(device: torch.device, dim_u1: int, checkpoint_dir: Path):
    """Load Stage 2 encoder/decoder/CGN weights for CGKN ERA5."""
    encoder = ERA5Encoder(dim_z=cfg.dim_z).to(device)
    decoder = ERA5Decoder(dim_z=cfg.dim_z).to(device)
    cgn = CGN(dim_u1=dim_u1, dim_z=cfg.dim_z, hidden=cfg.hidden).to(device)

    encoder_ckpt = checkpoint_dir / "stage2_encoder.pt"
    decoder_ckpt = checkpoint_dir / "stage2_decoder.pt"
    cgn_ckpt = checkpoint_dir / "stage2_cgn.pt"

    for path in [encoder_ckpt, decoder_ckpt, cgn_ckpt]:
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint file: {path}")

    encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device, weights_only=True))
    decoder.load_state_dict(torch.load(decoder_ckpt, map_location=device, weights_only=True))
    cgn.load_state_dict(torch.load(cgn_ckpt, map_location=device, weights_only=True))

    encoder.eval()
    decoder.eval()
    cgn.eval()

    ode_func = CGKN_ODE(cgn).to(device)
    ode_func.eval()

    return encoder, decoder, ode_func


def prepare_probe_sampler(probe_file: Path, channels: List[int]) -> ProbeSampler:
    coords_np = np.load(probe_file)
    if coords_np.ndim != 2 or coords_np.shape[1] != 2:
        raise ValueError(f"Probe coordinates must have shape [N,2], got {coords_np.shape}")
    coords: List[Tuple[int, int]] = [tuple(map(int, pair)) for pair in coords_np.tolist()]
    return ProbeSampler(coords, channels)


def rollout_prediction(
    encoder: ERA5Encoder,
    decoder: ERA5Decoder,
    ode_func: CGKN_ODE,
    probe_sampler: ProbeSampler,
    initial_frame: torch.Tensor,
    prediction_steps: int,
    dt: float,
) -> torch.Tensor:
    """
    Perform CGKN rollout starting from a single normalised frame.
    initial_frame: (C, H, W)
    returns tensor of shape (prediction_steps, C, H, W) in normalised space.
    """
    device = initial_frame.device
    seq = initial_frame.unsqueeze(0).unsqueeze(0)  # [1, 1, C, H, W]

    with torch.no_grad():
        v0 = encoder(seq)[:, 0, :]  # [1, dim_z]
        u1_0 = probe_sampler.sample(seq)[:, 0, :]  # [1, dim_u1]
        uext0 = torch.cat([u1_0, v0], dim=-1)

        tspan = torch.linspace(0.0, prediction_steps * dt, prediction_steps + 1, device=device)
        uext_pred = odeint(ode_func, uext0, tspan, method="rk4", options={"step_size": dt})  # [(T+1), 1, dim]
        uext_pred = uext_pred[1:].transpose(0, 1)  # [1, T, dim]
        dim_u1 = probe_sampler.dim_u1
        v_rollout = uext_pred[:, :, dim_u1:]  # [1, T, dim_z]
        rollout = decoder(v_rollout).squeeze(0)  # [T, C, H, W]

    return rollout


if __name__ == "__main__":
    prediction_steps = 50
    num_starts = 30
    start_min = 100
    seed = 42

    model_dir = Path("../../../../results/CGKN/ERA5")
    save_dir = Path("../../../../results/CGKN/figures")
    probe_file = model_dir / "probe_coords.npy"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    cfg.debug_stage2 = False

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

    channels = list(getattr(cfg, "use_channels", range(raw_test_data.shape[-1])))
    probe_sampler = prepare_probe_sampler(probe_file, channels)
    encoder, decoder, ode_func = load_stage2_models(device, probe_sampler.dim_u1, model_dir)

    metric_storage = {"mse": [], "rrmse": [], "ssim": []}

    for start_frame in tqdm(start_frames, desc="Evaluating CGKN ERA5 rollouts"):
        initial_frame = torch.tensor(raw_test_data[start_frame, ...], dtype=torch.float32).permute(2, 0, 1)
        groundtruth = torch.tensor(
            raw_test_data[start_frame + 1 : start_frame + 1 + prediction_steps, ...],
            dtype=torch.float32,
        ).permute(0, 3, 1, 2)

        normalized_initial = era5_test_dataset.normalize(initial_frame.unsqueeze(0))[0].to(device)
        rollout_norm = rollout_prediction(
            encoder,
            decoder,
            ode_func,
            probe_sampler,
            normalized_initial,
            prediction_steps,
            cfg.dt,
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
