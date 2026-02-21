"""Discrete CGKN ERA5 rollout evaluation (mean/std metrics).

This script mirrors the continuous CGKN ERA5 prediction interface but uses the
**discrete one-step mapping** from ``NSE(Noisy)_CGKN.py``:
    [u1, v] -> [f1 + g1 v, f2 + g2 v]
where ``u1`` is the flattened probe observation vector and ``v`` is the latent
state from the encoder. Rollout is performed by iterating this discrete mapping
(no ODE integration).

Ground truth tensors follow the repository convention ``[B, T, C, H, W]``.
Rollout initialization: the first frame (after normalization) is encoded to get
``v0``; probes on the same frame give ``u1_0``; these are concatenated into
``[u1_0, v0]`` as the initial extended state before discrete stepping.
"""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# Ensure project root is importable for Dataset and shared modules
current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
if src_directory not in sys.path:
    sys.path.append(src_directory)

from src.utils.Dataset import ERA5Dataset  # noqa: E402
from src.models.discreteCGKN.ERA5.era5_model import ERA5Decoder, ERA5Encoder  # noqa: E402
from src.models.discreteCGKN.ERA5.era5_train import (  # noqa: E402
    DiscreteCGN,
    ProbeSampler,
)


def compute_channel_metrics(gt: torch.Tensor, pred: torch.Tensor) -> Dict[str, np.ndarray]:
    """Compute per-channel metric sequences.

    Both ``gt`` and ``pred`` are tensors of shape ``(T, C, H, W)`` and are
    assumed to be **de-normalised** (physical units). Returns numpy arrays with
    shape ``(C, T)`` for each metric.
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


def load_models(device: torch.device, checkpoint_dir: Path, dim_z: int, dim_u1: int, ckpt_prefix: str):
    """Load encoder/decoder/dynamics weights saved by the discrete trainer."""

    encoder = ERA5Encoder(dim_z=dim_z).to(device)
    decoder = ERA5Decoder(dim_z=dim_z).to(device)
    cgn = DiscreteCGN(dim_u1=dim_u1, dim_z=dim_z).to(device)

    encoder_ckpt = checkpoint_dir / f"{ckpt_prefix}_encoder.pt"
    decoder_ckpt = checkpoint_dir / f"{ckpt_prefix}_decoder.pt"
    cgn_ckpt = checkpoint_dir / f"{ckpt_prefix}_cgn.pt"

    for path in [encoder_ckpt, decoder_ckpt, cgn_ckpt]:
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint file: {path}")

    encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device, weights_only=True))
    decoder.load_state_dict(torch.load(decoder_ckpt, map_location=device, weights_only=True))
    cgn.load_state_dict(torch.load(cgn_ckpt, map_location=device, weights_only=True))

    encoder.eval()
    decoder.eval()
    cgn.eval()
    return encoder, decoder, cgn


def prepare_probe_sampler(probe_file: Path, channels: List[int]) -> ProbeSampler:
    """Load probe coordinates and build the sampler for u1 extraction."""

    coords_np = np.load(probe_file)
    if coords_np.ndim != 2 or coords_np.shape[1] != 2:
        raise ValueError(f"Probe coordinates must have shape [N,2], got {coords_np.shape}")
    coords: List[Tuple[int, int]] = [tuple(map(int, pair)) for pair in coords_np.tolist()]
    return ProbeSampler(coords, channels)


def rollout_prediction(
    encoder: ERA5Encoder,
    decoder: ERA5Decoder,
    cgn: DiscreteCGN,
    probe_sampler: ProbeSampler,
    initial_frame: torch.Tensor,
    prediction_steps: int,
) -> torch.Tensor:
    """Roll out ``prediction_steps`` frames using the discrete CGN mapping.

    Args:
        initial_frame: tensor ``(C, H, W)`` already normalized.
    Returns:
        Tensor ``(prediction_steps, C, H, W)`` in normalized space.
    """

    device = initial_frame.device
    seq = initial_frame.unsqueeze(0).unsqueeze(0)  # [1, 1, C, H, W]

    with torch.no_grad():
        v0 = encoder(seq)[:, 0, :]  # [1, dim_z]
        u1_0 = probe_sampler.sample(seq)[:, 0, :]  # [1, dim_u1]
        u_ext = torch.cat([u1_0, v0], dim=-1)  # [1, dim_u1+dim_z]

        preds: List[torch.Tensor] = []
        for _ in range(prediction_steps):
            u_ext = cgn(u_ext)  # discrete one-step mapping
            preds.append(u_ext)

        u_stack = torch.stack(preds, dim=1)  # [1, T, dim_u1+dim_z]
        dim_u1 = probe_sampler.dim_u1
        v_rollout = u_stack[:, :, dim_u1:]  # [1, T, dim_z]
        rollout = decoder(v_rollout).squeeze(0)  # [T, C, H, W]

    return rollout


def main():
    parser = argparse.ArgumentParser(description="Discrete CGKN ERA5 rollout evaluation (mean/std)")
    parser.add_argument("--model_name", type=str, default="discreteCGKN")
    parser.add_argument("--ckpt_prefix", type=str, default="stage2")
    parser.add_argument("--data_path", type=str, default="../../../../data/ERA5/ERA5_data/test_seq_state.h5")
    parser.add_argument("--min_path", type=str, default="../../../../data/ERA5/ERA5_data/min_val.npy")
    parser.add_argument("--max_path", type=str, default="../../../../data/ERA5/ERA5_data/max_val.npy")
    parser.add_argument("--start_T", type=int, default=100, help="minimum frame index to start rollouts")
    parser.add_argument("--window_length", type=int, default=50, help="prediction horizon")
    parser.add_argument("--num_runs", type=int, default=30, help="number of rollout start frames")
    parser.add_argument("--obs_ratio", type=float, default=0.15, help="probe ratio used during training")
    parser.add_argument("--use_channels", type=int, nargs="+", default=[0, 1], help="channels used for probes")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    model_dir = Path(f"../../../../results/{args.model_name}/ERA5")
    save_dir = model_dir / "Pred"
    probe_file = model_dir / "probe_coords.npy"

    if not probe_file.exists():
        raise FileNotFoundError(f"probe_coords.npy not found at {probe_file}")

    dataset = ERA5Dataset(
        data_path=args.data_path,
        seq_length=12,
        min_path=args.min_path,
        max_path=args.max_path,
    )
    denorm = dataset.denormalizer()
    raw_data = dataset.data  # expected shape [N, H, W, C]
    total_frames = raw_data.shape[0]

    max_start = total_frames - (args.window_length + 1)
    if max_start <= args.start_T:
        raise ValueError("Not enough frames to sample the requested rollouts with the specified start_T/window.")

    start_candidates = list(range(args.start_T, max_start))
    if len(start_candidates) < args.num_runs:
        raise ValueError(f"Cannot sample {args.num_runs} unique start frames from range({args.start_T}, {max_start}).")
    start_frames = random.sample(start_candidates, args.num_runs)

    channels = args.use_channels
    sampler = prepare_probe_sampler(probe_file, channels)

    dim_z = raw_data.shape[-1]  # fallback, overwritten below if settings exist
    try:
        from src.models.CAE_Koopman.ERA5.era5_model import ERA5_settings

        dim_z = ERA5_settings["state_feature_dim"][-1]
    except Exception:
        pass

    encoder, decoder, cgn = load_models(device, model_dir, dim_z, sampler.dim_u1, args.ckpt_prefix)

    metric_storage = {"mse": [], "rrmse": [], "ssim": []}
    sample_pred = None
    sample_gt = None

    for idx, start_frame in enumerate(tqdm(start_frames, desc="Evaluating discrete CGKN ERA5 rollouts")):
        initial_frame = torch.tensor(raw_data[start_frame, ...], dtype=torch.float32).permute(2, 0, 1)
        groundtruth = torch.tensor(
            raw_data[start_frame + 1 : start_frame + 1 + args.window_length, ...], dtype=torch.float32
        ).permute(0, 3, 1, 2)

        normalized_initial = dataset.normalize(initial_frame.unsqueeze(0))[0].to(device)
        rollout_norm = rollout_prediction(
            encoder,
            decoder,
            cgn,
            sampler,
            normalized_initial,
            args.window_length,
        ).cpu()
        rollout = denorm(rollout_norm)

        metrics = compute_channel_metrics(groundtruth, rollout)
        for key in metric_storage:
            metric_storage[key].append(metrics[key])

        if sample_pred is None:
            sample_pred = rollout.numpy()
            sample_gt = groundtruth.numpy()

    stacked_metrics = {k: np.stack(v, axis=0) for k, v in metric_storage.items()}  # (num_runs, C, T)
    channel_step_means = {k: np.mean(val, axis=0) for k, val in stacked_metrics.items()}
    channel_step_stds = {k: np.std(val, axis=0) for k, val in stacked_metrics.items()}
    channel_means = {k: np.mean(val, axis=(0, 2)) for k, val in stacked_metrics.items()}
    channel_stds = {k: np.std(val, axis=(0, 2)) for k, val in stacked_metrics.items()}

    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / "metrics_era5_forward.npz"
    np.savez(
        metrics_path,
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

    if sample_pred is not None:
        np.savez(save_dir / "pred_sample.npz", pred=sample_pred, gt=sample_gt, start_frame=start_frames[0])

    channel_names = ["Geopotential", "Temperature", "Humidity", "Wind_u", "Wind_v"]
    print("\nPer-channel overall metrics (mean ± std across rollouts and steps):")
    for idx, name in enumerate(channel_names[: channel_means["mse"].shape[0]]):
        print(
            f"{name:<12} | "
            f"MSE={channel_means['mse'][idx]:.6e}±{channel_stds['mse'][idx]:.2e}, "
            f"RRMSE={channel_means['rrmse'][idx]:.6e}±{channel_stds['rrmse'][idx]:.2e}, "
            f"SSIM={channel_means['ssim'][idx]:.6f}±{channel_stds['ssim'][idx]:.3f}"
        )

    print(f"\nSaved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
