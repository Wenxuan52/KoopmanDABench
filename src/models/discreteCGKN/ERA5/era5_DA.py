"""
Discrete CGKN data assimilation on ERA5 with masked observation schedule.

This script mirrors the CAE_Koopman/CGKN ERA5 DA interface while using the
NSE(Noisy)_CGKN discrete predict-update equations. Observations arrive only on
`observation_schedule` steps; non-observed steps rely on model-predicted probe
values to avoid peeking at ground truth.

Key shapes (normalized space unless stated otherwise):
- Ground truth: [T, C, H, W]
- Probe observations u1: [dim_u1]
- Latent mean mu: [dim_z, 1]
- Covariance R: [dim_z, dim_z]

Discrete one-step mapping (from NSE reference):
    u_extended = [u1, v] -> [f1 + g1 v, f2 + g2 v]
Predict/update formulas follow the closed-form CGFilter:
    mu_next = f2 + g2 mu + (update ? g2 R g1^T @ inv(S1 S1^T + g1 R g1^T)
                                    @ (u_obs - f1 - g1 mu) : 0)
    R_next  = g2 R g2^T + S2 S2^T - (update ? g2 R g1^T @ inv(S1 S1^T + g1 R g1^T)
                                           @ g1 R g2^T : 0)
where S1 = diag(sigma[:dim_u1]), S2 = diag(sigma[dim_u1:]).
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

# Ensure project root is importable
current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
if src_directory not in sys.path:
    sys.path.append(src_directory)

from src.utils.Dataset import ERA5Dataset  # noqa: E402
from src.models.discreteCGKN.ERA5.era5_model import ERA5Decoder, ERA5Encoder  # noqa: E402
from src.models.discreteCGKN.ERA5.era5_train import DiscreteCGN, ProbeSampler  # noqa: E402


def set_seed(seed: int | None):
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)


def set_device() -> torch.device:
    if torch.cuda.device_count() == 0:
        return torch.device("cpu")
    torch.set_float32_matmul_precision("high")
    return torch.device("cuda")


def safe_denorm(x: torch.Tensor, dataset: ERA5Dataset) -> torch.Tensor:
    """Denormalize ERA5 tensors on CPU."""

    if isinstance(x, torch.Tensor):
        x_cpu = x.detach().cpu()
        min_val = dataset.min.reshape(-1, 1, 1)
        max_val = dataset.max.reshape(-1, 1, 1)
        return (x_cpu * (max_val - min_val) + min_val).cpu()
    return x


def compute_metrics(
    da_states: torch.Tensor, noda_states: torch.Tensor, groundtruth: torch.Tensor, dataset: ERA5Dataset
) -> Dict[str, np.ndarray]:
    """Compute per-step per-channel metrics (MSE/RRMSE/SSIM)."""

    mse = []
    rrmse = []
    ssim_scores = []

    for step in range(groundtruth.shape[0] - 1):
        target = groundtruth[step + 1]
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

    return {"mse": np.array(mse), "rrmse": np.array(rrmse), "ssim": np.array(ssim_scores)}


def load_models(
    device: torch.device, checkpoint_dir: Path, dim_z: int, dim_u1: int, ckpt_prefix: str
) -> Tuple[ERA5Encoder, ERA5Decoder, DiscreteCGN]:
    encoder = ERA5Encoder(dim_z=dim_z).to(device)
    decoder = ERA5Decoder(dim_z=dim_z).to(device)
    cgn = DiscreteCGN(dim_u1=dim_u1, dim_z=dim_z).to(device)

    encoder.load_state_dict(
        torch.load(checkpoint_dir / f"{ckpt_prefix}_encoder.pt", map_location=device, weights_only=True)
    )
    decoder.load_state_dict(
        torch.load(checkpoint_dir / f"{ckpt_prefix}_decoder.pt", map_location=device, weights_only=True)
    )
    cgn.load_state_dict(
        torch.load(checkpoint_dir / f"{ckpt_prefix}_cgn.pt", map_location=device, weights_only=True)
    )

    encoder.eval()
    decoder.eval()
    cgn.eval()
    return encoder, decoder, cgn


def predict_update(
    cgn: DiscreteCGN,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    R: torch.Tensor,
    u_forcing: torch.Tensor,
    do_update: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Predict/update step using NSE discrete CGFilter formulas.

    Args:
        mu: latent mean ``[dim_z, 1]``
        R: latent covariance ``[dim_z, dim_z]``
        u_forcing: probe vector for this step ``[dim_u1, 1]``; when ``do_update``
            is False, this is usually the model-predicted probe (no ground truth).
    Returns:
        (mu_next, R_next)
    """

    dim_u1 = cgn.dim_u1
    dim_z = cgn.dim_z
    f1, g1, f2, g2 = cgn()
    S1 = torch.diag(sigma[:dim_u1])
    S2 = torch.diag(sigma[dim_u1:])

    # Prediction (no observation correction yet)
    mu_pred = f2 + g2 @ mu
    R_pred = g2 @ R @ g2.T + S2 @ S2.T

    if do_update:
        innovation = u_forcing - f1 - g1 @ mu
        middle = torch.inverse(S1 @ S1.T + g1 @ R @ g1.T)
        gain = g2 @ R @ g1.T @ middle
        mu_pred = mu_pred + gain @ innovation
        R_pred = R_pred - gain @ g1 @ R @ g2.T

    return mu_pred, R_pred


def prepare_probe_sampler(probe_file: Path, channels: List[int]) -> ProbeSampler:
    coords_np = np.load(probe_file)
    coords: List[Tuple[int, int]] = [tuple(map(int, pair)) for pair in coords_np.tolist()]
    return ProbeSampler(coords, channels)


def run_multi_da_experiment(
    obs_ratio: float = 0.15,
    obs_noise_std: float = 0.05,
    observation_schedule: list = [0, 10, 20, 30, 40],
    observation_variance: float | None = None,
    window_length: int = 50,
    num_runs: int = 5,
    early_stop_config: Tuple[int, float] = (100, 1e-3),
    start_T: int = 0,
    model_name: str = "discreteCGKN",
    data_path: str = "../../../../data/ERA5/ERA5_data/test_seq_state.h5",
    min_path: str = "../../../../data/ERA5/ERA5_data/min_val.npy",
    max_path: str = "../../../../data/ERA5/ERA5_data/max_val.npy",
    ckpt_prefix: str = "stage2",
    use_channels: Sequence[int] = (0, 1),
    forward_step: int = 12,
    seed: int | None = 42,
    device: str | None = None,
):
    """Run masked-schedule DA and return mean/std metrics."""

    set_seed(seed)
    torch_device = torch.device(device) if device is not None else set_device()
    print(f"Using device: {torch_device}")

    # Dataset and ground truth slice
    dataset = ERA5Dataset(data_path=data_path, seq_length=forward_step, min_path=min_path, max_path=max_path)
    total_frames = window_length + 1

    run_metrics = {"mse": [], "rrmse": [], "ssim": []}
    first_run_states = None

    checkpoint_dir = Path(f"../../../../results/{model_name}/ERA5")
    sigma_path = checkpoint_dir / "sigma_hat.npy"
    probe_path = checkpoint_dir / "probe_coords.npy"
    if not sigma_path.exists():
        raise FileNotFoundError(f"Missing sigma_hat.npy at {sigma_path}")
    if not probe_path.exists():
        raise FileNotFoundError(f"Missing probe_coords.npy at {probe_path}")

    sigma_np = np.load(sigma_path)
    probe_sampler = prepare_probe_sampler(probe_path, list(use_channels))
    dim_u1 = probe_sampler.dim_u1
    dim_z = sigma_np.shape[0] - dim_u1
    encoder, decoder, cgn = load_models(torch_device, checkpoint_dir, dim_z=dim_z, dim_u1=dim_u1, ckpt_prefix=ckpt_prefix)

    sigma = torch.from_numpy(sigma_np).to(torch_device).float()
    if observation_variance is not None:
        sigma[:dim_u1] = torch.sqrt(torch.full((dim_u1,), observation_variance, device=torch_device))

    # Observation mask
    obs_set = set(int(s) for s in observation_schedule)

    for run_idx in range(num_runs):
        cur_start = start_T + run_idx
        raw_data = dataset.data[cur_start : cur_start + total_frames, ...]
        groundtruth = torch.tensor(raw_data, dtype=torch.float32).permute(0, 3, 1, 2)
        normalized_gt = dataset.normalize(groundtruth)
        print(f"Run {run_idx + 1}/{num_runs} - slice [{cur_start}, {cur_start + total_frames})")

        da_states = []
        noda_states = []

        # Initial latent mean/cov
        with torch.no_grad():
            v0 = encoder(normalized_gt[:1].unsqueeze(0).to(torch_device))[:, 0, :].squeeze(0)
        mu = v0.unsqueeze(-1)
        R = 0.01 * torch.eye(dim_z, device=torch_device)
        mu_noda = mu.clone()
        R_noda = R.clone()

        for step in range(window_length):
            target_frame = normalized_gt[step + 1 : step + 2].unsqueeze(0).to(torch_device)
            u_obs = None
            if step in obs_set:
                u_obs = probe_sampler.sample(target_frame).squeeze(0).squeeze(0)
                noise = torch.randn_like(u_obs) * obs_noise_std
                u_obs = (u_obs + noise).unsqueeze(-1)

            u_pred = (cgn.f1 + cgn.g1 @ mu).detach()
            u_forcing = u_obs if u_obs is not None else u_pred

            mu, R = predict_update(cgn, sigma, mu, R, u_forcing=u_forcing, do_update=u_obs is not None)
            da_frame = decoder(mu.transpose(0, 1)).squeeze(0).detach().cpu()
            da_states.append(da_frame)

            # No-DA rollout (always predict without update)
            noda_u = (cgn.f1 + cgn.g1 @ mu_noda).detach()
            mu_noda, R_noda = predict_update(cgn, sigma, mu_noda, R_noda, u_forcing=noda_u, do_update=False)
            noda_frame = decoder(mu_noda.transpose(0, 1)).squeeze(0).detach().cpu()
            noda_states.append(noda_frame)

        da_stack = torch.stack(da_states)
        noda_stack = torch.stack(noda_states)

        if first_run_states is None:
            first_run_states = da_stack.clone()

        metrics = compute_metrics(da_stack, noda_stack, groundtruth, dataset)
        for key in run_metrics:
            run_metrics[key].append(metrics[key])

    save_dir = Path(f"../../../../results/{model_name}/ERA5/DA")
    save_dir.mkdir(parents=True, exist_ok=True)

    if first_run_states is not None:
        np.save(save_dir / "multi.npy", safe_denorm(first_run_states, dataset).numpy())
        print(f"Saved sample DA trajectory to {save_dir / 'multi.npy'}")

    metrics_meanstd = {}
    for key in run_metrics:
        metric_array = np.stack(run_metrics[key], axis=0)  # (runs, steps, channels, 2)
        metrics_meanstd[f"{key}_mean"] = metric_array.mean(axis=0)
        metrics_meanstd[f"{key}_std"] = metric_array.std(axis=0)

    np.savez(
        save_dir / "multi_meanstd.npz",
        **metrics_meanstd,
        steps=np.arange(1, window_length + 1),
        metrics=["MSE", "RRMSE", "SSIM"],
    )
    print(f"Saved mean/std metrics to {save_dir / 'multi_meanstd.npz'}")

    # Overall stats
    for key in ["mse", "rrmse", "ssim"]:
        run_values = [m.mean() for m in run_metrics[key]]
        print(
            f"{key.upper()} mean over runs: {np.mean(run_values):.6f}, std: {np.std(run_values):.6f}"
        )


if __name__ == "__main__":
    run_multi_da_experiment()
