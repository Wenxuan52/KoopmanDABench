"""
Data assimilation evaluation for DBF on ERA5, aligned with the CAE_Linear workflow.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

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
    create_probe_mask,
)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def set_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_denorm(x: torch.Tensor, dataset: ERA5Dataset) -> torch.Tensor:
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
) -> Dict[str, np.ndarray]:
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

    return {
        "mse": np.array(mse),
        "rrmse": np.array(rrmse),
        "ssim": np.array(ssim_scores),
    }


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


def decode_pairs(decoder: ERA5Decoder, latent_pairs: torch.Tensor) -> torch.Tensor:
    b, num_pairs, _ = latent_pairs.shape
    latent_flat = latent_pairs.reshape(b, num_pairs * 2)
    frame = decoder.decoder(latent_flat)
    return frame


@torch.no_grad()
def run_multi_da_experiment(
    obs_ratio: float = 0.15,
    obs_noise_std: float = 0.05,
    observation_schedule = [0, 10, 20, 30, 40],
    observation_variance: float | None = None,
    window_length: int = 50,
    num_runs: int = 5,
    early_stop_config: Tuple[int, float] | None = None,
    start_T: int = 0,
    model_name: str = "DBF",
    checkpoint_name: str = "best_model.pt",
):
    set_seed(42)
    device = set_device()
    print(f"Using device: {device}")

    ckpt_path = Path(f"../../../../results/{model_name}/ERA5") / checkpoint_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    cfg = checkpoint.get("config", {})
    latent_dim = int(cfg.get("latent_dim", 512))
    init_cov = float(cfg.get("init_cov", 1.0))
    cov_epsilon = float(cfg.get("cov_epsilon", 1e-6))
    rho_clip = float(cfg.get("rho_clip", 0.2))
    process_noise_init = float(cfg.get("process_noise_init", -2.0))
    hidden_dims = cfg.get("ioo_hidden_dims", [1024, 1024])

    # Dataset
    forward_step = 12
    dataset = ERA5Dataset(
        data_path="../../../../data/ERA5/ERA5_data/test_seq_state.h5",
        seq_length=forward_step,
        min_path="../../../../data/ERA5/ERA5_data/min_val.npy",
        max_path="../../../../data/ERA5/ERA5_data/max_val.npy",
    )

    total_frames = window_length + 1
    raw_data = dataset.data[start_T : start_T + total_frames, ...]
    groundtruth = torch.tensor(raw_data, dtype=torch.float32).permute(0, 3, 1, 2)
    normalized_groundtruth = dataset.normalize(groundtruth)
    print(f"Ground truth slice shape: {groundtruth.shape}")

    # Observation mask
    if "mask" in checkpoint:
        observation_mask = ObservationMask(checkpoint["mask"].bool())
        print("Loaded observation mask from checkpoint.")
    else:
        print("Checkpoint missing mask; regenerating from obs_ratio and seed.")
        mask_tensor = create_probe_mask(
            channels=dataset.C,
            height=dataset.H,
            width=dataset.W,
            rate=obs_ratio,
            seed=int(cfg.get("mask_seed", 1024)),
        )
        observation_mask = ObservationMask(mask_tensor)
    observation_mask = observation_mask.to(device)
    print(f"Observation dimension: {observation_mask.obs_dim}")

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

    num_pairs = koopman.num_pairs

    obs_all = observation_mask.sample(normalized_groundtruth.unsqueeze(0))  # [1, T, obs_dim]
    assert obs_all.shape[1] >= window_length + 1, "Insufficient observation frames for window"

    run_metrics = {"mse": [], "rrmse": [], "ssim": []}
    run_times = []
    first_run_states = None

    for run_idx in range(num_runs):
        print(f"\nStarting assimilation run {run_idx + 1}/{num_runs}")
        set_seed(42 + run_idx)

        obs_sequence = obs_all[:, 1 : window_length + 1, :].clone()
        if obs_noise_std:
            obs_sequence = obs_sequence + torch.randn_like(obs_sequence) * obs_noise_std

        mu_prev, cov_prev = initial_latent_state(batch_size=1, num_pairs=num_pairs, device=device, init_cov=init_cov)
        init_obs = obs_all[:, 0, :]
        mu_init_flat, sigma2_init_flat = ioo(init_obs)
        mu_init_pairs = mu_init_flat.view(1, num_pairs, 2)
        cov_init = torch.diag_embed(sigma2_init_flat.view(1, num_pairs, 2))
        mu_prev, cov_prev = gaussian_update(mu_prev, cov_prev, mu_init_pairs, cov_init, cov_epsilon)

        mu_noda_prev = mu_prev.clone()
        cov_noda_prev = cov_prev.clone()

        da_states = []
        noda_states = []

        for step in range(window_length):
            mu_prior, cov_prior = koopman.predict(mu_prev, cov_prev)
            mu_noda_prior, cov_noda_prior = koopman.predict(mu_noda_prev, cov_noda_prev)

            if step in observation_schedule:
                obs_t = obs_sequence[:, step, :]
                mu_obs_flat, sigma2_obs_flat = ioo(obs_t)
                mu_obs_pairs = mu_obs_flat.view(1, num_pairs, 2)
                cov_obs = torch.diag_embed(sigma2_obs_flat.view(1, num_pairs, 2))
                mu_post, cov_post = gaussian_update(mu_prior, cov_prior, mu_obs_pairs, cov_obs, cov_epsilon)
            else:
                mu_post, cov_post = mu_prior, cov_prior

            decoded_da = decode_pairs(decoder, mu_post).squeeze(0).detach().cpu()
            da_states.append(decoded_da)

            mu_prev, cov_prev = mu_post, cov_post

            decoded_noda = decode_pairs(decoder, mu_noda_prior).squeeze(0).detach().cpu()
            noda_states.append(decoded_noda)
            mu_noda_prev, cov_noda_prev = mu_noda_prior, cov_noda_prior

        da_stack = torch.stack(da_states)
        noda_stack = torch.stack(noda_states)

        if first_run_states is None:
            first_run_states = da_stack.clone()

        metrics = compute_metrics(da_stack, noda_stack, groundtruth, dataset)
        for key in run_metrics:
            run_metrics[key].append(metrics[key])

        run_times.append(0.0)
        print(f"Run {run_idx + 1} completed.")

    save_dir = f"../../../../results/{model_name}/ERA5/DA"
    os.makedirs(save_dir, exist_ok=True)

    if first_run_states is not None:
        np.save(os.path.join(save_dir, "multi.npy"), safe_denorm(first_run_states, dataset).numpy())
        print(f"Saved DA trajectory to {os.path.join(save_dir, 'multi.npy')}")

    metrics_meanstd = {}
    for key in run_metrics:
        metric_array = np.stack(run_metrics[key], axis=0)
        metrics_meanstd[f"{key}_mean"] = metric_array.mean(axis=0)
        metrics_meanstd[f"{key}_std"] = metric_array.std(axis=0)

    np.savez(
        os.path.join(save_dir, "multi_meanstd.npz"),
        **metrics_meanstd,
        steps=np.arange(1, window_length + 1),
        metrics=["MSE", "RRMSE", "SSIM"],
    )
    print(f"Saved mean/std metrics to {os.path.join(save_dir, 'multi_meanstd.npz')}")

    for key in ["mse", "rrmse", "ssim"]:
        run_values = [m.mean() for m in run_metrics[key]]
        print(
            f"{key.upper()} mean over runs: {float(np.mean(run_values)):.6f}, std: {float(np.std(run_values)):.6f}"
        )

    print(f"Average assimilation time: {np.mean(run_times):.2f}s over {num_runs} runs")

    return run_metrics


if __name__ == "__main__":
    run_multi_da_experiment()
