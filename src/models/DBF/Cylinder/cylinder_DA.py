"""
Cylinder data assimilation (DA) evaluation for the DBF model.

The assimilation loop mirrors the ERA5 DBF evaluation:
- Use observation schedule on assimilation steps where step=0 corresponds to t=1.
- Initialize latent state with a Gaussian prior and fuse the t=0 observation.
- Run prediction/update for each step, decode DA/NoDA trajectories, and compute metrics.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

# Ensure repository root is on path
current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
if src_directory not in sys.path:
    sys.path.append(src_directory)

from src.models.DBF.Cylinder.cylinder_model import CylinderDecoder
from src.models.DBF.Cylinder.cylinder_trainer import (
    CylinderIOONetwork,
    ObservationMask,
    SpectralKoopmanOperator,
    create_probe_mask,
)
from src.utils.Dataset import CylinderDynamicsDataset


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def set_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_denorm(x: torch.Tensor, dataset: CylinderDynamicsDataset) -> torch.Tensor:
    """Denormalize Cylinder tensors on CPU."""
    if isinstance(x, torch.Tensor):
        x_cpu = x.detach().cpu()
        mean = dataset.mean.reshape(1, -1, 1, 1)
        std = dataset.std.reshape(1, -1, 1, 1)
        return (x_cpu * std + mean).cpu()
    return x


def compute_metrics(
    da_states: torch.Tensor,
    noda_states: torch.Tensor,
    groundtruth: torch.Tensor,
    dataset: CylinderDynamicsDataset,
) -> Dict[str, np.ndarray]:
    """Compute per-step, per-channel MSE/RRMSE/SSIM for one run."""
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


def _symmetrize(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + x.transpose(-1, -2))


def dbf_gaussian_update(
    mu_prior: torch.Tensor,
    cov_prior: torch.Tensor,
    mu_r: torch.Tensor,
    cov_r: torch.Tensor,
    mu_rho: torch.Tensor,
    cov_rho: torch.Tensor,
    epsilon: float,
    use_rho: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """DBF Gaussian fusion in information form with optional /rho term."""
    eye = torch.eye(2, device=mu_prior.device).view(1, 1, 2, 2)
    cov_prior = cov_prior + epsilon * eye
    cov_r = cov_r + epsilon * eye
    cov_rho = cov_rho + epsilon * eye

    precision_prior = torch.linalg.inv(cov_prior)
    precision_r = torch.linalg.inv(cov_r)
    precision_rho = torch.linalg.inv(cov_rho) if use_rho else torch.zeros_like(precision_prior)

    precision_post = precision_prior + precision_r - precision_rho
    precision_post = _symmetrize(precision_post)

    eigvals = torch.linalg.eigvalsh(precision_post)
    min_eig = eigvals.min().item()
    if min_eig <= 0:
        jitter = (abs(min_eig) + epsilon) * torch.ones_like(precision_post[..., 0, 0])
        precision_post = precision_post + jitter.unsqueeze(-1).unsqueeze(-1) * torch.eye(2, device=mu_prior.device)

    cov_post = torch.linalg.inv(precision_post)
    info_vec = torch.matmul(precision_prior, mu_prior.unsqueeze(-1)) + torch.matmul(precision_r, mu_r.unsqueeze(-1))
    if use_rho:
        info_vec = info_vec - torch.matmul(precision_rho, mu_rho.unsqueeze(-1))
    mu_post = torch.matmul(cov_post, info_vec).squeeze(-1)
    cov_post = _symmetrize(cov_post)
    return mu_post, cov_post


def decode_pairs(decoder: CylinderDecoder, latent_pairs: torch.Tensor) -> torch.Tensor:
    b, num_pairs, _ = latent_pairs.shape
    latent_flat = latent_pairs.reshape(b, num_pairs * 2)
    frame = decoder.decoder(latent_flat)
    return frame


@torch.no_grad()
def run_multi_da_experiment(
    obs_ratio: float = 0.15,
    obs_noise_std: float = 0.05,
    observation_schedule: Iterable[int] | None = None,
    observation_variance: float | None = None,
    window_length: int = 50,
    num_runs: int = 5,
    start_T: int = 0,
    sample_idx: int = 0,
    model_name: str = "DBF",
    checkpoint_name: str = "best_model.pt",
    use_rho: bool = True,
    save_prefix: str | None = None,
) -> Dict[str, object]:
    set_seed(42)
    device = set_device()
    print(f"Using device: {device}")

    ckpt_path = Path(f"../../../../results/{model_name}/Cylinder") / checkpoint_name
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
    mask_seed = int(cfg.get("mask_seed", 1024))

    # Dataset slice
    forward_step = 12
    cyl_train_dataset = CylinderDynamicsDataset(
        data_path="../../../../data/cylinder/cylinder_train_data.npy",
        seq_length=forward_step,
        mean=None,
        std=None,
    )
    cyl_val_dataset = CylinderDynamicsDataset(
        data_path="../../../../data/cylinder/cylinder_val_data.npy",
        seq_length=forward_step,
        mean=cyl_train_dataset.mean,
        std=cyl_train_dataset.std,
    )

    total_frames = window_length + 1
    if start_T + total_frames > cyl_val_dataset.data.shape[1]:
        raise ValueError("Requested window exceeds available cylinder sequence length.")

    raw_data = cyl_val_dataset.data[sample_idx, start_T : start_T + total_frames, ...]
    groundtruth = torch.tensor(raw_data, dtype=torch.float32)
    normalized_groundtruth = cyl_val_dataset.normalize(groundtruth)
    print(f"Ground truth slice shape: {groundtruth.shape}")

    # Observation mask
    if "mask" in checkpoint:
        observation_mask = ObservationMask(checkpoint["mask"].bool())
        print("Loaded observation mask from checkpoint.")
    else:
        print("Checkpoint missing mask; regenerating from obs_ratio and seed.")
        mask_tensor = create_probe_mask(
            channels=cyl_val_dataset.channel,
            height=cyl_val_dataset.H,
            width=cyl_val_dataset.W,
            rate=obs_ratio,
            seed=mask_seed,
        )
        observation_mask = ObservationMask(mask_tensor)
    observation_mask = observation_mask.to(device)
    print(f"Observation dimension: {observation_mask.obs_dim}")

    decoder = CylinderDecoder(dim_z=latent_dim).to(device)
    decoder.load_state_dict(checkpoint["decoder"])
    decoder.eval()

    ioo = CylinderIOONetwork(
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
    print(f"Latent spectral pairs: {num_pairs}")

    normalized_gt_device = normalized_groundtruth.to(device)
    obs_all = observation_mask.sample(normalized_gt_device.unsqueeze(0))
    assert obs_all.shape[1] >= window_length + 1, "Insufficient observation frames for window"

    obs_all_clean = obs_all.detach()

    if observation_schedule is None:
        observation_schedule = list(range(window_length))
    observation_schedule = set(observation_schedule)

    run_metrics = {"mse": [], "rrmse": [], "ssim": []}
    run_times = []
    first_run_states = None
    first_run_original_states = None

    cov_rho = init_cov * torch.eye(2, device=device).view(1, 1, 2, 2)
    mu_rho = torch.zeros(1, num_pairs, 2, device=device)

    for run_idx in range(num_runs):
        print(f"\nStarting assimilation run {run_idx + 1}/{num_runs}")
        set_seed(42 + run_idx)

        obs_all = obs_all_clean.clone()
        if obs_noise_std and obs_noise_std > 0:
            obs_all = obs_all + torch.randn_like(obs_all) * obs_noise_std

        obs_sequence = obs_all[:, 1 : window_length + 1, :].clone()

        mu_prev, cov_prev = initial_latent_state(
            batch_size=1, num_pairs=num_pairs, device=device, init_cov=init_cov
        )
        init_obs = obs_all[:, 0, :].to(device)
        mu_init_flat, sigma2_init_flat = ioo(init_obs)
        mu_init_pairs = mu_init_flat.view(1, num_pairs, 2)
        cov_init = torch.diag_embed(sigma2_init_flat.view(1, num_pairs, 2))
        mu_prev, cov_prev = dbf_gaussian_update(
            mu_prior=mu_prev,
            cov_prior=cov_prev,
            mu_r=mu_init_pairs,
            cov_r=cov_init,
            mu_rho=mu_rho,
            cov_rho=cov_rho,
            epsilon=cov_epsilon,
            use_rho=use_rho,
        )

        mu_noda_prev = mu_prev.clone()
        cov_noda_prev = cov_prev.clone()

        da_states = []
        noda_states = []
        run_start = perf_counter()

        for step in range(window_length):
            mu_prior, cov_prior = koopman.predict(mu_prev, cov_prev)
            mu_noda_prior, cov_noda_prior = koopman.predict(mu_noda_prev, cov_noda_prev)

            if step in observation_schedule:
                obs_t = obs_sequence[:, step, :]
                mu_obs_flat, sigma2_obs_flat = ioo(obs_t)
                mu_obs_pairs = mu_obs_flat.view(1, num_pairs, 2)
                cov_obs = torch.diag_embed(sigma2_obs_flat.view(1, num_pairs, 2))
                mu_post, cov_post = dbf_gaussian_update(
                    mu_prior=mu_prior,
                    cov_prior=cov_prior,
                    mu_r=mu_obs_pairs,
                    cov_r=cov_obs,
                    mu_rho=mu_rho,
                    cov_rho=cov_rho,
                    epsilon=cov_epsilon,
                    use_rho=use_rho,
                )
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
            first_run_original_states = noda_stack.clone()

        metrics = compute_metrics(da_stack, noda_stack, groundtruth, cyl_val_dataset)
        for key in run_metrics:
            run_metrics[key].append(metrics[key])

        run_times.append(perf_counter() - run_start)
        print(f"Run {run_idx + 1} completed.")

    save_dir = Path(f"../../../../results/{model_name}/Cylinder/DA")
    os.makedirs(save_dir, exist_ok=True)

    def prefixed(name: str) -> str:
        return f"{save_prefix}{name}" if save_prefix else name

    def _as_numpy(value):
        if value is None:
            return np.array(None, dtype=object)
        return np.array(value)

    if first_run_states is not None:
        np.save(
            save_dir / prefixed("multi.npy"),
            safe_denorm(first_run_states, cyl_val_dataset).numpy(),
        )
        print(f"Saved DA trajectory to {save_dir / prefixed('multi.npy')}")

    if first_run_original_states is not None:
        np.save(
            save_dir / prefixed("multi_original.npy"),
            safe_denorm(first_run_original_states, cyl_val_dataset).numpy(),
        )
        print(f"Saved NoDA trajectory to {save_dir / prefixed('multi_original.npy')}")

    metrics_meanstd = {}
    for key in run_metrics:
        metric_array = np.stack(run_metrics[key], axis=0)
        metrics_meanstd[f"{key}_mean"] = metric_array.mean(axis=0)
        metrics_meanstd[f"{key}_std"] = metric_array.std(axis=0)

    np.savez(
        save_dir / prefixed("multi_meanstd.npz"),
        **metrics_meanstd,
        steps=np.arange(1, window_length + 1),
        metrics=["MSE", "RRMSE", "SSIM"],
    )
    print(f"Saved mean/std metrics to {save_dir / prefixed('multi_meanstd.npz')}")

    time_info = {
        "assimilation_time": run_times,
        "assimilation_time_mean": float(np.mean(run_times)),
        "assimilation_time_std": float(np.std(run_times)),
    }
    time_info_path = save_dir / prefixed("time_info.npz")
    np.savez(
        time_info_path,
        assimilation_time=_as_numpy(run_times),
        assimilation_time_mean=time_info["assimilation_time_mean"],
        assimilation_time_std=time_info["assimilation_time_std"],
    )
    print(f"Saved time info to {time_info_path}")

    return time_info


if __name__ == "__main__":
    run_multi_da_experiment()
