"""
Data assimilation experiments for CGKN on Cylinder, aligned with the ERA5 CGKN DA workflow.
"""

import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchdiffeq
from skimage.metrics import structural_similarity as ssim

# Add src directory to path
current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
sys.path.append(src_directory)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_config")

from src.utils.Dataset import CylinderDynamicsDataset
from src.models.CGKN.Cylinder.cylinder_model import (
    CYLINDER_settings,
    CylinderDecoder,
    CylinderEncoder,
)
from src.models.CGKN.Cylinder.cylinder_trainer import make_probe_coords_from_ratio


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def set_seed(seed: int | None):
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)


def set_device() -> str:
    if torch.cuda.device_count() == 0:
        return "cpu"
    torch.set_float32_matmul_precision("high")
    return "cuda"


def safe_denorm(x: torch.Tensor, dataset: CylinderDynamicsDataset) -> torch.Tensor:
    """Denormalize Cylinder tensors on CPU."""
    if not isinstance(x, torch.Tensor):
        return x

    x_cpu = x.detach().cpu()
    mean = torch.as_tensor(dataset.mean, dtype=x_cpu.dtype, device=x_cpu.device)
    std = torch.as_tensor(dataset.std, dtype=x_cpu.dtype, device=x_cpu.device)

    if x_cpu.ndim == 4:  # (B,C,H,W)
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)
    elif x_cpu.ndim == 3:  # (C,H,W)
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
    else:
        raise ValueError(f"safe_denorm expects 3D or 4D tensor, got {tuple(x_cpu.shape)}")

    return x_cpu * std + mean


def compute_metrics(
    da_states: torch.Tensor,
    noda_states: torch.Tensor,
    groundtruth: torch.Tensor,
    dataset: CylinderDynamicsDataset,
    start_offset: int = 1,
) -> Dict[str, np.ndarray]:
    """Compute per-step MSE/RRMSE/SSIM on velocity magnitude."""
    mse = []
    rrmse = []
    ssim_scores = []

    eps = 1e-12
    T = da_states.shape[0]
    assert start_offset + T <= groundtruth.shape[0], "groundtruth length mismatch"

    for step in range(T):
        target = groundtruth[step + start_offset]
        da = safe_denorm(da_states[step], dataset)
        noda = safe_denorm(noda_states[step], dataset)

        if da.ndim == 4:
            da = da.squeeze(0)
        if noda.ndim == 4:
            noda = noda.squeeze(0)

        target_mag = torch.sqrt((target**2).sum(dim=0))
        da_mag = torch.sqrt((da**2).sum(dim=0))
        noda_mag = torch.sqrt((noda**2).sum(dim=0))

        diff_da = (da_mag - target_mag) ** 2
        diff_noda = (noda_mag - target_mag) ** 2

        mse_da = diff_da.mean().item()
        mse_noda = diff_noda.mean().item()

        denom = (target_mag**2).sum().clamp_min(eps)
        rrmse_da = torch.sqrt(diff_da.sum() / denom).item()
        rrmse_noda = torch.sqrt(diff_noda.sum() / denom).item()

        data_range = (target_mag.max() - target_mag.min()).item()
        if data_range > 0:
            ssim_da = ssim(target_mag.numpy(), da_mag.numpy(), data_range=data_range)
            ssim_noda = ssim(target_mag.numpy(), noda_mag.numpy(), data_range=data_range)
        else:
            ssim_da = 1.0
            ssim_noda = 1.0

        mse.append([(mse_da, mse_noda)])
        rrmse.append([(rrmse_da, rrmse_noda)])
        ssim_scores.append([(ssim_da, ssim_noda)])

    return {
        "mse": np.array(mse),
        "rrmse": np.array(rrmse),
        "ssim": np.array(ssim_scores),
    }


def _tensor_summary(name: str, tensor: torch.Tensor) -> str:
    data = tensor.detach()
    shape = tuple(data.shape)
    flat = data.reshape(-1)
    finite_mask = torch.isfinite(flat)
    finite_count = int(finite_mask.sum().item())
    total = flat.numel()
    non_finite = total - finite_count
    if finite_count == 0:
        return f"{name}: shape={shape}, finite=0/{total}"
    finite_values = flat[finite_mask].float()
    return (
        f"{name}: shape={shape}, finite={finite_count}/{total}, nonfinite={non_finite}, "
        f"min={torch.min(finite_values).item():.3e}, max={torch.max(finite_values).item():.3e}, "
        f"mean={torch.mean(finite_values).item():.3e}, std={torch.std(finite_values, unbiased=False).item():.3e}, "
        f"norm={torch.norm(finite_values).item():.3e}"
    )


def clamp_sigma_sections(
    sigma: torch.Tensor,
    dim_u1: int,
    obs_min: float | None = 1e-2,
    obs_max: float | None = 1.0,
    lat_min: float | None = 1e-2,
    lat_max: float | None = 1.0,
) -> torch.Tensor:
    sigma = sigma.clone()
    if obs_min is not None or obs_max is not None:
        sigma[:dim_u1] = sigma[:dim_u1].clamp(
            min=obs_min if obs_min is not None else -float("inf"),
            max=obs_max if obs_max is not None else float("inf"),
        )
    if lat_min is not None or lat_max is not None:
        sigma[dim_u1:] = sigma[dim_u1:].clamp(
            min=lat_min if lat_min is not None else -float("inf"),
            max=lat_max if lat_max is not None else float("inf"),
        )
    return sigma


def stabilise_covariance(
    R: torch.Tensor,
    min_var: float | None = 1e-6,
    max_var: float | None = 1.0,
    cov_clip: float | None = 1.0,
    use_diag: bool = True,
) -> torch.Tensor:
    if use_diag:
        diag = torch.diag(R)
        diag = diag.clamp(
            min=min_var if min_var is not None else -float("inf"),
            max=max_var if max_var is not None else float("inf"),
        )
        R = torch.diag(diag)
    else:
        R = 0.5 * (R + R.T)
        if cov_clip is not None:
            R = torch.clamp(R, min=-cov_clip, max=cov_clip)
        if min_var is not None or max_var is not None:
            diag = torch.diag(R).clamp(
                min=min_var if min_var is not None else -float("inf"),
                max=max_var if max_var is not None else float("inf"),
            )
            R = R - torch.diag(torch.diag(R)) + torch.diag(diag)
    if cov_clip is not None and use_diag:
        R = torch.clamp(R, min=-cov_clip, max=cov_clip)
    return R


# -----------------------------------------------------------------------------
# Sparse observation sampler (same structure as ERA5 script)
# -----------------------------------------------------------------------------
class ProbeSampler:
    def __init__(self, probe_coords: List[Tuple[int, int]], channels: Sequence[int]):
        probe_coords = [tuple(map(int, xy)) for xy in probe_coords]
        
        self.coords = list(dict.fromkeys(probe_coords))
        self.channels = list(channels)
        self.dim_u1 = len(self.coords) * len(self.channels)

    def _gather_single(self, frame: torch.Tensor) -> torch.Tensor:
        vals = []
        for (r, c) in self.coords:
            v = frame[:, r, c]
            v = v[self.channels]
            vals.append(v)
        return torch.cat(vals, dim=0)

    def sample(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: [B,T,C,H,W]
        B, T, _, _, _ = seq.shape
        out = torch.zeros(B, T, self.dim_u1, device=seq.device)
        for b in range(B):
            for t in range(T):
                out[b, t] = self._gather_single(seq[b, t])
        return out


# -----------------------------------------------------------------------------
# CGKN core components (same structure as ERA5 script)
# -----------------------------------------------------------------------------
class CGN(nn.Module):
    def __init__(self, dim_u1: int, dim_z: int, hidden: int = 128):
        super().__init__()
        self.dim_u1 = dim_u1
        self.dim_z = dim_z

        def mlp(in_d, out_d):
            return nn.Sequential(
                nn.Linear(in_d, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, out_d),
            )

        self.net_f1 = mlp(dim_u1, dim_u1)
        self.net_g1 = mlp(dim_u1, dim_u1 * dim_z)
        self.net_f2 = mlp(dim_u1, dim_z)
        self.net_g2 = mlp(dim_u1, dim_z * dim_z)

    def forward(self, u1: torch.Tensor):
        B = u1.shape[0]
        f1 = self.net_f1(u1).unsqueeze(-1)
        g1 = self.net_g1(u1).view(B, self.dim_u1, self.dim_z)
        f2 = self.net_f2(u1).unsqueeze(-1)
        g2 = self.net_g2(u1).view(B, self.dim_z, self.dim_z)
        return f1, g1, f2, g2


class CGKN_ODE(nn.Module):
    def __init__(self, cgn: CGN):
        super().__init__()
        self.cgn = cgn

    def forward(self, t, uext: torch.Tensor):
        dim_u1 = self.cgn.dim_u1
        u1 = uext[:, :dim_u1]
        v = uext[:, dim_u1:]
        f1, g1, f2, g2 = self.cgn(u1)
        v_col = v.unsqueeze(-1)
        u1_dot = f1 + torch.bmm(g1, v_col)
        v_dot = f2 + torch.bmm(g2, v_col)
        return torch.cat([u1_dot.squeeze(-1), v_dot.squeeze(-1)], dim=-1)


@torch.no_grad()
def CGFilter(
    cgn: CGN,
    sigma: torch.Tensor,
    u1_init: torch.Tensor,
    obs_series: torch.Tensor,
    update_mask: torch.Tensor,
    mu0: torch.Tensor,
    R0: torch.Tensor,
    dt: float,
    kalman_reg: float = 1e-6,
    kalman_cov_clip: float | None = None,
    kalman_gain_clip: float | None = None,
    kalman_gain_scale: float = 1.0,
    kalman_innovation_clip: float | None = None,
    kalman_drift_clip: float | None = None,
    kalman_state_clip: float | None = None,
    use_diag_cov: bool = False,
    observation_variance: float | None = None,
    debug: bool = False,
):
    device = obs_series.device
    T, dim_u1, _ = obs_series.shape
    dim_z = mu0.shape[0]

    sigma = clamp_sigma_sections(
        sigma, dim_u1,
        obs_min=1e-6, obs_max=None,
        lat_min=1e-6, lat_max=None,
    )
    s1 = torch.diag(sigma[:dim_u1]).to(device)
    s2 = torch.diag(sigma[dim_u1:]).to(device)

    eps = kalman_reg
    eye_u1 = torch.eye(dim_u1, device=device)
    eye_z = torch.eye(dim_z, device=device)

    obs_var = observation_variance if observation_variance is not None else 0.0
    s1_cov = s1 @ s1.T + eps * eye_u1
    if obs_var > 0:
        s1_cov = s1_cov + (2.0 * obs_var / dt) * eye_u1
    s2_cov = s2 @ s2.T + eps * eye_z
    S_inv = torch.linalg.inv(s1_cov)

    mu_prev = mu0.clone()
    R_prev = R0.clone()
    mu_post = [mu_prev.unsqueeze(0)]
    R_post = [R_prev.unsqueeze(0)]
    u1_forcing = []

    assert u1_init.shape == (dim_u1,), f"u1_init shape {u1_init.shape} != ({dim_u1},)"
    u1_curr = u1_init.to(device)
    u1_forcing.append(u1_curr.unsqueeze(0))

    for n in range(1, T):
        u1_prev = u1_curr
        obs_prev = None
        obs_curr = None

        if update_mask[n - 1]:
            obs_prev = obs_series[n - 1, :, 0]
            if torch.isnan(obs_prev).any():
                obs_prev = None

        if update_mask[n]:
            obs_curr = obs_series[n, :, 0]
            if torch.isnan(obs_curr).any():
                obs_curr = None

        if obs_curr is not None:
            u1_curr = obs_curr

        f1, g1, f2, g2 = cgn(u1_prev.unsqueeze(0))
        f1 = f1.squeeze(0)
        g1 = g1.squeeze(0)
        f2 = f2.squeeze(0)
        g2 = g2.squeeze(0)

        s2_cov = stabilise_covariance(
            s2_cov,
            min_var=None,
            max_var=None,
            cov_clip=kalman_cov_clip,
            use_diag=use_diag_cov,
        )

        drift = f2 + g2 @ mu_prev
        mu_pred = mu_prev + drift * dt
        R_pred = R_prev + (g2 @ R_prev + R_prev @ g2.T + s2_cov) * dt
        R_pred = stabilise_covariance(
            R_pred,
            min_var=None,
            max_var=None,
            cov_clip=kalman_cov_clip,
            use_diag=use_diag_cov,
        )

        if obs_prev is None or obs_curr is None:
            mu_new = mu_pred
            R_new = R_pred
        else:
            du1 = obs_curr.unsqueeze(-1) - obs_prev.unsqueeze(-1)
            innovation = du1 - (f1 + g1 @ mu_prev) * dt
            if kalman_innovation_clip is not None:
                innovation = innovation.clamp(min=-kalman_innovation_clip, max=kalman_innovation_clip)

            if not torch.isfinite(innovation).all():
                raise RuntimeError(_tensor_summary("innovation", innovation))

            K = (R_prev @ g1.T) @ S_inv
            if kalman_gain_scale is not None:
                K = K * kalman_gain_scale
            if kalman_gain_clip is not None:
                K = K.clamp(min=-kalman_gain_clip, max=kalman_gain_clip)

            mu_new = mu_pred + K @ innovation
            if kalman_drift_clip is not None:
                mu_new = mu_new.clamp(min=-kalman_drift_clip, max=kalman_drift_clip)

            R_new = R_prev + (
                g2 @ R_prev
                + R_prev @ g2.T
                + s2_cov
                - R_prev @ g1.T @ S_inv @ g1 @ R_prev
            ) * dt
            R_new = stabilise_covariance(
                R_new,
                min_var=None,
                max_var=None,
                cov_clip=kalman_cov_clip,
                use_diag=use_diag_cov,
            )
            if kalman_state_clip is not None:
                mu_new = mu_new.clamp(min=-kalman_state_clip, max=kalman_state_clip)

            if debug:
                diag = torch.diag(R_new)
                print(
                    "[CGFilter] step "
                    f"{n} | innov_norm={torch.norm(innovation).item():.3e} | "
                    f"K_norm={torch.norm(K).item():.3e} | "
                    f"R_diag_min={diag.min().item():.3e} | "
                    f"R_diag_max={diag.max().item():.3e}"
                )

        mu_prev, R_prev = mu_new, R_new
        if not torch.isfinite(mu_new).all():
            raise RuntimeError(_tensor_summary("mu_new", mu_new))
        if not torch.isfinite(R_new).all():
            raise RuntimeError(_tensor_summary("R_new", R_new))

        mu_post.append(mu_new.unsqueeze(0))
        R_post.append(R_new.unsqueeze(0))
        u1_forcing.append(u1_curr.unsqueeze(0))

    return torch.cat(mu_post, dim=0), torch.cat(R_post, dim=0), torch.cat(u1_forcing, dim=0)


# -----------------------------------------------------------------------------
# Assimilation runner
# -----------------------------------------------------------------------------
@torch.no_grad()
def run_multi_da_experiment(
    obs_ratio: float = 0.02,
    obs_noise_std: float = 0.05,
    observation_schedule: list = list(range(2+1)),
    observation_variance: float | None = None,
    window_length: int = 2,
    num_runs: int = 5,
    early_stop_config: Tuple[int, float] = (100, 1e-3),
    start_T: int = 700,
    sample_idx: int = 3,
    model_name: str = "CGKN",
    ckpt_prefix: str = "stage2",
    probe_layout: str = "random",
    probe_seed: int = 42,
    probe_min_spacing: int = 4,
    use_channels: Sequence[int] = (0, 1),
    dt: float = 0.001,
    save_prefix: str | None = None,
):
    set_seed(42)
    device = set_device()
    print(f"Using device: {device}")

    results_dir = Path(f"../../../../results/{model_name}/Cylinder")
    ckpt_paths = {
        "encoder": results_dir / f"{ckpt_prefix}_encoder.pt",
        "decoder": results_dir / f"{ckpt_prefix}_decoder.pt",
        "cgn": results_dir / f"{ckpt_prefix}_cgn.pt",
    }
    missing = [name for name, path in ckpt_paths.items() if not path.exists()]
    if missing:
        missing_list = ", ".join(str(ckpt_paths[name]) for name in missing)
        raise FileNotFoundError(f"Missing checkpoint(s): {missing_list}")

    expected_dim_z = CYLINDER_settings["state_feature_dim"][-1]
    encoder = CylinderEncoder(dim_z=expected_dim_z).to(device)
    decoder = CylinderDecoder(dim_z=expected_dim_z).to(device)
    encoder.load_state_dict(torch.load(ckpt_paths["encoder"], map_location=device, weights_only=True))
    decoder.load_state_dict(torch.load(ckpt_paths["decoder"], map_location=device, weights_only=True))

    sigma_path = results_dir / "sigma_hat.npy"
    if not sigma_path.exists():
        raise FileNotFoundError(f"sigma_hat.npy not found at {sigma_path}")
    sigma_hat = torch.tensor(np.load(sigma_path), dtype=torch.float32)

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

    H, W = int(cyl_val_dataset.H), int(cyl_val_dataset.W)
    coords_path = results_dir / "probe_coords.npy"
    if coords_path.exists():
        coords = np.load(coords_path).tolist()
        print(f"Loaded probe coordinates from {coords_path}")
    else:
        coords = make_probe_coords_from_ratio(
            H, W, obs_ratio, layout=probe_layout, seed=probe_seed, min_spacing=probe_min_spacing
        )
        print(f"Probe coordinates not found at {coords_path}, generated {len(coords)} points instead.")

    probe_sampler = ProbeSampler(coords, use_channels)
    dim_u1 = probe_sampler.dim_u1
    print(f"[OBS] dim_u1 = {dim_u1}  (points={len(coords)}, channels={len(use_channels)})")

    observation_steps = list(range(total_frames)) if observation_schedule is None else list(observation_schedule)
    for step in observation_steps:
        if step < 0 or step >= total_frames:
            raise ValueError(f"observation_schedule step {step} outside [0, {total_frames - 1}]")
    update_mask = torch.zeros(total_frames, dtype=torch.bool)
    for step in observation_steps:
        update_mask[step] = True

    latent_dim = sigma_hat.numel() - dim_u1
    assert latent_dim > 0, "sigma_hat length must exceed dim_u1 to include latent components"
    if latent_dim != expected_dim_z:
        raise ValueError(f"latent dim from sigma_hat ({latent_dim}) does not match expected {expected_dim_z}")
    if sigma_hat.numel() != dim_u1 + latent_dim:
        raise ValueError(
            f"sigma_hat length {sigma_hat.numel()} does not match expected dim_u1+dim_z ({dim_u1 + latent_dim})."
        )

    cgn = CGN(dim_u1=dim_u1, dim_z=latent_dim, hidden=128).to(device)
    cgn.load_state_dict(torch.load(ckpt_paths["cgn"], map_location=device, weights_only=True))
    ode_func = CGKN_ODE(cgn).to(device)

    u1_full = probe_sampler.sample(normalized_groundtruth.unsqueeze(0))
    assert u1_full.shape[1] == total_frames, "Probe sampling length mismatch"
    u1_gt = u1_full[0]

    update_mask = update_mask.to(device)

    v0_enc = encoder(normalized_groundtruth[:1].unsqueeze(0).to(device))[:, 0, :]
    mu0 = v0_enc.squeeze(0).detach().unsqueeze(-1)
    R0 = 1e-4 * torch.eye(latent_dim, device=device)

    run_metrics = {"mse": [], "rrmse": [], "ssim": []}
    first_run_states = None
    first_run_original_states = None
    run_times: List[float] = []

    for run_idx in range(num_runs):
        print(f"\nStarting assimilation run {run_idx + 1}/{num_runs}")
        set_seed(42 + run_idx)

        u1_obs = torch.full_like(u1_gt, float("nan"))
        for step in observation_steps:
            obs = u1_gt[step].clone()
            if obs_noise_std > 0:
                obs = obs + torch.randn_like(obs) * obs_noise_std
            u1_obs[step] = obs

        if 0 in observation_steps:
            u1_init = u1_obs[0].clone()
        else:
            u1_init = u1_gt[0].clone()

        obs_series = u1_obs.unsqueeze(-1).to(device)

        run_start = perf_counter()
        obs_floor_std = 0.02
        obs_var = (max(obs_noise_std, obs_floor_std) ** 2) if observation_variance is None else observation_variance
        mu_post, _, _ = CGFilter(
            cgn,
            sigma_hat.to(device),
            u1_init.to(device),
            obs_series,
            update_mask,
            mu0,
            R0,
            dt,
            observation_variance=obs_var,
        )
        mu_v = mu_post.squeeze(-1)
        da_fields = decoder(mu_v.unsqueeze(0)).squeeze(0).detach().cpu()
        da_stack = da_fields[1:]
        run_times.append(perf_counter() - run_start)

        tspan = torch.linspace(0.0, window_length * dt, window_length + 1, device=device)
        v0 = torch.zeros(1, latent_dim, device=device)
        uext0 = torch.cat([u1_init.to(device).unsqueeze(0), v0], dim=-1)
        uext_pred = torchdiffeq.odeint(ode_func, uext0, tspan, method="rk4", options={"step_size": dt})
        uext_pred = uext_pred.transpose(0, 1)
        v_pred = uext_pred[:, :, dim_u1:]
        noda_fields = decoder(v_pred).squeeze(0).detach().cpu()[1:]

        if first_run_states is None:
            first_run_states = da_stack.clone()
            first_run_original_states = noda_fields.clone()

        metrics = compute_metrics(da_stack, noda_fields, groundtruth, cyl_val_dataset)
        for key in run_metrics:
            run_metrics[key].append(metrics[key])

    save_dir = results_dir / "DA"
    os.makedirs(save_dir, exist_ok=True)

    def prefixed(name: str) -> str:
        return f"{save_prefix}{name}" if save_prefix else name

    def _as_numpy(value):
        if value is None:
            return np.array(None, dtype=object)
        return np.array(value)

    if first_run_states is not None:
        np.save(save_dir / prefixed("multi.npy"), safe_denorm(first_run_states, cyl_val_dataset).numpy())
        print(f"Saved sample DA trajectory to {save_dir / prefixed('multi.npy')}")

    if first_run_original_states is not None:
        np.save(
            save_dir / prefixed("multi_original.npy"),
            safe_denorm(first_run_original_states, cyl_val_dataset).numpy(),
        )
        print(f"Saved sample NoDA trajectory to {save_dir / prefixed('multi_original.npy')}")

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

    for key in ["mse", "rrmse", "ssim"]:
        run_values = [m.mean() for m in run_metrics[key]]
        print(
            f"{key.upper()} mean over runs: {float(np.mean(run_values)):.6f}, std: {float(np.std(run_values)):.6f}"
        )

    if run_times:
        print(f"Average assimilation time: {np.mean(run_times):.2f}s over {num_runs} runs")

    time_info = {
        "assimilation_time": run_times,
        "assimilation_time_mean": float(np.mean(run_times)) if run_times else 0.0,
        "assimilation_time_std": float(np.std(run_times)) if run_times else 0.0,
        "iteration_counts": None,
        "iteration_count_mean": None,
        "iteration_count_std": None,
    }
    time_info_path = save_dir / prefixed("time_info.npz")
    np.savez(
        time_info_path,
        assimilation_time=_as_numpy(run_times),
        assimilation_time_mean=time_info["assimilation_time_mean"],
        assimilation_time_std=time_info["assimilation_time_std"],
        iteration_counts=_as_numpy(None),
        iteration_count_mean=_as_numpy(None),
        iteration_count_std=_as_numpy(None),
    )
    print(f"Saved time info to {time_info_path}")
    return time_info


if __name__ == "__main__":
    run_multi_da_experiment()
