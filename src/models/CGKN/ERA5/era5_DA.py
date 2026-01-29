"""
Data assimilation experiments for CGKN on ERA5, aligned with the CAE_Linear workflow.

This script mirrors the interface, saving structure, and metric computation of
``src/models/CAE_Linear/ERA5/era5_DA.py`` while reusing the analytic CGFilter
logic from the ERA5 CGKN training script.
"""

import math
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

from src.utils.Dataset import ERA5Dataset
from src.models.CGKN.ERA5.era5_model import ERA5Decoder, ERA5Encoder, ERA5_settings


# -----------------------------------------------------------------------------
# Utility helpers (mirroring ERA5 training script)
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
    stats = {
        "min": float(torch.min(finite_values).item()),
        "max": float(torch.max(finite_values).item()),
        "mean": float(torch.mean(finite_values).item()),
        "std": float(torch.std(finite_values, unbiased=False).item()),
        "norm": float(torch.norm(finite_values).item()),
    }
    return (
        f"{name}: shape={shape}, finite={finite_count}/{total}, nonfinite={non_finite}, "
        f"min={stats['min']:.3e}, max={stats['max']:.3e}, "
        f"mean={stats['mean']:.3e}, std={stats['std']:.3e}, norm={stats['norm']:.3e}"
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


def make_probe_coords_from_ratio(
    H: int,
    W: int,
    ratio: float,
    layout: str = "uniform",
    seed: int = 42,
    min_spacing: int = 4,
) -> List[Tuple[int, int]]:
    assert 0 < ratio <= 1.0
    n = max(1, int(round(H * W * ratio)))

    if layout == "uniform":
        r = max(1, int(round(math.sqrt(n * H / W))))
        c = max(1, int(round(n / r)))
        step_r = H // r
        step_c = W // c
        rows = [min(H - 1, i * step_r + step_r // 2) for i in range(r)]
        cols = [min(W - 1, j * step_c + step_c // 2) for j in range(c)]
        coords = [(int(rr), int(cc)) for rr in rows for cc in cols]
        if len(coords) > n:
            coords = coords[:n]
    else:
        rng = np.random.default_rng(seed)
        coords = []
        attempts = 0
        max_attempts = H * W * 10
        while len(coords) < n and attempts < max_attempts:
            r = int(rng.integers(0, H))
            c = int(rng.integers(0, W))
            if all(abs(r - rr) >= min_spacing or abs(c - cc) >= min_spacing for rr, cc in coords):
                coords.append((r, c))
            attempts += 1
        if len(coords) < n:
            print(f"[WARN] Only placed {len(coords)} of {n} requested probe points after {attempts} attempts.")
    return coords


class ProbeSampler:
    def __init__(self, coords: Sequence[Tuple[int, int]], channels: Sequence[int]):
        self.coords = list(coords)
        self.channels = list(channels)
        self.dim_u1 = len(self.coords) * len(self.channels)

    def _gather_single(self, frame: torch.Tensor) -> torch.Tensor:
        # frame: [C, H, W]
        values = []
        for c in self.channels:
            chan = frame[c]
            for (r, cc) in self.coords:
                values.append(chan[..., r, cc])
        return torch.stack(values)

    def sample(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: [B, T, C, H, W]
        B, T, _, _, _ = seq.shape
        out = torch.zeros(B, T, self.dim_u1, device=seq.device)
        for b in range(B):
            for t in range(T):
                out[b, t] = self._gather_single(seq[b, t])
        return out


# -----------------------------------------------------------------------------
# CGKN core components (copied from training script)
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
    kalman_cov_clip: float | None = 1.0,
    kalman_gain_clip: float | None = 50.0,
    kalman_gain_scale: float = 1e-3,
    kalman_innovation_clip: float | None = 1.0,
    kalman_drift_clip: float | None = 0.1,
    kalman_state_clip: float | None = 10.0,
    use_diag_cov: bool = True,
    observation_variance: float | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = obs_series.device
    T, dim_u1, _ = obs_series.shape
    dim_z = mu0.shape[0]
    sigma = clamp_sigma_sections(sigma, dim_u1)
    s1 = torch.diag(sigma[:dim_u1]).to(device)
    s2 = torch.diag(sigma[dim_u1:]).to(device)
    eps = kalman_reg
    eye_u1 = torch.eye(dim_u1, device=device)
    eye_z = torch.eye(dim_z, device=device)
    obs_var = observation_variance if observation_variance is not None else 0.0
    s1_cov = s1 @ s1.T + eps * eye_u1 + obs_var * eye_u1
    s2_cov = s2 @ s2.T + eps * eye_z

    mu_prev = mu0.clone()
    R_prev = R0.clone()
    mu_post = []
    R_post = []
    u1_forcing = []

    assert u1_init.shape == (dim_u1,), f"u1_init shape {u1_init.shape} != ({dim_u1},)"
    u1_curr = u1_init.to(device)

    for n in range(T):
        f1, g1, f2, g2 = cgn(u1_curr.unsqueeze(0))
        f1 = f1.squeeze(0)
        g1 = g1.squeeze(0)
        f2 = f2.squeeze(0)
        g2 = g2.squeeze(0)

        if not update_mask[n]:
            obs = None
        else:
            obs = obs_series[n, :, 0]
            if torch.isnan(obs).any():
                raise ValueError(f"Observation missing at step {n} where update_mask is True")

        s2_cov = stabilise_covariance(s2_cov, use_diag=use_diag_cov)
        A = eye_z + dt * g2
        q = dt * (f2 + g2 @ mu_prev) + 0.5 * (dt**2) * (g2 @ f2)
        mu_pred = A @ mu_prev + q
        R_pred = A @ R_prev @ A.T + dt * (g2 @ R_prev @ g2.T) + s2_cov
        R_pred = stabilise_covariance(R_pred, use_diag=use_diag_cov)

        if obs is None:
            mu_new = mu_pred
            R_new = R_pred
        else:
            H = dt * g1
            innovation = obs.unsqueeze(-1) - dt * (f1 + g1 @ mu_pred)
            if kalman_innovation_clip is not None:
                innovation = innovation.clamp(min=-kalman_innovation_clip, max=kalman_innovation_clip)
            S = H @ R_pred @ H.T + s1_cov
            S = stabilise_covariance(S, use_diag=use_diag_cov)
            S_inv = torch.linalg.pinv(S)
            K = R_pred @ H.T @ S_inv
            if kalman_gain_scale is not None:
                K = K * kalman_gain_scale
            if kalman_gain_clip is not None:
                K = K.clamp(min=-kalman_gain_clip, max=kalman_gain_clip)
            mu_new = mu_pred + K @ innovation
            if kalman_drift_clip is not None:
                mu_new = mu_new.clamp(min=-kalman_drift_clip, max=kalman_drift_clip)
            R_new = (eye_z - K @ H) @ R_pred @ (eye_z - K @ H).T + K @ s1_cov @ K.T
            R_new = stabilise_covariance(R_new, use_diag=use_diag_cov)
            if kalman_state_clip is not None:
                mu_new = mu_new.clamp(min=-kalman_state_clip, max=kalman_state_clip)
        mu_prev, R_prev = mu_new, R_new

        if not torch.isfinite(mu_new).all():
            raise RuntimeError(_tensor_summary("mu_new", mu_new))
        if not torch.isfinite(R_new).all():
            raise RuntimeError(_tensor_summary("R_new", R_new))
        mu_post.append(mu_new.unsqueeze(0))
        R_post.append(R_new.unsqueeze(0))
        u1_forcing.append(u1_curr.unsqueeze(0))

        if n < T - 1:
            v_curr = mu_new.squeeze(-1)
            u1_dot = f1 + g1 @ v_curr.unsqueeze(-1)
            v_dot = f2 + g2 @ v_curr.unsqueeze(-1)
            u1_next = u1_curr + dt * u1_dot.squeeze(-1)
            u1_curr = u1_next.detach()

    return torch.cat(mu_post, dim=0), torch.cat(R_post, dim=0), torch.cat(u1_forcing, dim=0)


# -----------------------------------------------------------------------------
# Assimilation runner (mirrors CAE_Linear interface)
# -----------------------------------------------------------------------------
@torch.no_grad()
def run_multi_da_experiment(
    obs_ratio: float = 0.15,
    obs_noise_std: float = 0.05,
    observation_schedule: list = [0, 10, 20, 30, 40],
    observation_variance: float | None = None,
    window_length: int = 50,
    num_runs: int = 5,
    early_stop_config: Tuple[int, float] = (100, 1e-3),
    start_T: int = 0,
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

    results_dir = Path(f"../../../../results/{model_name}/ERA5")
    ckpt_paths = {
        "encoder": results_dir / f"{ckpt_prefix}_encoder.pt",
        "decoder": results_dir / f"{ckpt_prefix}_decoder.pt",
        "cgn": results_dir / f"{ckpt_prefix}_cgn.pt",
    }
    missing = [name for name, path in ckpt_paths.items() if not path.exists()]
    if missing:
        missing_list = ", ".join(str(ckpt_paths[name]) for name in missing)
        raise FileNotFoundError(f"Missing checkpoint(s): {missing_list}")

    expected_dim_z = ERA5_settings["state_feature_dim"][-1]
    encoder = ERA5Encoder(dim_z=expected_dim_z).to(device)
    decoder = ERA5Decoder(dim_z=expected_dim_z).to(device)
    encoder.load_state_dict(torch.load(ckpt_paths["encoder"], map_location=device, weights_only=True))
    decoder.load_state_dict(torch.load(ckpt_paths["decoder"], map_location=device, weights_only=True))

    sigma_path = results_dir / "sigma_hat.npy"
    if not sigma_path.exists():
        raise FileNotFoundError(f"sigma_hat.npy not found at {sigma_path}")
    sigma_hat = torch.tensor(np.load(sigma_path), dtype=torch.float32)

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

    H, W = int(dataset.H), int(dataset.W)
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
        raise ValueError(f"sigma_hat length {sigma_hat.numel()} does not match expected dim_u1+dim_z ({dim_u1 + latent_dim}).")

    cgn = CGN(dim_u1=dim_u1, dim_z=latent_dim, hidden=128).to(device)
    cgn.load_state_dict(torch.load(ckpt_paths["cgn"], map_location=device, weights_only=True))
    ode_func = CGKN_ODE(cgn).to(device)

    u1_full = probe_sampler.sample(normalized_groundtruth.unsqueeze(0))
    assert u1_full.shape[1] == total_frames, "Probe sampling length mismatch"
    u1_gt = u1_full[0]  # [T, dim_u1]

    update_mask = update_mask.to(device)

    mu0 = torch.zeros(latent_dim, 1, device=device)
    R0 = 1e-2 * torch.eye(latent_dim, device=device)

    run_metrics = {"mse": [], "rrmse": [], "ssim": []}
    first_run_states = None
    first_run_original_states = None
    run_times: List[float] = []

    for run_idx in range(num_runs):
        print(f"\nStarting assimilation run {run_idx + 1}/{num_runs}")
        set_seed(42 + run_idx)

        # --- (NEW) build noisy observations for this run ---
        u1_obs = torch.full_like(u1_gt, float("nan"))

        for step in observation_steps:
            obs = u1_gt[step].clone()
            if obs_noise_std > 0:
                obs = obs + torch.randn_like(obs) * obs_noise_std
            u1_obs[step] = obs

        # initial observed u1 at t=0 if available; otherwise use truth
        if 0 in observation_steps:
            u1_init = u1_obs[0].clone()
        else:
            u1_init = u1_gt[0].clone()

        obs_series = u1_obs.unsqueeze(-1).to(device)   # [T, dim_u1, 1]

        run_start = perf_counter()
        obs_var = observation_variance if observation_variance is not None else obs_noise_std**2
        mu_post, _, u1_used = CGFilter(
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
        v0 = encoder(normalized_groundtruth[:1].unsqueeze(0).to(device))[:, 0, :]
        uext0 = torch.cat([u1_init.to(device).unsqueeze(0), v0], dim=-1)
        uext_pred = torchdiffeq.odeint(ode_func, uext0, tspan, method="rk4", options={"step_size": dt})
        uext_pred = uext_pred.transpose(0, 1)
        v_pred = uext_pred[:, :, dim_u1:]
        noda_fields = decoder(v_pred).squeeze(0).detach().cpu()[1:]

        if first_run_states is None:
            first_run_states = da_stack.clone()
            first_run_original_states = noda_fields.clone()

        metrics = compute_metrics(da_stack, noda_fields, groundtruth, dataset)
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
        np.save(save_dir / prefixed("multi.npy"), safe_denorm(first_run_states, dataset).numpy())
        print(f"Saved sample DA trajectory to {save_dir / prefixed('multi.npy')}")
    
    if first_run_original_states is not None:
        np.save(
            save_dir / prefixed("multi_original.npy"),
            safe_denorm(first_run_original_states, dataset).numpy(),
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
