# era5_high_cgkn_train.py
# CGKN for ERA5 high-resolution data with sparse observations as u1 and CAE-ROM latent v
# Structure adapted from the author's L96 CGKN reference implementation.

import os
import time
import math
import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ODE integrator (same as in L96 script)
import torchdiffeq

import os
import sys
current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
sys.path.append(src_directory)

from src.utils.Dataset import ERA5HighDataset
from torch.utils.data import Dataset  # just for type hints if needed

from era5_high_model import ERA5Encoder, ERA5Decoder


class Config:
    # Paths
    data_path = "../../../../data/ERA5_high/raw_data/weatherbench_train.h5"
    min_path = "../../../../data/ERA5_high/raw_data/era5high_240x121_min.npy"
    max_path = "../../../../data/ERA5_high/raw_data/era5high_240x121_max.npy"
    out_dir = "../../../../results/CGKN/ERA5_High"
    os.makedirs(out_dir, exist_ok=True)

    # Data / time
    seq_length = 50
    dt = 0.001
    train_split = 1.0

    # Observations (u1): ratio-based
    obs_ratio = 0.15
    obs_layout = "random"
    min_spacing = 4
    save_probe_coords = True
    use_channels = [0, 1]

    # Latent
    dim_z = 512
    hidden = 128

    # Training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    bs = 32
    num_workers = 4

    # Stage 1 (short forecast)
    s1_epochs = 20
    s1_short_steps = 6
    s1_lr = 1e-3

    # Stage 2 (add DA)
    s2_epochs = 13
    s2_short_steps = 6
    s2_long_steps = 20
    s2_cut_warmup = 5
    s2_lr = 1e-3

    # Numerical stability
    kalman_reg = 1e-6

    # Stage 1 checkpoint options
    use_pretrained_stage1 = True
    stage1_ckpt_prefix = "stage1"
    stage1_ckpt_dir = out_dir

    # Debugging
    debug_stage2 = False
    debug_batches = 1

    # Kalman / DA stabilisation
    kalman_sigma_min_obs = 1e-2
    kalman_sigma_max_obs = 1.0
    kalman_sigma_min_latent = 1e-2
    kalman_sigma_max_latent = 1.0
    kalman_min_var = 1e-6
    kalman_max_var = 1.0
    kalman_cov_clip = 1.0
    kalman_gain_clip = 50.0
    kalman_gain_scale = 1e-3
    kalman_innovation_clip = 1.0
    kalman_drift_clip = 0.1
    kalman_state_clip = 10.0
    kalman_use_diag_cov = True

    # Loss weights
    lam_ae = 1.0
    lam_forecast = 1.0
    lam_latent_forecast = 1.0
    lam_da = 1.0


cfg = Config()


# -------------------------
# Utils
# -------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


set_seed(cfg.seed)


def nrmse(true: torch.Tensor, pred: torch.Tensor) -> float:
    var = torch.var(true.reshape(-1), unbiased=False) + 1e-12
    mse = torch.mean((true - pred) ** 2)
    return torch.sqrt(mse / var).item()


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


def _module_grad_summary(name: str, module: nn.Module) -> str:
    grads = [p.grad.detach().reshape(-1) for p in module.parameters() if p.grad is not None]
    if not grads:
        return f"{name}: no gradients"
    flat = torch.cat(grads)
    return _tensor_summary(name, flat)


def clamp_sigma_sections(sigma: torch.Tensor, dim_u1: int) -> torch.Tensor:
    """Clamp observation and latent sigma components using config bounds."""
    sigma = sigma.clone()
    obs_min = getattr(cfg, "kalman_sigma_min_obs", None)
    obs_max = getattr(cfg, "kalman_sigma_max_obs", None)
    lat_min = getattr(cfg, "kalman_sigma_min_latent", None)
    lat_max = getattr(cfg, "kalman_sigma_max_latent", None)

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


def stabilise_covariance(R: torch.Tensor) -> torch.Tensor:
    """Keep covariance matrix bounded and (optionally) diagonal."""
    min_var = getattr(cfg, "kalman_min_var", None)
    max_var = getattr(cfg, "kalman_max_var", None)
    cov_clip = getattr(cfg, "kalman_cov_clip", None)
    use_diag = getattr(cfg, "kalman_use_diag_cov", False)

    if use_diag:
        diag = torch.diag(R)
        if min_var is not None or max_var is not None:
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


def build_probe_coords(
    H: int,
    W: int,
    ratio: float,
    layout: str,
    min_spacing: int,
    seed: int,
) -> np.ndarray:
    total = H * W
    n_obs = max(1, int(total * ratio))

    rng = np.random.RandomState(seed)

    if layout == "uniform":
        grid_size = int(np.sqrt(n_obs))
        grid_size = max(1, grid_size)
        xs = np.linspace(0, H - 1, grid_size, dtype=int)
        ys = np.linspace(0, W - 1, grid_size, dtype=int)
        coords = np.array([(x, y) for x in xs for y in ys])
        if coords.shape[0] > n_obs:
            coords = coords[:n_obs]
        return coords

    coords = []
    attempts = 0
    max_attempts = n_obs * 50
    while len(coords) < n_obs and attempts < max_attempts:
        x = rng.randint(0, H)
        y = rng.randint(0, W)
        if all(abs(x - cx) >= min_spacing or abs(y - cy) >= min_spacing for cx, cy in coords):
            coords.append((x, y))
        attempts += 1

    if len(coords) < n_obs:
        for _ in range(n_obs - len(coords)):
            coords.append((rng.randint(0, H), rng.randint(0, W)))
    return np.array(coords)


def extract_obs_from_state(state: torch.Tensor, coords: np.ndarray, channels: List[int]) -> torch.Tensor:
    """Extract observations from state [B,T,C,H,W] at given coords for selected channels."""
    B, T, C, H, W = state.shape
    obs_list = []
    for ch in channels:
        obs_ch = []
        for (x, y) in coords:
            obs_ch.append(state[:, :, ch, x, y].unsqueeze(-1))
        obs_ch = torch.cat(obs_ch, dim=-1)
        obs_list.append(obs_ch)
    obs = torch.cat(obs_list, dim=-1)
    return obs


def build_cgkn_model(dim_u1: int, dim_z: int, hidden: int) -> nn.Module:
    class CGKN(nn.Module):
        def __init__(self):
            super().__init__()
            self.dim_u1 = dim_u1
            self.dim_z = dim_z
            self.encoder = ERA5Encoder(dim_z=dim_z)
            self.decoder = ERA5Decoder(dim_z=dim_z)

            self.net = nn.Sequential(
                nn.Linear(dim_u1 + dim_z, hidden),
                nn.Tanh(),
                nn.Linear(hidden, dim_z),
            )

        def forward(self, u1, z, dt):
            inp = torch.cat([u1, z], dim=-1)
            dz = self.net(inp)
            return z + dz * dt

    return CGKN()


def run_stage1(
    model: nn.Module,
    train_loader: DataLoader,
    coords: np.ndarray,
    device: str,
    dt: float,
    short_steps: int,
    optimizer: torch.optim.Optimizer,
    lam_forecast: float,
):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        seq, _ = batch
        seq = seq.to(device)
        obs = extract_obs_from_state(seq, coords, cfg.use_channels).to(device)
        z = model.encoder(seq)

        loss = 0.0
        for t in range(short_steps):
            u1_t = obs[:, t]
            z_t = z[:, t]
            z_pred = model(u1_t, z_t, dt)
            loss = loss + torch.mean((z_pred - z[:, t + 1]) ** 2)

        loss = lam_forecast * loss / short_steps
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(1, len(train_loader))


def run_stage2(
    model: nn.Module,
    train_loader: DataLoader,
    coords: np.ndarray,
    device: str,
    dt: float,
    short_steps: int,
    long_steps: int,
    optimizer: torch.optim.Optimizer,
    lam_forecast: float,
    lam_latent_forecast: float,
    lam_da: float,
):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        seq, _ = batch
        seq = seq.to(device)
        obs = extract_obs_from_state(seq, coords, cfg.use_channels).to(device)
        z = model.encoder(seq)

        loss = 0.0
        for t in range(short_steps):
            u1_t = obs[:, t]
            z_t = z[:, t]
            z_pred = model(u1_t, z_t, dt)
            loss = loss + torch.mean((z_pred - z[:, t + 1]) ** 2)

        for t in range(long_steps - 1):
            u1_t = obs[:, t]
            z_t = z[:, t]
            z_pred = model(u1_t, z_t, dt)
            loss = loss + lam_latent_forecast * torch.mean((z_pred - z[:, t + 1]) ** 2)

        loss = lam_forecast * loss / short_steps
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(1, len(train_loader))


def train_cgkn():
    device = cfg.device
    dataset = ERA5HighDataset(
        data_path=cfg.data_path,
        seq_length=cfg.seq_length,
        min_path=cfg.min_path,
        max_path=cfg.max_path,
    )
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.bs,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    _, _, H, W = dataset[0][0].shape
    coords = build_probe_coords(
        H=H,
        W=W,
        ratio=cfg.obs_ratio,
        layout=cfg.obs_layout,
        min_spacing=cfg.min_spacing,
        seed=cfg.seed,
    )

    if cfg.save_probe_coords:
        coord_path = os.path.join(cfg.out_dir, "probe_coords.npy")
        np.save(coord_path, coords)

    dim_u1 = len(cfg.use_channels) * coords.shape[0]
    model = build_cgkn_model(dim_u1=dim_u1, dim_z=cfg.dim_z, hidden=cfg.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.s1_lr)

    for epoch in range(cfg.s1_epochs):
        loss = run_stage1(
            model,
            train_loader,
            coords,
            device,
            cfg.dt,
            cfg.s1_short_steps,
            optimizer,
            cfg.lam_forecast,
        )
        print(f"[Stage1][Epoch {epoch + 1}/{cfg.s1_epochs}] Loss: {loss:.6f}")

    stage1_ckpt = os.path.join(cfg.stage1_ckpt_dir, f"{cfg.stage1_ckpt_prefix}.pth")
    torch.save(model.state_dict(), stage1_ckpt)

    if cfg.use_pretrained_stage1:
        model.load_state_dict(torch.load(stage1_ckpt, map_location=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.s2_lr)
    for epoch in range(cfg.s2_epochs):
        loss = run_stage2(
            model,
            train_loader,
            coords,
            device,
            cfg.dt,
            cfg.s2_short_steps,
            cfg.s2_long_steps,
            optimizer,
            cfg.lam_forecast,
            cfg.lam_latent_forecast,
            cfg.lam_da,
        )
        print(f"[Stage2][Epoch {epoch + 1}/{cfg.s2_epochs}] Loss: {loss:.6f}")

    model_path = os.path.join(cfg.out_dir, "cgkn_model.pth")
    torch.save(model.state_dict(), model_path)


def main():
    train_cgkn()


if __name__ == "__main__":
    main()
