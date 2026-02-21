"""
Utility helpers for CGKN data assimilation on Kolmogorov flow.
"""

import math
import os
import sys
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

# Add src directory to path
current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
sys.path.append(src_directory)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_config")

from src.models.KKR.KMG.kol_model import (
    KOL_K_S,
    KOL_K_S_preimage,
    KOL_settings,
)


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
            print(
                f"[WARN] Only placed {len(coords)} of {n} requested probe points after {attempts} attempts."
            )
    return coords


class ProbeSampler:
    def __init__(self, coords: Sequence[Tuple[int, int]], channels: Sequence[int]):
        self.coords = list(coords)
        self.channels = list(channels)
        self.dim_u1 = len(self.coords) * len(self.channels)

    def _gather_single(self, frame: torch.Tensor) -> torch.Tensor:
        values = []
        for c in self.channels:
            chan = frame[c]
            for (r, cc) in self.coords:
                values.append(chan[..., r, cc])
        return torch.stack(values)

    def sample(self, seq: torch.Tensor) -> torch.Tensor:
        B, T, _, _, _ = seq.shape
        out = torch.zeros(B, T, self.dim_u1, device=seq.device)
        for b in range(B):
            for t in range(T):
                out[b, t] = self._gather_single(seq[b, t])
        return out


class KMGEncoder(nn.Module):
    """Encode Kolmogorov fields into KMG latent representations."""

    def __init__(self, dim_z: int):
        super().__init__()
        self.encoder = KOL_K_S()
        expected_dim = KOL_settings["state_feature_dim"][-1]
        if dim_z != expected_dim:
            raise ValueError(f"KMGEncoder latent dim should be {expected_dim}, got {dim_z}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _, _, _ = x.shape
        z_list = []
        for t in range(T):
            frame = x[:, t]
            z = self.encoder(frame)
            z_list.append(z.unsqueeze(1))
        return torch.cat(z_list, dim=1)


class KMGDecoder(nn.Module):
    """Decode KMG latent representations into Kolmogorov fields."""

    def __init__(self, dim_z: int):
        super().__init__()
        self.decoder = KOL_K_S_preimage()
        expected_dim = KOL_settings["state_feature_dim"][-1]
        if dim_z != expected_dim:
            raise ValueError(f"KMGDecoder latent dim should be {expected_dim}, got {dim_z}")

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        B, T, _ = z_seq.shape
        frames = []
        for t in range(T):
            z = z_seq[:, t]
            frame = self.decoder(z)
            frames.append(frame.unsqueeze(1))
        return torch.cat(frames, dim=1)


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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = obs_series.device
    T, dim_u1, _ = obs_series.shape
    dim_z = mu0.shape[0]
    sigma = clamp_sigma_sections(
        sigma,
        dim_u1,
        obs_min=1e-6,
        obs_max=None,
        lat_min=1e-6,
        lat_max=None,
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

        drift = (f2 + g2 @ mu_prev) * dt
        if kalman_drift_clip is not None:
            drift = drift.clamp(min=-kalman_drift_clip, max=kalman_drift_clip)

        if obs_prev is None or obs_curr is None:
            du1 = torch.zeros_like(u1_prev).unsqueeze(-1)
        else:
            du1 = (obs_curr - obs_prev).unsqueeze(-1)
        innovation = du1 - (f1 + g1 @ mu_prev) * dt
        if kalman_innovation_clip is not None:
            innovation = innovation.clamp(min=-kalman_innovation_clip, max=kalman_innovation_clip)

        gain = R_prev @ g1.T
        if kalman_gain_clip is not None:
            gain = gain.clamp(min=-kalman_gain_clip, max=kalman_gain_clip)
        K = gain @ S_inv
        if kalman_gain_scale != 1.0:
            K = K * kalman_gain_scale

        mu_curr = mu_prev + drift + K @ innovation
        if kalman_state_clip is not None:
            mu_curr = mu_curr.clamp(min=-kalman_state_clip, max=kalman_state_clip)

        cov_drift = g2 @ R_prev + R_prev @ g2.T + s2_cov
        assim = gain @ S_inv @ gain.T
        if kalman_gain_scale != 1.0:
            assim = assim * (kalman_gain_scale ** 2)
        R_curr = R_prev + (cov_drift - assim) * dt
        R_curr = stabilise_covariance(
            R_curr,
            min_var=None,
            max_var=None,
            cov_clip=kalman_cov_clip,
            use_diag=use_diag_cov,
        )

        mu_prev = mu_curr
        R_prev = R_curr
        mu_post.append(mu_curr.unsqueeze(0))
        R_post.append(R_curr.unsqueeze(0))
        u1_forcing.append(u1_curr.unsqueeze(0))

        if debug and (not torch.isfinite(mu_curr).all() or not torch.isfinite(R_curr).all()):
            print("[WARN] Non-finite values detected in CGFilter.")

    mu_post = torch.cat(mu_post, dim=0)
    R_post = torch.cat(R_post, dim=0)
    u1_forcing = torch.cat(u1_forcing, dim=0)
    return mu_post, R_post, u1_forcing
