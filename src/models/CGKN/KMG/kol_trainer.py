# kol_cgkn_train.py
# CGKN for Kolmogorov flow with sparse observations as u1 and CAE-ROM latent v
# Structure adapted from the author's L96 CGKN reference implementation. :contentReference[oaicite:2]{index=2}

import os
import time
import math
import numpy as np
import yaml
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ODE integrator (same as in L96 script) :contentReference[oaicite:3]{index=3}
import torchdiffeq

# === Your dataset class is assumed available in the path ===
import os
import sys
current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
sys.path.append(src_directory)

from src.utils.Dataset import KolDynamicsDataset
# As you pasted in chat, we assume it's importable in this script context.
from torch.utils.data import Dataset  # just for type hints if needed

# === PREP FOR CAE-ROM ===
from src.models.KKR.KMG.kol_model import (
    KOL_K_S,
    KOL_K_S_preimage,
    KOL_settings,
)


class KMGEncoder(nn.Module):
    """
    Wrap KOL_K_S to produce latent representations for every frame in a sequence.
    Input:  [B, T, C, H, W]
    Output: [B, T, dim_z]
    """

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
    """
    Wrap KOL_K_S_preimage to reconstruct frames from latent representations.
    Input:  [B, T, dim_z]
    Output: [B, T, C, H, W]
    """

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

# -------------------------
# Configs
# -------------------------
class Config:
    pass


def load_config() -> Config:
    config_path = "../../../../configs/CGKN.yaml"
    with open(config_path, "r") as handle:
        config = yaml.safe_load(handle)["Kolmogorov"]

    cfg_obj = Config()
    for key, value in config.items():
        setattr(cfg_obj, key, value)
    if getattr(cfg_obj, "device", "cpu") == "cuda" and not torch.cuda.is_available():
        cfg_obj.device = "cpu"
    elif getattr(cfg_obj, "device", "cpu") == "cuda":
        cfg_obj.device = "cuda:0"
    os.makedirs(cfg_obj.out_dir, exist_ok=True)
    return cfg_obj


cfg = load_config()

# -------------------------
# Utils
# -------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seed(cfg.seed)

def nrmse(true: torch.Tensor, pred: torch.Tensor) -> float:
    var = torch.var(true.reshape(-1), unbiased=False) + 1e-12
    mse = torch.mean((true - pred)**2)
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
    return (f"{name}: shape={shape}, finite={finite_count}/{total}, nonfinite={non_finite}, "
            f"min={stats['min']:.3e}, max={stats['max']:.3e}, "
            f"mean={stats['mean']:.3e}, std={stats['std']:.3e}, norm={stats['norm']:.3e}")


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
        sigma[:dim_u1] = sigma[:dim_u1].clamp(min=obs_min if obs_min is not None else -float("inf"),
                                              max=obs_max if obs_max is not None else float("inf"))
    if lat_min is not None or lat_max is not None:
        sigma[dim_u1:] = sigma[dim_u1:].clamp(min=lat_min if lat_min is not None else -float("inf"),
                                              max=lat_max if lat_max is not None else float("inf"))
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
            diag = diag.clamp(min=min_var if min_var is not None else -float("inf"),
                              max=max_var if max_var is not None else float("inf"))
        R = torch.diag(diag)
    else:
        R = 0.5 * (R + R.T)
        if cov_clip is not None:
            R = torch.clamp(R, min=-cov_clip, max=cov_clip)
        if min_var is not None or max_var is not None:
            diag = torch.diag(R).clamp(min=min_var if min_var is not None else -float("inf"),
                                       max=max_var if max_var is not None else float("inf"))
            R = R - torch.diag(torch.diag(R)) + torch.diag(diag)
    if cov_clip is not None and use_diag:
        R = torch.clamp(R, min=-cov_clip, max=cov_clip)
    return R

def make_probe_coords_from_ratio(H: int, W: int, ratio: float,
                                 layout: str = "uniform",
                                 seed: int = 42,
                                 min_spacing: int = 4) -> List[Tuple[int, int]]:
    """按比率从 HxW 网格上选取像素坐标 (row, col)。"""
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
        elif len(coords) < n:
            rng = np.random.default_rng(seed)
            remain = n - len(coords)
            all_idx = [(i, j) for i in range(H) for j in range(W)]
            rng.shuffle(all_idx)
            pick = []
            used = set(coords)
            for rc in all_idx:
                if rc not in used:
                    pick.append(rc)
                    used.add(rc)
                if len(pick) >= remain:
                    break
            coords += pick
        return coords

    elif layout == "random":
        rng = np.random.default_rng(seed)
        coords = []
        used = np.zeros((H, W), dtype=bool)
        trials = 0
        max_trials = 50_000
        while len(coords) < n and trials < max_trials:
            trials += 1
            r0 = int(rng.integers(0, H))
            c0 = int(rng.integers(0, W))
            if used[r0, c0]:
                continue
            rmin = max(0, r0 - min_spacing)
            rmax = min(H - 1, r0 + min_spacing)
            cmin = max(0, c0 - min_spacing)
            cmax = min(W - 1, c0 + min_spacing)
            if used[rmin:rmax+1, cmin:cmax+1].any():
                continue
            coords.append((r0, c0))
            used[r0, c0] = True
        if len(coords) < n:
            remain = n - len(coords)
            all_idx = [(i, j) for i in range(H) for j in range(W) if not used[i, j]]
            rng.shuffle(all_idx)
            coords += all_idx[:remain]
        return coords

    else:
        raise ValueError(f"Unknown obs_layout: {layout}")

class ProbeSampler:
    """Extract sparse observations u1 from frames."""
    def __init__(self, probe_coords: List[Tuple[int,int]], channels: List[int]):
        self.coords = list(dict.fromkeys(probe_coords))
        self.channels = channels
        self.dim_u1 = len(self.coords) * len(self.channels)

    def sample(self, frames: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = frames.shape
        assert max(self.channels) < C, "use_channels 索引越界"
        values = []
        for (r, c) in self.coords:
            v = frames[..., r, c]
            v = v[..., self.channels]
            values.append(v)
        u1 = torch.cat(values, dim=-1)
        return u1

# -------------------------
# Data
# -------------------------
class KolDynamicsDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset: KolDynamicsDataset, indices: np.ndarray):
        self.base = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        pre_seq, post_seq = self.base[self.indices[i]]
        return pre_seq, post_seq

def build_dataloaders():
    base = KolDynamicsDataset(
        data_path=cfg.data_path,
        seq_length=cfg.seq_length,
    )
    N = len(base)
    idx = np.arange(N)
    np.random.shuffle(idx)
    ntr = int(N * cfg.train_split)
    tr_idx, te_idx = idx[:ntr], idx[ntr:]

    train_set = KolDynamicsDatasetWrapper(base, tr_idx)
    test_set  = KolDynamicsDatasetWrapper(base, te_idx)

    train_loader = DataLoader(train_set, batch_size=cfg.bs, shuffle=True,
                              num_workers=cfg.num_workers, drop_last=True)
    test_loader  = DataLoader(test_set, batch_size=cfg.bs, shuffle=False,
                              num_workers=cfg.num_workers, drop_last=False)
    return base, train_loader, test_loader

# -------------------------
# CGKN components
# -------------------------
class CGN(nn.Module):
    """
    Dynamics networks to parameterize:
      du1/dt = f1(u1) + g1(u1) v
      dv/dt  = f2(u1) + g2(u1) v
    For simplicity we output dense g1, g2 (can be constrained/sparse later).
    """
    def __init__(self, dim_u1: int, dim_z: int, hidden: int = 128):
        super().__init__()
        self.dim_u1 = dim_u1
        self.dim_z  = dim_z

        def mlp(in_d, out_d):
            return nn.Sequential(
                nn.Linear(in_d, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, out_d)
            )

        self.net_f1 = mlp(dim_u1, dim_u1)
        self.net_g1 = mlp(dim_u1, dim_u1 * dim_z)   # flattened
        self.net_f2 = mlp(dim_u1, dim_z)
        self.net_g2 = mlp(dim_u1, dim_z * dim_z)    # flattened

    def forward(self, u1: torch.Tensor):
        """
        u1: [B, dim_u1]
        returns:
          f1: [B, dim_u1, 1]
          g1: [B, dim_u1, dim_z]
          f2: [B, dim_z, 1]
          g2: [B, dim_z, dim_z]
        """
        B = u1.shape[0]
        f1 = self.net_f1(u1).unsqueeze(-1)
        g1 = self.net_g1(u1).view(B, self.dim_u1, self.dim_z)
        f2 = self.net_f2(u1).unsqueeze(-1)
        g2 = self.net_g2(u1).view(B, self.dim_z, self.dim_z)
        return f1, g1, f2, g2

class CGKN_ODE(nn.Module):
    """
    ODE right-hand side for torchdiffeq.odeint
    Mirrors the L96 code's CGKN forward logic in matrix form. :contentReference[oaicite:5]{index=5}
    """
    def __init__(self, cgn: CGN):
        super().__init__()
        self.cgn = cgn

    def forward(self, t, uext: torch.Tensor):
        """
        uext: [B, dim_u1 + dim_z]
        """
        dim_u1 = self.cgn.dim_u1
        u1 = uext[:, :dim_u1]                        # [B, dim_u1]
        v  = uext[:, dim_u1:]                        # [B, dim_z]
        f1, g1, f2, g2 = self.cgn(u1)                # f1:[B,dim_u1,1], g1:[B,dim_u1,dim_z]
        v_col = v.unsqueeze(-1)                      # [B,dim_z,1]
        u1_dot = f1 + torch.bmm(g1, v_col)           # [B,dim_u1,1]
        v_dot  = f2 + torch.bmm(g2, v_col)           # [B,dim_z,1]
        return torch.cat([u1_dot.squeeze(-1), v_dot.squeeze(-1)], dim=-1)

# -------------------------
# CGFilter : analytic DA (Eq. 2.12) in discrete-time stepping
# (structure ported from L96 script) :contentReference[oaicite:6]{index=6}
# -------------------------
@torch.no_grad()
def CGFilter(cgn: CGN,
             sigma: torch.Tensor,
             u1_series: torch.Tensor,
             mu0: torch.Tensor,
             R0: torch.Tensor,
             dt: float,
             debug: bool = False,
             debug_prefix: str = ""):
    """
    u1_series: [T, dim_u1, 1]
    sigma:     [dim_u1 + dim_z] -> we use diag for s1,s2 as in L96 code
    mu0:       [dim_z, 1]
    R0:        [dim_z, dim_z]
    returns:
      mu_post: [T, dim_z, 1]
      R_post:  [T, dim_z, dim_z]
    """
    device = u1_series.device
    T, dim_u1, _ = u1_series.shape
    dim_z = mu0.shape[0]
    sigma = clamp_sigma_sections(sigma, dim_u1)
    s1 = torch.diag(sigma[:dim_u1]).to(device)
    s2 = torch.diag(sigma[dim_u1:]).to(device)
    eps = getattr(cfg, 'kalman_reg', 0.0)
    eye_u1 = torch.eye(dim_u1, device=device)
    eye_z = torch.eye(dim_z, device=device)
    s1_cov = s1 @ s1.T + eps * eye_u1
    s2_cov = s2 @ s2.T + eps * eye_z
    invs1os1 = torch.linalg.inv(s1_cov)
    s2os2 = s2_cov

    mu_post = torch.zeros((T, dim_z, 1), device=device)
    R_post  = torch.zeros((T, dim_z, dim_z), device=device)
    mu = mu0.clone()
    R  = R0.clone()
    R = stabilise_covariance(R)
    mu_post[0] = mu
    R_post[0]  = R

    if debug:
        print(f"{debug_prefix} CGFilter init { _tensor_summary('sigma', sigma.cpu()) }")
        print(f"{debug_prefix} CGFilter init { _tensor_summary('u1_series', u1_series.cpu()) }")

    for n in range(1, T):
        u1_prev = u1_series[n-1].T  # [1, dim_u1]
        f1, g1, f2, g2 = [e.squeeze(0) for e in cgn(u1_prev)]  # remove batch
        du1 = u1_series[n] - u1_series[n-1]                    # [dim_u1,1]
        innovation = du1 - (f1 + g1 @ mu) * dt
        innovation_clip = getattr(cfg, 'kalman_innovation_clip', None)
        if innovation_clip is not None and innovation_clip > 0:
            innovation = torch.clamp(innovation, min=-innovation_clip, max=innovation_clip)
        drift = (f2 + g2 @ mu) * dt
        drift_clip = getattr(cfg, 'kalman_drift_clip', None)
        if drift_clip is not None and drift_clip > 0:
            drift = torch.clamp(drift, min=-drift_clip, max=drift_clip)
        gain = R @ g1.T
        gain_clip = getattr(cfg, 'kalman_gain_clip', None)
        if gain_clip is not None and gain_clip > 0:
            gain = torch.clamp(gain, min=-gain_clip, max=gain_clip)
        K = gain @ invs1os1
        gain_scale = getattr(cfg, 'kalman_gain_scale', 1.0)
        if gain_clip is not None and gain_clip > 0:
            K = torch.clamp(K, min=-gain_clip, max=gain_clip)
        if gain_scale != 1.0:
            K = K * gain_scale
        mu1 = mu + drift + K @ innovation
        state_clip = getattr(cfg, 'kalman_state_clip', None)
        if state_clip is not None and state_clip > 0:
            mu1 = torch.clamp(mu1, min=-state_clip, max=state_clip)
        cov_drift = g2 @ R + R @ g2.T + s2os2
        assim = gain @ invs1os1 @ gain.T
        if gain_scale != 1.0:
            assim = assim * (gain_scale ** 2)
        R1  = R + (cov_drift - assim) * dt
        R1 = stabilise_covariance(R1)
        if debug and (n == 1 or not torch.isfinite(mu1).all() or not torch.isfinite(R1).all()):
            print(f"{debug_prefix} CGFilter step={n}")
            print(f"{debug_prefix}   { _tensor_summary('du1', du1.cpu()) }")
            print(f"{debug_prefix}   { _tensor_summary('f1', f1.cpu()) }")
            print(f"{debug_prefix}   { _tensor_summary('g1', g1.cpu()) }")
            print(f"{debug_prefix}   { _tensor_summary('f2', f2.cpu()) }")
            print(f"{debug_prefix}   { _tensor_summary('g2', g2.cpu()) }")
            print(f"{debug_prefix}   { _tensor_summary('mu', mu.cpu()) }")
            print(f"{debug_prefix}   { _tensor_summary('mu1', mu1.cpu()) }")
            print(f"{debug_prefix}   { _tensor_summary('R', R.cpu()) }")
            print(f"{debug_prefix}   { _tensor_summary('R1', R1.cpu()) }")
        mu, R = mu1, R1
        mu_post[n] = mu
        R_post[n]  = R
    return mu_post, R_post

# -------------------------
# Training / Eval
# -------------------------
def train_stage1(train_loader, encoder, decoder, cgn, ode_func, probe_sampler, denorm, save_prefix="stage1"):
    encoder.train(); decoder.train(); cgn.train()
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(cgn.parameters()),
                           lr=cfg.s1_lr)

    hist = {"forecast": [], "ae": [], "latent_forecast": []}

    for ep in range(1, cfg.s1_epochs + 1):
        t0 = time.time()
        ep_fore, ep_ae, ep_lf = 0.0, 0.0, 0.0
        nb = 0
        for pre_seq, _ in train_loader:
            nb += 1
            # pre_seq: [B, T, C, H, W]
            pre_seq = pre_seq.to(cfg.device)

            # === u1 (sparse obs) ===
            u1 = probe_sampler.sample(pre_seq)  # [B,T,dim_u1]
            # === v (latent) ===
            # Encoder should support time-wise encoding: here we pass [B,T,C,H,W] -> [B,T,dim_z]
            v = encoder(pre_seq)                # [B,T,dim_z]

            # AE recon
            rec = decoder(v)                    # [B,T,C,H,W]
            loss_ae = F.mse_loss(pre_seq, rec)

            # Short forecast on [0:K]
            K = min(cfg.s1_short_steps, pre_seq.shape[1])  # safety
            tspan = torch.linspace(0.0, (K-1)*cfg.dt, K, device=cfg.device)

            uext0 = torch.cat([u1[:,0,:], v[:,0,:]], dim=-1)     # [B, dim_u1+dim_z]
            uext_pred = torchdiffeq.odeint(ode_func, uext0, tspan, method="rk4",
                                           options={"step_size": cfg.dt})      # [K,B,dim]
            uext_pred = uext_pred.transpose(0,1)                               # [B,K,dim]
            v_pred = uext_pred[:,:,probe_sampler.sample(pre_seq[:,:1]).shape[-1]:]  # [B,K,dim_z]

            # latent forecast consistency
            loss_latent = F.mse_loss(v[:, :K, :], v_pred)

            # reconstruct forecast field
            field_pred = decoder(v_pred)         # [B,K,C,H,W]
            # build u_pred (we primarily supervise via field MSE)
            loss_forecast = F.mse_loss(pre_seq[:, :K], field_pred)

            loss = (cfg.lam_ae * loss_ae +
                    cfg.lam_latent_forecast * loss_latent +
                    cfg.lam_forecast * loss_forecast)

            opt.zero_grad()
            loss.backward()
            opt.step()

            ep_fore += loss_forecast.item()
            ep_ae   += loss_ae.item()
            ep_lf   += loss_latent.item()

        hist["forecast"].append(ep_fore/nb)
        hist["ae"].append(ep_ae/nb)
        hist["latent_forecast"].append(ep_lf/nb)
        print(f"[S1][{ep:04d}] t={time.time()-t0:.2f}s  "
              f"fore={hist['forecast'][-1]:.4e}  ae={hist['ae'][-1]:.4e}  z-fore={hist['latent_forecast'][-1]:.4e}")

    # save
    torch.save(encoder.state_dict(), os.path.join(cfg.out_dir, f"{save_prefix}_encoder.pt"))
    torch.save(decoder.state_dict(), os.path.join(cfg.out_dir, f"{save_prefix}_decoder.pt"))
    torch.save(cgn.state_dict(),     os.path.join(cfg.out_dir, f"{save_prefix}_cgn.pt"))
    np.save(os.path.join(cfg.out_dir, f"{save_prefix}_hist.npy"), hist)
    return hist



def load_stage1_checkpoints(encoder, decoder, cgn, prefix: str, directory: str, map_location):
    """Load pretrained Stage 1 checkpoints into the provided modules."""
    ckpt_paths = {
        "encoder": os.path.join(directory, f"{prefix}_encoder.pt"),
        "decoder": os.path.join(directory, f"{prefix}_decoder.pt"),
        "cgn": os.path.join(directory, f"{prefix}_cgn.pt"),
    }
    missing = [name for name, path_str in ckpt_paths.items() if not os.path.exists(path_str)]
    if missing:
        missing_paths = ", ".join(ckpt_paths[name] for name in missing)
        raise FileNotFoundError(f"Missing Stage 1 checkpoint(s): {missing_paths}")
    encoder.load_state_dict(torch.load(ckpt_paths["encoder"], map_location=map_location, weights_only=True))
    decoder.load_state_dict(torch.load(ckpt_paths["decoder"], map_location=map_location, weights_only=True))
    cgn.load_state_dict(torch.load(ckpt_paths["cgn"], map_location=map_location, weights_only=True))
    return ckpt_paths

@torch.no_grad()
def estimate_noise_sigma(train_loader, encoder, decoder, cgn, probe_sampler):
    """
    Estimate sigma via quadratic variation (analogue of Eq. (2.19) idea used in L96 code). :contentReference[oaicite:7]{index=7}
    We compute derivatives from data vs model on u1 and v, then take RMS.
    """
    encoder.eval(); decoder.eval(); cgn.eval()
    dim_u1 = probe_sampler.dim_u1
    dim_z  = cfg.dim_z

    diffs = []
    for pre_seq, _ in train_loader:
        pre_seq = pre_seq.to(cfg.device)                   # [B,T,C,H,W]
        u1 = probe_sampler.sample(pre_seq)                 # [B,T,dim_u1]
        v  = encoder(pre_seq)                              # [B,T,dim_z]

        # finite diff
        du1 = (u1[:,1:,:] - u1[:,:-1,:]) / cfg.dt         # [B,T-1,dim_u1]
        dv  = (v[:,1:,:]  - v[:,:-1,:])  / cfg.dt         # [B,T-1,dim_z]

        # model drift at t-1
        u1_mid = u1[:,:-1,:].reshape(-1, dim_u1)
        f1,g1,f2,g2 = cgn(u1_mid)                         # f1:[N,dim_u1,1] ...
        v_mid  = v[:,:-1,:].reshape(-1, dim_z).unsqueeze(-1)
        du1_model = (f1 + torch.bmm(g1, v_mid)).squeeze(-1)  # [N,dim_u1]
        dv_model  = (f2 + torch.bmm(g2, v_mid)).squeeze(-1)  # [N,dim_z]
        du1_model = du1_model.view(du1.shape)
        dv_model  = dv_model.view(dv.shape)

        diff_u1 = (du1 - du1_model)
        diff_v  = (dv  - dv_model)
        diffs.append(torch.cat([diff_u1, diff_v], dim=-1)) # [B,T-1, dim_u1+dim_z]

    diffs = torch.cat(diffs, dim=0).reshape(-1, dim_u1+dim_z)
    sigma_hat = torch.sqrt(cfg.dt * torch.mean(diffs**2, dim=0))  # [dim_u1+dim_z]
    # As in L96 code, we can pin sigma_v if desired; here keep as estimated for start. :contentReference[oaicite:8]{index=8}
    sigma_hat = clamp_sigma_sections(sigma_hat, dim_u1)
    return sigma_hat.detach()

def train_stage2(train_loader, encoder, decoder, cgn, ode_func, probe_sampler, sigma_hat, save_prefix="stage2"):
    encoder.train(); decoder.train(); cgn.train()
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(cgn.parameters()),
                           lr=cfg.s2_lr)

    hist = {"forecast": [], "ae": [], "da": [], "latent_forecast": []}

    dim_u1 = probe_sampler.dim_u1
    dim_z  = cfg.dim_z
    sigma_device = clamp_sigma_sections(sigma_hat, dim_u1).to(cfg.device)

    for ep in range(1, cfg.s2_epochs + 1):
        t0 = time.time()
        nb = 0
        ep_fore=ep_ae=ep_da=ep_lf=0.0

        for batch_idx, (pre_seq, _) in enumerate(train_loader, start=1):
            nb += 1
            debug_active = bool(cfg.debug_stage2 and batch_idx <= cfg.debug_batches)
            debug_prefix = f"[DEBUG][S2][ep{ep:04d}][batch{batch_idx:03d}]"
            pre_seq = pre_seq.to(cfg.device)                  # [B,T,C,H,W]
            B, T, C, H, W = pre_seq.shape
            K = min(cfg.s2_short_steps, T)

            # === u1 and v ===
            u1 = probe_sampler.sample(pre_seq)                # [B,T,dim_u1]
            v  = encoder(pre_seq)                             # [B,T,dim_z]
            if debug_active:
                print(f"{debug_prefix} { _tensor_summary('pre_seq[0]', pre_seq[0]) }")
                print(f"{debug_prefix} { _tensor_summary('u1[0]', u1[0]) }")
                print(f"{debug_prefix} { _tensor_summary('v[0]', v[0]) }")
                print(f"{debug_prefix} { _tensor_summary('sigma_hat', sigma_device) }")

            # --- Short forecast terms (as S1) ---
            tspan = torch.linspace(0.0, (K-1)*cfg.dt, K, device=cfg.device)
            uext0 = torch.cat([u1[:,0,:], v[:,0,:]], dim=-1)
            uext_pred = torchdiffeq.odeint(ode_func, uext0, tspan, method="rk4",
                                           options={"step_size": cfg.dt}).transpose(0,1)
            v_pred = uext_pred[:,:,dim_u1:]
            loss_latent = F.mse_loss(v[:, :K, :], v_pred)
            field_pred = decoder(v_pred)
            loss_forecast = F.mse_loss(pre_seq[:, :K], field_pred)

            # --- Autoencoder recon ---
            rec = decoder(v)
            loss_ae = F.mse_loss(pre_seq, rec)

            # --- Long-horizon DA term via analytic filter (Eq. 2.12) ---
            L = min(cfg.s2_long_steps, T)
            cut = min(cfg.s2_cut_warmup, L-1)
            # filter expects u1 series as [T,dim_u1,1]
            mu0 = torch.zeros(dim_z, 1, device=cfg.device)
            R0  = 1e-2 * torch.eye(dim_z, device=cfg.device)
            loss_da_batch = 0.0
            for sample_idx in range(B):
                sample_prefix = f"{debug_prefix}[sample{sample_idx:03d}]"
                mu_post, _ = CGFilter(
                    cgn, sigma_device,
                    u1[sample_idx,:L,:].unsqueeze(-1), mu0, R0, cfg.dt,
                    debug=debug_active and sample_idx == 0,
                    debug_prefix=sample_prefix
                )  # [L, dim_z, 1]
                if not torch.isfinite(mu_post).all():
                    print(f"{sample_prefix} mu_post contains non-finite values")
                    print(f"{sample_prefix} { _tensor_summary('mu_post', mu_post) }")
                    raise RuntimeError("CGFilter produced non-finite mu_post")
                mu_v = mu_post.squeeze(-1)                    # [L, dim_z]
                # decode posterior mean to field
                mu_field = decoder(mu_v.unsqueeze(0))         # [1,L,C,H,W]
                if not torch.isfinite(mu_field).all():
                    print(f"{sample_prefix} mu_field contains non-finite values")
                    print(f"{sample_prefix} { _tensor_summary('mu_field', mu_field) }")
                    raise RuntimeError("Decoder produced non-finite mu_field from filter output")
                # compare with ground truth u2 part (we supervise full field here)
                loss_da_sample = F.mse_loss(pre_seq[sample_idx,cut:L], mu_field[0,cut:L])
                if not torch.isfinite(loss_da_sample):
                    print(f"{sample_prefix} loss_da_sample is non-finite")
                    print(f"{sample_prefix} { _tensor_summary('pre_seq_truth', pre_seq[sample_idx,cut:L]) }")
                    print(f"{sample_prefix} { _tensor_summary('mu_field_pred', mu_field[0,cut:L]) }")
                    raise RuntimeError("DA loss became non-finite")
                if debug_active and sample_idx == 0:
                    print(f"{sample_prefix} loss_da_sample={loss_da_sample.item():.3e}")
                    print(f"{sample_prefix} { _tensor_summary('mu_v', mu_v) }")
                    print(f"{sample_prefix} { _tensor_summary('mu_field', mu_field) }")
                loss_da_batch += loss_da_sample
            loss_da = loss_da_batch / B

            loss = (cfg.lam_forecast * loss_forecast +
                    cfg.lam_ae * loss_ae +
                    cfg.lam_latent_forecast * loss_latent +
                    cfg.lam_da * loss_da)

            if not torch.isfinite(loss):
                print(f"{debug_prefix} loss became non-finite")
                print(f"{debug_prefix} loss_forecast={loss_forecast.item():.3e}")
                print(f"{debug_prefix} loss_ae={loss_ae.item():.3e}")
                print(f"{debug_prefix} loss_latent={loss_latent.item():.3e}")
                print(f"{debug_prefix} loss_da={loss_da.item():.3e}")
                print(f"{debug_prefix} { _tensor_summary('v_pred[0]', v_pred[0]) }")
                print(f"{debug_prefix} { _tensor_summary('field_pred[0]', field_pred[0]) }")
                raise RuntimeError("Stage 2 loss is non-finite")

            if debug_active:
                print(f"{debug_prefix} losses: fore={loss_forecast.item():.3e}, ae={loss_ae.item():.3e}, "
                      f"latent={loss_latent.item():.3e}, da={loss_da.item():.3e}, total={loss.item():.3e}")
                print(f"{debug_prefix} { _tensor_summary('v_pred[0]', v_pred[0]) }")
                print(f"{debug_prefix} { _tensor_summary('field_pred[0]', field_pred[0]) }")
                print(f"{debug_prefix} { _tensor_summary('rec[0]', rec[0]) }")

            opt.zero_grad()
            loss.backward()
            if debug_active:
                print(f"{debug_prefix} {_module_grad_summary('cgn_grad', cgn)}")
                print(f"{debug_prefix} {_module_grad_summary('encoder_grad', encoder)}")
                print(f"{debug_prefix} {_module_grad_summary('decoder_grad', decoder)}")
            opt.step()

            ep_fore += loss_forecast.item()
            ep_ae   += loss_ae.item()
            ep_da   += loss_da.item()
            ep_lf   += loss_latent.item()

        hist["forecast"].append(ep_fore/nb)
        hist["ae"].append(ep_ae/nb)
        hist["da"].append(ep_da/nb)
        hist["latent_forecast"].append(ep_lf/nb)
        print(f"[S2][{ep:04d}] t={time.time()-t0:.2f}s  "
              f"fore={hist['forecast'][-1]:.4e}  ae={hist['ae'][-1]:.4e}  "
              f"DA={hist['da'][-1]:.4e}  z-fore={hist['latent_forecast'][-1]:.4e}")

    # save
    torch.save(encoder.state_dict(), os.path.join(cfg.out_dir, f"{save_prefix}_encoder.pt"))
    torch.save(decoder.state_dict(), os.path.join(cfg.out_dir, f"{save_prefix}_decoder.pt"))
    torch.save(cgn.state_dict(),     os.path.join(cfg.out_dir, f"{save_prefix}_cgn.pt"))
    np.save(os.path.join(cfg.out_dir, f"{save_prefix}_hist.npy"), hist)
    return hist

# -------------------------
# Evaluation helpers
# -------------------------
@torch.no_grad()
def eval_short_forecast(test_loader, encoder, decoder, cgn, ode_func, probe_sampler):
    encoder.eval(); decoder.eval(); cgn.eval()
    errs=[]
    for pre_seq, _ in test_loader:
        pre_seq = pre_seq.to(cfg.device)
        u1 = probe_sampler.sample(pre_seq)
        v  = encoder(pre_seq)
        K = min(cfg.s1_short_steps, pre_seq.shape[1])
        tspan = torch.linspace(0.0, (K-1)*cfg.dt, K, device=cfg.device)
        uext0 = torch.cat([u1[:,0,:], v[:,0,:]], dim=-1)
        uext_pred = torchdiffeq.odeint(ode_func, uext0, tspan, method="rk4",
                                       options={"step_size": cfg.dt}).transpose(0,1)
        v_pred = uext_pred[:,:,u1.shape[-1]:]
        field_pred = decoder(v_pred)
        errs.append(nrmse(pre_seq[:,:K], field_pred))
    e = float(np.mean(errs))
    print(f"[EVAL] short-forecast NRMSE: {e:.4f}")
    return e

@torch.no_grad()
def eval_DA(test_loader, encoder, decoder, cgn, probe_sampler, sigma_hat):
    encoder.eval(); decoder.eval(); cgn.eval()
    dim_u1 = probe_sampler.dim_u1
    dim_z  = cfg.dim_z
    errs=[]
    for pre_seq, _ in test_loader:
        pre_seq = pre_seq.to(cfg.device)
        u1 = probe_sampler.sample(pre_seq)
        B,T,_ = u1.shape
        mu0 = torch.zeros(dim_z,1,device=cfg.device); R0=1e-2*torch.eye(dim_z,device=cfg.device)
        L = T
        cut = min(cfg.s2_cut_warmup, L-1)
        for b in range(B):
            mu_post,_ = CGFilter(cgn, sigma_hat.to(cfg.device), u1[b,:L,:].unsqueeze(-1), mu0, R0, cfg.dt)
            mu_v = mu_post.squeeze(-1)          # [L, dim_z]
            mu_field = decoder(mu_v.unsqueeze(0))  # [1,L,C,H,W]
            errs.append(nrmse(pre_seq[b,cut:L], mu_field[0,cut:L]))
    e=float(np.mean(errs))
    print(f"[EVAL] DA NRMSE (posterior mean): {e:.4f}")
    return e

# -------------------------
# Main
# -------------------------
def main():
    base, train_loader, test_loader = build_dataloaders()
    denorm = base.denormalizer()

    # 根据比率生成观测点
    H, W = int(base.H), int(base.W)
    coords = make_probe_coords_from_ratio(
        H, W, cfg.obs_ratio, layout=cfg.obs_layout, seed=cfg.seed, min_spacing=cfg.min_spacing
    )
    if cfg.save_probe_coords:
        np.save(os.path.join(cfg.out_dir, "probe_coords.npy"), np.array(coords, dtype=np.int32))
    probe_sampler = ProbeSampler(coords, cfg.use_channels)
    print(f"[OBS] dim_u1 = {probe_sampler.dim_u1}  (points={len(coords)}, channels={len(cfg.use_channels)})")

    # 构建模型
    encoder = KMGEncoder(dim_z=cfg.dim_z).to(cfg.device)
    decoder = KMGDecoder(dim_z=cfg.dim_z).to(cfg.device)
    cgn = CGN(dim_u1=probe_sampler.dim_u1, dim_z=cfg.dim_z, hidden=cfg.hidden).to(cfg.device)
    ode_func = CGKN_ODE(cgn).to(cfg.device)

    # Stage 1
    if cfg.use_pretrained_stage1:
        ckpts = load_stage1_checkpoints(
            encoder, decoder, cgn,
            prefix=cfg.stage1_ckpt_prefix,
            directory=cfg.stage1_ckpt_dir,
            map_location=cfg.device
        )
        print(f"[LOAD] Stage 1 checkpoints loaded from {ckpts['encoder']}, {ckpts['decoder']}, {ckpts['cgn']}")
    else:
        print("=== Stage 1 training ===")
        train_stage1(train_loader, encoder, decoder, cgn, ode_func, probe_sampler, denorm)

    # 估计 sigma
    print("Estimating noise sigma ...")
    sigma_hat = estimate_noise_sigma(train_loader, encoder, decoder, cgn, probe_sampler)
    np.save(os.path.join(cfg.out_dir, "sigma_hat.npy"), sigma_hat.cpu().numpy())

    # Stage 2
    print("=== Stage 2 training ===")
    train_stage2(train_loader, encoder, decoder, cgn, ode_func, probe_sampler, sigma_hat)

    # Eval
    eval_short_forecast(test_loader, encoder, decoder, cgn, ode_func, probe_sampler)
    eval_DA(test_loader, encoder, decoder, cgn, probe_sampler, sigma_hat)

if __name__ == "__main__":
    main()
