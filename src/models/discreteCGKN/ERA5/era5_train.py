"""Two-stage training for discrete CGKN on ERA5 using the discrete one-step mapping.

Stage 1 (lambda_DA = 0):
    * Train reconstruction + short-horizon forecast losses.
    * Estimate sigma_hat from one-step residuals.
    * Save stage1_* checkpoints and config.

Stage 2 (lambda_DA > 0):
    * Load best stage1 checkpoints (encoder/decoder/cgn, sigma_hat, probe coords).
    * Continue training with DA loss over an assimilation horizon with warm-up.
    * Save stage2_* checkpoints and config.

The discrete CGN step follows the NSE reference implementation:
    [u1, v] -> [f1 + g1 v, f2 + g2 v]
where u1 collects probe observations and v is the latent state. All ground-truth
tensor shapes follow [B, T, C, H, W]; latent and probe sequences are reshaped
internally as needed.
"""

import argparse
import json
import math
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Ensure project root is importable for Dataset
current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
sys.path.append(src_directory)

from src.utils.Dataset import ERA5Dataset  # noqa: E402
from era5_model import ERA5Decoder, ERA5Encoder  # noqa: E402
from src.models.CAE_Koopman.ERA5.era5_model_FTF import ERA5_settings  # noqa: E402


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_probe_coords_from_ratio(
    H: int, W: int, ratio: float, layout: str = "uniform", seed: int = 42, min_spacing: int = 4
) -> List[Tuple[int, int]]:
    """按比例从 HxW 网格选择观测坐标 (row, col)。"""
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

    rng = np.random.default_rng(seed)
    coords: List[Tuple[int, int]] = []
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
        if used[rmin : rmax + 1, cmin : cmax + 1].any():
            continue
        coords.append((r0, c0))
        used[r0, c0] = True
    if len(coords) < n:
        remain = n - len(coords)
        all_idx = [(i, j) for i in range(H) for j in range(W) if not used[i, j]]
        rng.shuffle(all_idx)
        coords += all_idx[:remain]
    return coords


class ProbeSampler:
    """
    Extract sparse observations u1 from frames.

    u1 is built by concatenating the selected channels at each probe coordinate:
      frames[..., r, c] -> pick channels -> concatenate over probes.
    Output shape: [B, T, dim_u1].
    """

    def __init__(self, probe_coords: List[Tuple[int, int]], channels: List[int]):
        # Ensure coordinates are tuples for hashing/deduplication (np.load may
        # yield nested Python lists which are unhashable).
        tuple_coords = [tuple(c) for c in probe_coords]
        self.coords = list(dict.fromkeys(tuple_coords))
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
        return torch.cat(values, dim=-1)


class DiscreteCGN(nn.Module):
    """
    Discrete CGN dynamics with one-step affine mapping (from NSE reference):
      u_extended = [u1, v] -> u_extended_next = [f1 + g1 v, f2 + g2 v]
    Inputs/outputs are flattened vectors with shapes [B, dim_u1 + dim_z].
    """

    def __init__(self, dim_u1: int, dim_z: int):
        super().__init__()
        self.dim_u1 = dim_u1
        self.dim_z = dim_z
        self.f1 = nn.Parameter((1 / dim_u1**0.5) * torch.rand(dim_u1, 1))
        self.g1 = nn.Parameter((1 / (dim_u1 * dim_z) ** 0.5) * torch.rand(dim_u1, dim_z))
        self.f2 = nn.Parameter((1 / dim_z**0.5) * torch.rand(dim_z, 1))
        self.g2 = nn.Parameter((1 / dim_z) * torch.rand(dim_z, dim_z))

    def forward(self, u_extended: torch.Tensor) -> torch.Tensor:
        z = u_extended[:, self.dim_u1 :]
        z = z.unsqueeze(-1)
        u1_pred = self.f1 + self.g1 @ z
        z_pred = self.f2 + self.g2 @ z
        return torch.cat([u1_pred.squeeze(-1), z_pred.squeeze(-1)], dim=-1)

    def cgn(self):
        """Return parameters (f1, g1, f2, g2) for filtering as in NSE reference."""
        return self.f1, self.g1, self.f2, self.g2


class _SubsetDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset: ERA5Dataset, indices: np.ndarray):
        self.base = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[self.indices[i]]


def build_dataloaders(args: argparse.Namespace):
    base = ERA5Dataset(
        data_path=args.data_path,
        seq_length=args.forward_step,
        min_path=args.min_path,
        max_path=args.max_path,
    )
    N = len(base)
    idx = np.arange(N)
    np.random.shuffle(idx)
    ntr = int(N * args.train_split)
    tr_idx, te_idx = idx[:ntr], idx[ntr:]

    train_loader = DataLoader(
        _SubsetDataset(base, tr_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    test_loader = DataLoader(
        _SubsetDataset(base, te_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    return base, train_loader, test_loader


def compute_sigma_hat(
    model_cgn: DiscreteCGN,
    loader: DataLoader,
    sampler: ProbeSampler,
    encoder: ERA5Encoder,
    device: torch.device,
):
    """Estimate sigma_hat via 1-step residuals (target from t+1, pred from t)."""

    dim_u1 = sampler.dim_u1
    diffs = []
    with torch.no_grad():
        for pre_seq, post_seq in loader:
            pre_seq = pre_seq.to(device)
            post_seq = post_seq.to(device)
            u1_seq = sampler.sample(pre_seq)  # [B, T, dim_u1]
            z_seq = encoder(pre_seq)  # [B, T, dim_z]
            target_u1 = sampler.sample(post_seq)  # [B, T, dim_u1]
            target_z = encoder(post_seq)
            # One-step prediction aligns pre_seq[t] -> post_seq[t]
            u_extended = torch.cat([u1_seq, z_seq], dim=-1)
            preds = model_cgn(u_extended.reshape(-1, dim_u1 + model_cgn.dim_z))
            preds = preds.view(u_extended.shape[0], u_extended.shape[1], -1)
            target = torch.cat([target_u1, target_z], dim=-1)
            diffs.append((target - preds).reshape(-1, dim_u1 + model_cgn.dim_z))
    diffs_cat = torch.cat(diffs, dim=0)
    sigma_hat = torch.sqrt(torch.mean(diffs_cat**2, dim=0))
    return sigma_hat.cpu().numpy()


def cg_filter_batch(cgn: DiscreteCGN, sigma: torch.Tensor, u1_seq: torch.Tensor):
    """Batched closed-form filter from NSE reference.

    Args:
        cgn: DiscreteCGN module providing affine parameters.
        sigma: Tensor [dim_u1 + dim_z] with observation/latent noise std.
        u1_seq: Tensor [B, T, dim_u1] of probe observations.

    Returns:
        mu_pred: [B, T, dim_z, 1] posterior mean sequence.
        R_pred: [B, T, dim_z, dim_z] posterior covariance sequence.
    """

    device = u1_seq.device
    B, T, dim_u1 = u1_seq.shape
    dim_z = cgn.dim_z
    s1 = torch.diag(sigma[:dim_u1]).to(device)
    s2 = torch.diag(sigma[dim_u1:]).to(device)
    mu_pred = torch.zeros(B, T, dim_z, 1, device=device)
    R_pred = torch.zeros(B, T, dim_z, dim_z, device=device)

    f1, g1, f2, g2 = cgn.cgn()
    f1_b = f1.unsqueeze(0).to(device)
    g1_b = g1.unsqueeze(0).to(device)
    f2_b = f2.unsqueeze(0).to(device)
    g2_b = g2.unsqueeze(0).to(device)

    mu0 = torch.zeros(B, dim_z, 1, device=device)
    R0 = 0.01 * torch.eye(dim_z, device=device).unsqueeze(0).repeat(B, 1, 1)
    eye_u1 = torch.eye(dim_u1, device=device)

    for n in range(T):
        u1n = u1_seq[:, n].unsqueeze(-1)  # [B, dim_u1, 1]
        innov = u1n - f1_b - torch.bmm(g1_b, mu0)  # [B, dim_u1, 1]
        R0g1T = torch.bmm(R0, g1_b.transpose(1, 2))
        S = s1 @ s1.T + torch.bmm(g1_b, R0g1T)
        S = S + 1e-6 * eye_u1  # stabilise inverse
        invS = torch.linalg.inv(S)
        gain_lat = torch.bmm(g2_b, R0g1T)
        mu1 = f2_b + torch.bmm(g2_b, mu0) + torch.bmm(torch.bmm(gain_lat, invS), innov)
        R1 = (
            torch.bmm(torch.bmm(g2_b, R0), g2_b.transpose(1, 2))
            + s2 @ s2.T
            - torch.bmm(torch.bmm(gain_lat, invS), gain_lat.transpose(1, 2))
        )
        mu_pred[:, n] = mu1
        R_pred[:, n] = R1
        mu0 = mu1
        R0 = R1
    return mu_pred, R_pred


def compute_da_loss(
    encoder: ERA5Encoder,
    decoder: ERA5Decoder,
    cgn: DiscreteCGN,
    sigma_hat: torch.Tensor,
    frames: torch.Tensor,
    sampler: ProbeSampler,
    da_horizon: int,
    da_warmup: int,
):
    """Compute DA loss using the closed-form filter over an assimilation horizon.

    frames: [B, T, C, H, W] normalized ERA5 frames (pre_seq from dataset).
    DA horizon truncates to the first L frames; warm-up steps are excluded from
    the MSE accumulation (used only to spin up the filter as in the reference).
    """

    B, T, _, _, _ = frames.shape
    horizon = min(T, da_horizon)
    if horizon <= 1:
        return torch.tensor(0.0, device=frames.device)
    warm = min(da_warmup, horizon)
    frames_h = frames[:, :horizon]
    u1_seq = sampler.sample(frames_h)
    mu_pred, _ = cg_filter_batch(cgn, sigma_hat, u1_seq)
    mu_flat = mu_pred.squeeze(-1)  # [B, horizon, dim_z]
    decoded = decoder(mu_flat)
    if warm >= horizon:
        return torch.tensor(0.0, device=frames.device)
    return F.mse_loss(decoded[:, warm:], frames_h[:, warm:])


def train_epoch(
    encoder: ERA5Encoder,
    decoder: ERA5Decoder,
    cgn: DiscreteCGN,
    loader: DataLoader,
    sampler: ProbeSampler,
    device: torch.device,
    forward_step: int,
    lam_ae: float,
    lam_forecast: float,
    lam_latent: float,
    optimizer: torch.optim.Optimizer,
    sigma_hat: torch.Tensor | None = None,
    lambda_DA: float = 0.0,
    da_horizon: int = 0,
    da_warmup: int = 0,
):
    encoder.train()
    decoder.train()
    cgn.train()

    total_loss = 0.0
    for pre_seq, post_seq in loader:
        pre_seq = pre_seq.to(device)
        post_seq = post_seq.to(device)

        z_seq = encoder(pre_seq)  # [B, T, dim_z]
        recon_seq = decoder(z_seq)
        loss_ae = F.mse_loss(pre_seq, recon_seq)

        u1_seq = sampler.sample(pre_seq)  # [B, T, dim_u1]
        z_target_seq = encoder(post_seq)
        u_state = torch.cat([u1_seq[:, 0], z_seq[:, 0]], dim=-1)

        preds = []
        for _ in range(forward_step):
            u_state = cgn(u_state)
            preds.append(u_state)
        pred_stack = torch.stack(preds, dim=1)  # [B, forward_step, dim_u1+dim_z]

        pred_u1 = pred_stack[..., : sampler.dim_u1]
        pred_z = pred_stack[..., sampler.dim_u1 :]

        target_u1 = sampler.sample(post_seq[:, :forward_step])
        target_z = z_target_seq[:, :forward_step]

        loss_forecast = F.mse_loss(pred_u1, target_u1)
        loss_latent = F.mse_loss(pred_z, target_z)

        loss = lam_ae * loss_ae + lam_forecast * loss_forecast + lam_latent * loss_latent

        loss_da = torch.tensor(0.0, device=device)
        if sigma_hat is not None and lambda_DA > 0.0:
            loss_da = compute_da_loss(
                encoder,
                decoder,
                cgn,
                sigma_hat,
                pre_seq,
                sampler,
                da_horizon,
                da_warmup,
            )
            loss = loss + lambda_DA * loss_da

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def save_stage(prefix: str, args: argparse.Namespace, encoder, decoder, cgn, sigma_hat, coords, config):
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(encoder.state_dict(), os.path.join(args.out_dir, f"{prefix}_encoder.pt"))
    torch.save(decoder.state_dict(), os.path.join(args.out_dir, f"{prefix}_decoder.pt"))
    torch.save(cgn.state_dict(), os.path.join(args.out_dir, f"{prefix}_cgn.pt"))
    np.save(os.path.join(args.out_dir, f"{prefix}_sigma_hat.npy"), sigma_hat)
    np.save(os.path.join(args.out_dir, f"{prefix}_probe_coords.npy"), np.array(coords, dtype=np.int32))
    with open(os.path.join(args.out_dir, f"{prefix}_config.json"), "w") as f:
        json.dump(config, f, indent=2)


def load_stage1(args: argparse.Namespace, encoder, decoder, cgn):
    enc_path = os.path.join(args.out_dir, "stage1_encoder.pt")
    dec_path = os.path.join(args.out_dir, "stage1_decoder.pt")
    cgn_path = os.path.join(args.out_dir, "stage1_cgn.pt")
    sigma_path = os.path.join(args.out_dir, "stage1_sigma_hat.npy")
    coords_path = os.path.join(args.out_dir, "stage1_probe_coords.npy")
    required = [enc_path, dec_path, cgn_path, sigma_path, coords_path]
    for p in required:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {p}; run stage1 first.")
    encoder.load_state_dict(torch.load(enc_path, map_location=args.device))
    decoder.load_state_dict(torch.load(dec_path, map_location=args.device))
    cgn.load_state_dict(torch.load(cgn_path, map_location=args.device))
    sigma_hat = np.load(sigma_path)
    coords = np.load(coords_path).tolist()
    return sigma_hat, coords


def train_stage1(args: argparse.Namespace):
    base_dataset, train_loader, test_loader = build_dataloaders(args)
    H, W, C = base_dataset.H, base_dataset.W, base_dataset.C

    coords = make_probe_coords_from_ratio(
        H=H,
        W=W,
        ratio=args.obs_ratio,
        layout=args.obs_layout,
        seed=args.seed,
        min_spacing=args.min_spacing,
    )
    sampler = ProbeSampler(coords, args.use_channels)

    dim_z = ERA5_settings["state_feature_dim"][-1]
    encoder = ERA5Encoder(dim_z=dim_z).to(args.device)
    decoder = ERA5Decoder(dim_z=dim_z).to(args.device)
    cgn = DiscreteCGN(dim_u1=sampler.dim_u1, dim_z=dim_z).to(args.device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + list(cgn.parameters()), lr=args.lr
    )

    best_loss = float("inf")
    best_sigma = None
    for ep in range(1, args.stage1_epochs + 1):
        start = time.time()
        loss = train_epoch(
            encoder,
            decoder,
            cgn,
            train_loader,
            sampler,
            args.device,
            forward_step=args.forward_step,
            lam_ae=args.lam_ae,
            lam_forecast=args.lam_forecast,
            lam_latent=args.lam_latent,
            optimizer=optimizer,
        )
        duration = time.time() - start
        print(f"[Stage1] Epoch {ep}/{args.stage1_epochs} - loss {loss:.4f} - time {duration:.2f}s")
        if loss < best_loss:
            best_loss = loss
            best_sigma = compute_sigma_hat(cgn, train_loader, sampler, encoder, args.device)
            config = {
                "dim_z": dim_z,
                "dim_u1": sampler.dim_u1,
                "use_channels": args.use_channels,
                "obs_ratio": args.obs_ratio,
                "forward_step": args.forward_step,
                "model_name": args.model_name,
                "lam_ae": args.lam_ae,
                "lam_forecast": args.lam_forecast,
                "lam_latent": args.lam_latent,
                "seed": args.seed,
            }
            save_stage("stage1", args, encoder, decoder, cgn, best_sigma, coords, config)
            print(f"[Stage1] Best checkpoint updated at epoch {ep} with loss {best_loss:.4f}")

    print(f"[Stage1] Training complete. Best loss: {best_loss:.4f}")


def train_stage2(args: argparse.Namespace):
    base_dataset, train_loader, test_loader = build_dataloaders(args)
    H, W, C = base_dataset.H, base_dataset.W, base_dataset.C

    dim_z = ERA5_settings["state_feature_dim"][-1]
    coords_path = os.path.join(args.out_dir, "stage1_probe_coords.npy")
    if not os.path.exists(coords_path):
        raise FileNotFoundError("Stage1 probe coordinates not found; run stage1 first.")
    coords = np.load(coords_path).tolist()
    dim_u1 = len(coords) * len(args.use_channels)

    encoder = ERA5Encoder(dim_z=dim_z).to(args.device)
    decoder = ERA5Decoder(dim_z=dim_z).to(args.device)
    cgn = DiscreteCGN(dim_u1=dim_u1, dim_z=dim_z).to(args.device)

    sigma_hat_np, _ = load_stage1(args, encoder, decoder, cgn)
    sampler = ProbeSampler(coords, args.use_channels)

    sigma_hat = torch.from_numpy(sigma_hat_np).float().to(args.device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + list(cgn.parameters()), lr=args.lr
    )

    best_loss = float("inf")
    for ep in range(1, args.stage2_epochs + 1):
        start = time.time()
        loss = train_epoch(
            encoder,
            decoder,
            cgn,
            train_loader,
            sampler,
            args.device,
            forward_step=args.forward_step,
            lam_ae=args.lam_ae,
            lam_forecast=args.lam_forecast,
            lam_latent=args.lam_latent,
            optimizer=optimizer,
            sigma_hat=sigma_hat,
            lambda_DA=args.lambda_DA,
            da_horizon=args.da_horizon,
            da_warmup=args.da_warmup,
        )
        duration = time.time() - start
        print(f"[Stage2] Epoch {ep}/{args.stage2_epochs} - loss {loss:.4f} - time {duration:.2f}s")
        if loss < best_loss:
            best_loss = loss
            # Stage2 keeps sigma_hat fixed (copied from stage1); adjust here if future updates are needed.
            config = {
                "dim_z": dim_z,
                "dim_u1": sampler.dim_u1,
                "use_channels": args.use_channels,
                "obs_ratio": args.obs_ratio,
                "forward_step": args.forward_step,
                "model_name": args.model_name,
                "lam_ae": args.lam_ae,
                "lam_forecast": args.lam_forecast,
                "lam_latent": args.lam_latent,
                "lambda_DA": args.lambda_DA,
                "da_horizon": args.da_horizon,
                "da_warmup": args.da_warmup,
                "seed": args.seed,
            }
            save_stage("stage2", args, encoder, decoder, cgn, sigma_hat_np, coords, config)
            print(f"[Stage2] Best checkpoint updated at epoch {ep} with loss {best_loss:.4f}")

    print(f"[Stage2] Training complete. Best loss: {best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Two-stage training for discrete CGKN on ERA5")
    parser.add_argument("--data_path", type=str, default="../../../../data/ERA5/ERA5_data/train_seq_state.h5")
    parser.add_argument("--min_path", type=str, default="../../../../data/ERA5/ERA5_data/min_val.npy")
    parser.add_argument("--max_path", type=str, default="../../../../data/ERA5/ERA5_data/max_val.npy")
    parser.add_argument("--model_name", type=str, default="discreteCGKN")
    parser.add_argument("--out_dir", type=str, default="../../../../results/discreteCGKN/ERA5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--stage1_epochs", type=int, default=100)
    parser.add_argument("--stage2_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--forward_step", type=int, default=12)
    parser.add_argument("--obs_ratio", type=float, default=0.15)
    parser.add_argument("--obs_layout", type=str, default="random")
    parser.add_argument("--min_spacing", type=int, default=4)
    parser.add_argument("--use_channels", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lam_ae", type=float, default=1.0)
    parser.add_argument("--lam_forecast", type=float, default=1.0)
    parser.add_argument("--lam_latent", type=float, default=1.0)
    parser.add_argument("--lambda_DA", type=float, default=1.0)
    parser.add_argument("--da_horizon", type=int, default=50)
    parser.add_argument("--da_warmup", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_stage2", action="store_true", default=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    args.device = torch.device(args.device)

    if args.train_stage2:
        train_stage2(args)
    else:
        train_stage1(args)


if __name__ == "__main__":
    main()