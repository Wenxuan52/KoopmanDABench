"""Train discrete CGKN for ERA5 using discrete one-step mapping.

This script mirrors the continuous ERA5 training interface but swaps the
continuous ODE integrator for the discrete CGN mapping from
``NSE(Noisy)_CGKN.py``. The discrete step is
    [u1, v] -> [f1 + g1 v, f2 + g2 v]
with flattened vectors (u1: sparse probes; v: latent from the encoder).
Ground truth tensors use shape [B, T, C, H, W].
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
    dim_u1 = sampler.dim_u1
    dim_z = model_cgn.dim_z
    diffs = []
    with torch.no_grad():
        for pre_seq, post_seq in loader:
            pre_seq = pre_seq.to(device)
            post_seq = post_seq.to(device)
            u1_seq = sampler.sample(pre_seq)
            z_seq = encoder(pre_seq)
            u0 = torch.cat([u1_seq[:, 0], z_seq[:, 0]], dim=-1)
            preds = []
            u_prev = u0
            for _ in range(pre_seq.shape[1]):
                u_next = model_cgn(u_prev)
                preds.append(u_next)
                u_prev = u_next
            preds = torch.stack(preds, dim=1)  # [B, T, dim_u1+dim_z]
            target_u1 = sampler.sample(post_seq)
            target_z = encoder(post_seq)
            target = torch.cat([target_u1, target_z], dim=-1)
            diffs.append((target - preds).reshape(-1, dim_u1 + dim_z))
    diffs_cat = torch.cat(diffs, dim=0)
    sigma_hat = torch.sqrt(torch.mean(diffs_cat**2, dim=0))
    return sigma_hat.cpu().numpy()


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
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser(description="Train discrete CGKN on ERA5 (discrete one-step mapping)")
    parser.add_argument("--data_path", type=str, default="../../../../data/ERA5/ERA5_data/train_seq_state.h5")
    parser.add_argument("--min_path", type=str, default="../../../../data/ERA5/ERA5_data/min_val.npy")
    parser.add_argument("--max_path", type=str, default="../../../../data/ERA5/ERA5_data/max_val.npy")
    parser.add_argument("--model_name", type=str, default="discreteCGKN")
    parser.add_argument("--out_dir", type=str, default="../../../../results/discreteCGKN/ERA5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=20)
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
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device(args.device)

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
    encoder = ERA5Encoder(dim_z=dim_z).to(device)
    decoder = ERA5Decoder(dim_z=dim_z).to(device)
    cgn = DiscreteCGN(dim_u1=sampler.dim_u1, dim_z=dim_z).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + list(cgn.parameters()), lr=args.lr
    )

    for ep in range(1, args.epochs + 1):
        start = time.time()
        loss = train_epoch(
            encoder,
            decoder,
            cgn,
            train_loader,
            sampler,
            device,
            forward_step=args.forward_step,
            lam_ae=args.lam_ae,
            lam_forecast=args.lam_forecast,
            lam_latent=args.lam_latent,
            optimizer=optimizer,
        )
        duration = time.time() - start
        print(f"Epoch {ep}/{args.epochs} - loss {loss:.4f} - time {duration:.2f}s")

    # Estimate sigma_hat on the training loader for DA usage
    sigma_hat = compute_sigma_hat(cgn, train_loader, sampler, encoder, device)

    torch.save(encoder.state_dict(), os.path.join(args.out_dir, "stage2_encoder.pt"))
    torch.save(decoder.state_dict(), os.path.join(args.out_dir, "stage2_decoder.pt"))
    torch.save(cgn.state_dict(), os.path.join(args.out_dir, "stage2_cgn.pt"))
    np.save(os.path.join(args.out_dir, "sigma_hat.npy"), sigma_hat)
    np.save(os.path.join(args.out_dir, "probe_coords.npy"), np.array(coords, dtype=np.int32))

    config = {
        "dim_z": dim_z,
        "dim_u1": sampler.dim_u1,
        "use_channels": args.use_channels,
        "obs_ratio": args.obs_ratio,
        "forward_step": args.forward_step,
        "model_name": args.model_name,
    }
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
