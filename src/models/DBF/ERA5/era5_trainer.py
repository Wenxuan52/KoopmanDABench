import argparse
import copy
import json
import math
import os
import random
import sys
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
sys.path.append(src_directory)

from src.utils.Dataset import ERA5Dataset
from src.models.DBF.ERA5.era5_model import ERA5Decoder


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ObservationMask:
    def __init__(self, probe_mask: torch.Tensor) -> None:
        if probe_mask.dtype != torch.bool:
            probe_mask = probe_mask.bool()
        self.mask = probe_mask
        self.channel_indices = []
        flat_size = probe_mask.shape[-2] * probe_mask.shape[-1]
        for c in range(probe_mask.shape[0]):
            idx = torch.nonzero(probe_mask[c].view(flat_size), as_tuple=False).squeeze(-1)
            self.channel_indices.append(idx)
        self.obs_dim = sum(len(idx) for idx in self.channel_indices)

    def to(self, device: torch.device) -> "ObservationMask":
        self.mask = self.mask.to(device)
        self.channel_indices = [idx.to(device) for idx in self.channel_indices]
        return self

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        added_time_dim = False
        if x.ndim == 4:
            x = x.unsqueeze(1)
            added_time_dim = True
        if x.ndim != 5:
            raise ValueError("Input must have shape [B,T,C,H,W] or [B,C,H,W]")
        b, t, c, h, w = x.shape
        assert c == self.mask.shape[0], "Channel mismatch with mask"
        x_flat = x.reshape(b, t, c, h * w)
        obs_parts = []
        for ch in range(c):
            gather_idx = self.channel_indices[ch].view(1, 1, -1).expand(b, t, -1)
            sampled = torch.gather(x_flat[:, :, ch, :], dim=2, index=gather_idx)
            obs_parts.append(sampled)
        obs = torch.cat(obs_parts, dim=-1)
        if added_time_dim:
            obs = obs.squeeze(1)
        return obs

    def expand_mask(self) -> torch.Tensor:
        return self.mask.float()


class ERA5IOONetwork(nn.Module):
    def __init__(self, obs_dim: int, latent_dim: int, hidden_dims) -> None:
        super().__init__()
        output_dim = latent_dim * 2
        dims = [obs_dim] + list(hidden_dims) + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(0.1))
        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor):
        original_shape = obs.shape
        if obs.ndim == 3:
            obs = obs.reshape(-1, obs.shape[-1])
            out = self.network(obs)
            out = out.view(*original_shape[:-1], -1)
        elif obs.ndim == 2:
            out = self.network(obs)
        else:
            raise ValueError("Observation tensor must be 2D or 3D")
        mu, logvar = out.chunk(2, dim=-1)
        sigma2 = F.softplus(logvar) + 1e-4
        return mu, sigma2


class SpectralKoopmanOperator(nn.Module):
    def __init__(self, latent_dim: int, rho_clip: float, process_noise_init: float) -> None:
        super().__init__()
        if latent_dim % 2 != 0:
            raise ValueError("Latent dimension must be even for spectral Koopman blocks")
        self.latent_dim = latent_dim
        self.num_pairs = latent_dim // 2
        self.rho = nn.Parameter(torch.zeros(self.num_pairs))
        self.omega = nn.Parameter(torch.zeros(self.num_pairs))
        self.log_process_noise = nn.Parameter(torch.full((self.num_pairs, 2), process_noise_init))
        self.rho_clip = float(rho_clip)

    def koopman_blocks(self) -> torch.Tensor:
        rho = torch.clamp(self.rho, min=-self.rho_clip, max=self.rho_clip)
        magnitude = torch.exp(rho)
        cos_term = torch.cos(self.omega)
        sin_term = torch.sin(self.omega)
        blocks = torch.zeros(self.num_pairs, 2, 2, device=self.rho.device)
        blocks[:, 0, 0] = magnitude * cos_term
        blocks[:, 0, 1] = -magnitude * sin_term
        blocks[:, 1, 0] = magnitude * sin_term
        blocks[:, 1, 1] = magnitude * cos_term
        return blocks

    def process_noise_covariance(self) -> torch.Tensor:
        diag = F.softplus(self.log_process_noise) + 1e-4
        return torch.diag_embed(diag)

    def predict(self, mean_pairs: torch.Tensor, cov_pairs: torch.Tensor):
        blocks = self.koopman_blocks().unsqueeze(0)
        noise = self.process_noise_covariance().unsqueeze(0)
        blocks_t = blocks.transpose(-1, -2)
        mean_pred = torch.matmul(blocks, mean_pairs.unsqueeze(-1)).squeeze(-1)
        cov_pred = torch.matmul(blocks, torch.matmul(cov_pairs, blocks_t)) + noise
        cov_pred = 0.5 * (cov_pred + cov_pred.transpose(-1, -2))
        return mean_pred, cov_pred


DEFAULT_CONFIG: Dict[str, object] = {
    "seed": 42,
    "seq_length": 64,
    "mask_rate": 0.05,
    "mask_seed": 1024,
    "latent_dim": 512,
    "train_data": "../../../../data/ERA5/ERA5_data/train_seq_state.h5",
    "val_data": "../../../../data/ERA5/ERA5_data/val_seq_state.h5",
    "min_data": "../../../../data/ERA5/ERA5_data/min_val.npy",
    "max_data": "../../../../data/ERA5/ERA5_data/max_val.npy",
    "save_folder": "../../../../results/DBF/ERA5",
    "batch_size": 64,
    "num_epochs": 100,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "grad_clip": 1.0,
    "patience": 30,
    "num_workers": 0,
    "pin_memory": True,
    "ioo_hidden_dims": [1024, 1024],
    "scheduler": {"step_size": 100, "gamma": 0.5},
    "kl_beta": 1.0,
    "recon_sigma2": 1e-2,
    "obs_noise_std": 0.05,
    "train_decoder": True,
    "init_cov": 1.0,
    "rho_clip": 0.2,
    "process_noise_init": -2.0,
    "cov_epsilon": 1e-6,
}


def deep_update(base: Dict, updates: Dict) -> Dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Optional[str]) -> Dict[str, object]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if not path:
        return config
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    with open(path, "r") as handle:
        if ext == ".json":
            user_cfg = json.load(handle)
        elif ext in {".yml", ".yaml"}:
            import yaml

            user_cfg = yaml.safe_load(handle)
        else:
            raise ValueError(f"Unsupported config extension: {ext}")
    return deep_update(config, user_cfg or {})


def create_probe_mask(channels: int, height: int, width: int, rate: float, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    total_points = height * width
    num_points = max(1, int(total_points * rate))
    mask = torch.zeros(channels, height, width, dtype=torch.bool)
    for c in range(channels):
        perm = torch.randperm(total_points, generator=generator)
        selected = perm[:num_points]
        mask[c].view(-1)[selected] = True
    return mask


class ERA5DBFTrainer:
    def __init__(self, config: Dict[str, object]) -> None:
        self.config = config
        set_seed(int(config["seed"]))
        torch.set_default_dtype(torch.float32)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

        os.makedirs(config["save_folder"], exist_ok=True)

        self.train_dataset = ERA5Dataset(
            data_path=config["train_data"],
            seq_length=config["seq_length"],
            min_path=config["min_data"],
            max_path=config["max_data"],
        )
        self.val_dataset = ERA5Dataset(
            data_path=config["val_data"],
            seq_length=config["seq_length"],
            min_path=config["min_data"],
            max_path=config["max_data"],
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
        )

        mask_tensor = create_probe_mask(
            channels=self.train_dataset.C,
            height=self.train_dataset.H,
            width=self.train_dataset.W,
            rate=config["mask_rate"],
            seed=config["mask_seed"],
        )
        self.observation_mask = ObservationMask(mask_tensor).to(self.device)
        print(f"[INFO] Observation dimension: {self.observation_mask.obs_dim}")

        latent_dim = int(config["latent_dim"])
        self.decoder = ERA5Decoder(dim_z=latent_dim).to(self.device)
        self.train_decoder = bool(config.get("train_decoder", True))
        if not self.train_decoder:
            self.decoder.eval()
            for param in self.decoder.parameters():
                param.requires_grad = False

        self.ioo = ERA5IOONetwork(
            obs_dim=self.observation_mask.obs_dim,
            latent_dim=latent_dim,
            hidden_dims=config["ioo_hidden_dims"],
        ).to(self.device)
        self.koopman = SpectralKoopmanOperator(
            latent_dim=latent_dim,
            rho_clip=float(config["rho_clip"]),
            process_noise_init=float(config["process_noise_init"]),
        ).to(self.device)

        params = []
        if self.train_decoder:
            params += list(self.decoder.parameters())
        params += list(self.ioo.parameters())
        params += list(self.koopman.parameters())

        self.optimizer = Adam(
            params,
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )

        scheduler_cfg = config.get("scheduler", {})
        step_size = scheduler_cfg.get("step_size", 0)
        gamma = scheduler_cfg.get("gamma", 1.0)
        self.scheduler = (
            StepLR(self.optimizer, step_size=step_size, gamma=gamma)
            if step_size and gamma != 1.0
            else None
        )

        self.grad_clip = config.get("grad_clip")
        self.kl_beta = float(config.get("kl_beta", 1.0))
        self.recon_sigma2 = float(config.get("recon_sigma2", 1e-2))
        self.obs_noise_std = float(config.get("obs_noise_std", 0.0))
        self.init_cov = float(config.get("init_cov", 1.0))
        self.cov_epsilon = float(config.get("cov_epsilon", 1e-6))

        self.best_val_loss = float("inf")
        self.epochs_without_improve = 0
        self.log = []

    def _initial_latent_state(self, batch_size: int, device: torch.device):
        num_pairs = self.koopman.num_pairs
        mean = torch.zeros(batch_size, num_pairs, 2, device=device)
        base_cov = torch.eye(2, device=device).unsqueeze(0).unsqueeze(0)
        cov = self.init_cov * base_cov.expand(batch_size, num_pairs, 2, 2).clone()
        return mean, cov

    def _gaussian_update(self, mu_prior, cov_prior, mu_obs, cov_obs):
        eye = torch.eye(2, device=mu_prior.device).view(1, 1, 2, 2)
        cov_prior = cov_prior + self.cov_epsilon * eye
        cov_obs = cov_obs + self.cov_epsilon * eye
        cov_prior_inv = torch.linalg.inv(cov_prior)
        cov_obs_inv = torch.linalg.inv(cov_obs)
        cov_post_inv = cov_prior_inv + cov_obs_inv
        cov_post = torch.linalg.inv(cov_post_inv)
        fused_mean = torch.matmul(cov_prior_inv, mu_prior.unsqueeze(-1)) + torch.matmul(cov_obs_inv, mu_obs.unsqueeze(-1))
        mu_post = torch.matmul(cov_post, fused_mean).squeeze(-1)
        cov_post = 0.5 * (cov_post + cov_post.transpose(-1, -2))
        return mu_post, cov_post

    def _gaussian_kl(self, mu_post, cov_post, mu_prior, cov_prior):
        eye = torch.eye(2, device=mu_post.device).view(1, 1, 2, 2)
        cov_prior = cov_prior + self.cov_epsilon * eye
        cov_post = cov_post + self.cov_epsilon * eye
        cov_prior_inv = torch.linalg.inv(cov_prior)
        diff = (mu_post - mu_prior).unsqueeze(-1)
        trace_term = torch.einsum("bpij,bpji->bp", cov_prior_inv, cov_post)
        quad_term = torch.matmul(
            diff.transpose(-2, -1), torch.matmul(cov_prior_inv, diff)
        ).squeeze(-1).squeeze(-1)
        sign_prior, logdet_prior = torch.linalg.slogdet(cov_prior)
        sign_post, logdet_post = torch.linalg.slogdet(cov_post)
        if torch.any(sign_prior <= 0) or torch.any(sign_post <= 0):
            raise RuntimeError("Non-positive definite covariance encountered during KL computation")
        kl = 0.5 * (trace_term + quad_term - 2 + (logdet_prior - logdet_post))
        kl = kl.sum(dim=-1)
        return kl

    def _decode(self, latent_pairs: torch.Tensor) -> torch.Tensor:
        latent_flat = latent_pairs.reshape(latent_pairs.shape[0], -1)
        return self.decoder.decoder(latent_flat)

    def _process_batch(self, seq_batch: torch.Tensor, training: bool):
        seq_batch = seq_batch.to(self.device)
        obs_all = self.observation_mask.sample(seq_batch)
        if training and self.obs_noise_std > 0:
            obs_all = obs_all + torch.randn_like(obs_all) * self.obs_noise_std

        batch_size, time_steps = seq_batch.shape[0], seq_batch.shape[1]
        mu_prev, cov_prev = self._initial_latent_state(batch_size, self.device)

        recon_losses = []
        kl_losses = []
        decoded_frames = []
        for t in range(time_steps):
            obs_t = obs_all[:, t, :]
            mu_obs_flat, sigma2_obs_flat = self.ioo(obs_t)
            mu_obs_pairs = mu_obs_flat.view(batch_size, self.koopman.num_pairs, 2)
            cov_obs = torch.diag_embed(sigma2_obs_flat.view(batch_size, self.koopman.num_pairs, 2))

            mu_prior, cov_prior = self.koopman.predict(mu_prev, cov_prev)
            mu_post, cov_post = self._gaussian_update(mu_prior, cov_prior, mu_obs_pairs, cov_obs)

            recon = self._decode(mu_post)
            target = seq_batch[:, t, :, :, :]
            n_elements = target[0].numel()
            const_term = 0.5 * math.log(2 * math.pi * self.recon_sigma2) * n_elements
            mse_term = ((target - recon) ** 2).reshape(batch_size, -1).sum(dim=-1) / (2 * self.recon_sigma2)
            recon_step = mse_term + const_term
            kl_step = self._gaussian_kl(mu_post, cov_post, mu_prior, cov_prior)

            recon_losses.append(recon_step)
            kl_losses.append(kl_step)
            decoded_frames.append(recon.unsqueeze(1))

            mu_prev = mu_post
            cov_prev = cov_post

        recon_tensor = torch.stack(recon_losses, dim=1)
        kl_tensor = torch.stack(kl_losses, dim=1)
        recon_loss = recon_tensor.mean()
        kl_loss = kl_tensor.mean()
        total_loss = recon_loss + self.kl_beta * kl_loss

        decoded_seq = torch.cat(decoded_frames, dim=1)
        return {
            "total": total_loss,
            "recon": recon_loss,
            "kl": kl_loss,
            "decoded": decoded_seq.detach(),
        }

    def train_epoch(self, epoch: int):
        if self.train_decoder:
            self.decoder.train()
        else:
            self.decoder.eval()
        self.ioo.train()
        self.koopman.train()

        metrics = {"total": 0.0, "recon": 0.0, "kl": 0.0}
        num_batches = 0

        for seq_batch, _ in tqdm(self.train_loader, desc=f"Train {epoch:03d}", leave=False):
            outputs = self._process_batch(seq_batch, training=True)
            loss = outputs["total"]
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip:
                params_to_clip = []
                for group in self.optimizer.param_groups:
                    params_to_clip.extend(group["params"])
                torch.nn.utils.clip_grad_norm_(params_to_clip, self.grad_clip)
            self.optimizer.step()

            metrics["total"] += outputs["total"].item()
            metrics["recon"] += outputs["recon"].item()
            metrics["kl"] += outputs["kl"].item()
            num_batches += 1

        for key in metrics:
            metrics[key] /= max(1, num_batches)
        return metrics

    @torch.no_grad()
    def validate_epoch(self, epoch: int):
        self.decoder.eval()
        self.ioo.eval()
        self.koopman.eval()

        metrics = {"total": 0.0, "recon": 0.0, "kl": 0.0}
        num_batches = 0

        for seq_batch, _ in tqdm(self.val_loader, desc=f"Val {epoch:03d}", leave=False):
            outputs = self._process_batch(seq_batch, training=False)
            metrics["total"] += outputs["total"].item()
            metrics["recon"] += outputs["recon"].item()
            metrics["kl"] += outputs["kl"].item()
            num_batches += 1

        for key in metrics:
            metrics[key] /= max(1, num_batches)
        return metrics

    def save_checkpoint(self, epoch: int, name: str) -> None:
        checkpoint = {
            "epoch": epoch,
            "config": self.config,
            "decoder": self.decoder.state_dict(),
            "ioo": self.ioo.state_dict(),
            "koopman": self.koopman.state_dict(),
            "mask": self.observation_mask.expand_mask().cpu(),
            "train_min": self.train_dataset.min,
            "train_max": self.train_dataset.max,
        }
        path = os.path.join(self.config["save_folder"], f"{name}.pt")
        torch.save(checkpoint, path)
        print(f"[INFO] Saved checkpoint: {path}")

    def run(self) -> None:
        total_epochs = self.config["num_epochs"]
        patience = self.config.get("patience")
        for epoch in range(1, total_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate_epoch(epoch)

            if self.scheduler:
                self.scheduler.step()

            log_entry = {
                "epoch": epoch,
                "train_total": train_metrics["total"],
                "train_recon": train_metrics["recon"],
                "train_kl": train_metrics["kl"],
                "val_total": val_metrics["total"],
                "val_recon": val_metrics["recon"],
                "val_kl": val_metrics["kl"],
            }
            self.log.append(log_entry)

            print(
                f"[Epoch {epoch:03d}] "
                f"train_total: {train_metrics['total']:.4f} "
                f"train_recon: {train_metrics['recon']:.4f} "
                f"train_kl: {train_metrics['kl']:.4f}"
            )
            print(
                " " * 9
                + f"val_total: {val_metrics['total']:.4f} "
                + f"val_recon: {val_metrics['recon']:.4f} "
                + f"val_kl: {val_metrics['kl']:.4f}"
            )

            val_total = val_metrics["total"]
            if val_total < self.best_val_loss:
                self.best_val_loss = val_total
                self.epochs_without_improve = 0
                self.save_checkpoint(epoch, "best_model")
            else:
                self.epochs_without_improve += 1
                if patience and self.epochs_without_improve >= patience:
                    print("[INFO] Early stopping triggered")
                    break

        self.save_checkpoint(epoch, "last_model")
        log_path = os.path.join(self.config["save_folder"], "training_log.json")
        with open(log_path, "w") as log_file:
            json.dump(self.log, log_file, indent=2)
        print(f"[INFO] Training log written to {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ERA5 DBF Trainer")
    parser.add_argument("--config", type=str, default=None, help="Path to config file (JSON or YAML)")
    args = parser.parse_args()

    config = load_config(args.config)
    trainer = ERA5DBFTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
