import torch
import torch.nn as nn

from src.models.CAE_Koopman.ERA5_High.era5_high_model import (
    ERA5_K_S,
    ERA5_K_S_preimage,
    ERA5_settings,
)


class ERA5Encoder(nn.Module):
    """
    Wrap ERA5_K_S to produce latent features per time step.
    Input:  [B, T, C, H, W]
    Output: [B, T, dim_z]
    """

    def __init__(self, dim_z: int):
        super().__init__()
        self.encoder = ERA5_K_S()
        expected_dim = ERA5_settings["state_feature_dim"][-1]
        assert (
            dim_z == expected_dim
        ), f"ERA5Encoder latent dim should be {expected_dim}, got {dim_z}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        z_list = []
        for t in range(T):
            frame = x[:, t]  # [B, C, H, W]
            z = self.encoder(frame)  # [B, dim_z]
            z_list.append(z.unsqueeze(1))
        return torch.cat(z_list, dim=1)  # [B, T, dim_z]


class ERA5Decoder(nn.Module):
    """
    Wrap ERA5_K_S_preimage to reconstruct frames from latent z.
    Input:  [B, T, dim_z]
    Output: [B, T, C, H, W]
    """

    def __init__(self, dim_z: int):
        super().__init__()
        self.decoder = ERA5_K_S_preimage()
        expected_dim = ERA5_settings["state_feature_dim"][-1]
        assert (
            dim_z == expected_dim
        ), f"ERA5Decoder latent dim should be {expected_dim}, got {dim_z}"

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        B, T, _ = z_seq.shape
        frames = []
        for t in range(T):
            z = z_seq[:, t]  # [B, dim_z]
            frame = self.decoder(z)  # [B, C, H, W]
            frames.append(frame.unsqueeze(1))
        return torch.cat(frames, dim=1)  # [B, T, C, H, W]
