from torch.nn.modules import Module
from base import *
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat

import os
import sys
current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", ".."))
sys.path.append(src_directory)

from src.utils.transformer import TransformerBlock, OverlapPatchEmbed, Downsample_Flex, LayerNorm

# State dimension = 1 channel (vorticity), 64x64 resolution for Kolmogorov flow
KOL_settings = {"obs_dim": [1, 64, 64], 
                "state_dim": [1, 64, 64], 
                "seq_length": 10,
                "obs_feature_dim": [512, 128, 64, 32, 16, 8], 
                "state_filter_feature_dim": [16, 32, 64, 128, 256],
                "state_feature_dim": [128 * 16 * 16, 512],
                }

'''
================================
Simplified Factorized Attention Components
================================
'''

class RoPE2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, pos_x, pos_y):
        B, H, W, D = x.shape
        freqs_x = torch.exp(torch.arange(0, D//4, dtype=torch.float32, device=x.device) * -(np.log(10000.0) / (D//4)))
        freqs_y = torch.exp(torch.arange(0, D//4, dtype=torch.float32, device=x.device) * -(np.log(10000.0) / (D//4)))

        pos_x = pos_x.unsqueeze(-1) * freqs_x
        pos_y = pos_y.unsqueeze(-1) * freqs_y

        cos_x, sin_x = torch.cos(pos_x), torch.sin(pos_x)
        cos_y, sin_y = torch.cos(pos_y), torch.sin(pos_y)

        x_rot = x.clone()

        x1, x2 = x[..., :D//4], x[..., D//4:D//2]
        cos_x = cos_x.unsqueeze(0).unsqueeze(2).expand(B, -1, W, -1)
        sin_x = sin_x.unsqueeze(0).unsqueeze(2).expand(B, -1, W, -1)
        x_rot[..., :D//4] = x1 * cos_x - x2 * sin_x
        x_rot[..., D//4:D//2] = x1 * sin_x + x2 * cos_x

        x3, x4 = x[..., D//2:3*D//4], x[..., 3*D//4:]
        cos_y = cos_y.unsqueeze(0).unsqueeze(1).expand(B, H, -1, -1)
        sin_y = sin_y.unsqueeze(0).unsqueeze(1).expand(B, H, -1, -1)
        x_rot[..., D//2:3*D//4] = x3 * cos_y - x4 * sin_y
        x_rot[..., 3*D//4:] = x3 * sin_y + x4 * cos_y

        return x_rot


class FactorizedAttention2D(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.rope = RoPE2D(dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, pos_x, pos_y):
        B, H, W, D = x.shape
        x_norm = self.norm(x)
        x_rope = self.rope(x_norm, pos_x, pos_y)

        qkv = self.to_qkv(x_rope).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b h w (head d) -> b head h w d', head=self.heads), qkv)

        q_x = rearrange(q, 'b head h w d -> (b head w) h d')
        k_x = rearrange(k, 'b head h w d -> (b head w) h d')
        v_x = rearrange(v, 'b head h w d -> (b head w) h d')
        attn_x = torch.softmax(torch.bmm(q_x, k_x.transpose(-2, -1)) / (self.head_dim ** 0.5), dim=-1)
        out_x = torch.bmm(attn_x, v_x)
        out_x = rearrange(out_x, '(b head w) h d -> b head h w d', b=B, head=self.heads, w=W)

        q_y = rearrange(out_x, 'b head h w d -> (b head h) w d')
        k_y = rearrange(out_x, 'b head h w d -> (b head h) w d')
        v_y = rearrange(out_x, 'b head h w d -> (b head h) w d')
        attn_y = torch.softmax(torch.bmm(q_y, k_y.transpose(-2, -1)) / (self.head_dim ** 0.5), dim=-1)
        out_y = torch.bmm(attn_y, v_y)
        out_y = rearrange(out_y, '(b head h) w d -> b head h w d', b=B, head=self.heads, h=H)

        out = rearrange(out_y, 'b head h w d -> b h w (head d)')
        return self.to_out(out)


class FactorizedBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=2):
        super().__init__()
        self.attn = FactorizedAttention2D(dim, heads)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x, pos_x, pos_y):
        x = x + self.attn(x, pos_x, pos_y)
        x = x + self.mlp(x)
        return x


class KOL_K_S(Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim, self.w, self.h = KOL_settings["state_dim"]
        self.hidden_dims = KOL_settings["state_feature_dim"]

        self.embed_dim = 128
        self.num_layers = 1
        self.num_heads = 4

        self.input_proj = nn.Conv2d(1, self.embed_dim, kernel_size=7, stride=1, padding=3)
        self.downsample = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=4, stride=4)

        self.factorized_layers = nn.ModuleList([
            FactorizedBlock(self.embed_dim, self.num_heads)
            for _ in range(self.num_layers)
        ])

        self.flatten = nn.Flatten()
        self.final_proj = nn.Linear(self.hidden_dims[0], self.hidden_dims[1])
        self.dropout = nn.Dropout(0.3)

    def forward(self, state):
        B, C, H, W = state.shape

        x = self.input_proj(state)  # [B, embed_dim, 64, 64]
        x = self.downsample(x)      # [B, embed_dim, 16, 16]

        x = x.permute(0, 2, 3, 1)  # [B, 16, 16, embed_dim]

        pos_x = torch.linspace(0, 2 * np.pi, x.shape[1], device=state.device)
        pos_y = torch.linspace(0, 2 * np.pi, x.shape[2], device=state.device)

        for layer in self.factorized_layers:
            x = layer(x, pos_x, pos_y)

        x = x.permute(0, 3, 1, 2)  # [B, embed_dim, 16, 16]
        x = self.flatten(x)
        x = self.dropout(x)
        z = self.final_proj(x)

        return z


class KOL_K_S_preimage(nn.Module):
    """Simplified Kolmogorov Flow decoder"""
    def __init__(self, *args, **kwargs) -> None:
        super(KOL_K_S_preimage, self).__init__(*args, **kwargs)
        self.input_dim, self.w, self.h = KOL_settings["state_dim"]
        self.hidden_dims = KOL_settings["state_feature_dim"]
        
        # Simple decoder with transpose convolutions
        self.linear = nn.Linear(self.hidden_dims[1], 128 * 16 * 16)  # Updated to match new dimensions
        
        self.decoder = nn.Sequential(
            # Start from [B, 128, 16, 16]
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # -> [B, 64, 32, 32]
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # -> [B, 32, 64, 64]
            nn.GELU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),                       # -> [B, 16, 64, 64]
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1)                                    # -> [B, 1, 64, 64]
        )
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, z: torch.Tensor):
        # Linear transformation and reshape
        x = self.linear(z)
        x = self.dropout(x)
        x = x.view(-1, 128, 16, 16)  # Updated to match new dimensions
        
        # Decode through transpose convolutions
        recon_s = self.decoder(x)
        
        return recon_s


'''
=============================
Operators for Kolmogorov Flow system
=============================
'''


class KOL_C_FORWARD(base_forward_model):
    def __init__(self, *args, **kwargs) -> None:
        K_S = KOL_K_S()
        K_S_preimage = KOL_K_S_preimage()
        seq_length = KOL_settings["seq_length"]
        super(KOL_C_FORWARD, self).__init__(K_S=K_S,
                                           K_S_preimage=K_S_preimage, 
                                           seq_length=seq_length,
                                           *args, **kwargs)