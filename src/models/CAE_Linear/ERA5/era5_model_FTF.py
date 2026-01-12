from torch.nn.modules import Module

from src.models.CAE_Linear.base import *
import numpy as np
from einops import rearrange

# State dimension = 5 channels, 64x32 resolution
ERA5_settings = {"obs_dim": [5, 64, 32],
                "state_dim": [5, 64, 32], 
                "seq_length": 10}

# Calculate feature dimension: embed_dim * downsampled_H * downsampled_W
# After 4x4 downsampling: 64x32 -> 16x8, with embed_dim=64: 64 * 16 * 8 = 8192
ERA5_settings["state_feature_dim"] = [64 * 16 * 8, 512]


'''
================================
Factorized Attention Components
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

        # Row-wise attention
        q_x = rearrange(q, 'b head h w d -> (b head w) h d')
        k_x = rearrange(k, 'b head h w d -> (b head w) h d')
        v_x = rearrange(v, 'b head h w d -> (b head w) h d')
        attn_x = torch.softmax(torch.bmm(q_x, k_x.transpose(-2, -1)) / (self.head_dim ** 0.5), dim=-1)
        out_x = torch.bmm(attn_x, v_x)
        out_x = rearrange(out_x, '(b head w) h d -> b head h w d', b=B, head=self.heads, w=W)

        # Column-wise attention
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


'''
================================
NN features for ERA5 system with Factorized Attention
================================
'''


class ERA5_K_S(Module):
    def __init__(self, *args, **kwargs) -> None:
        super(ERA5_K_S, self).__init__(*args, **kwargs)
        self.input_dim, self.w, self.h = ERA5_settings["state_dim"]  # 5, 64, 32
        self.hidden_dims = ERA5_settings["state_feature_dim"]

        # Factorized attention parameters
        self.embed_dim = 64
        self.num_layers = 1
        self.num_heads = 4

        # Input projection: convert 5-channel input to embedding dimension
        self.input_proj = nn.Conv2d(self.input_dim, self.embed_dim, kernel_size=7, stride=1, padding=3)
        
        # Downsample to reduce spatial dimensions (64x32 -> 16x8)
        self.downsample = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=4, stride=4)

        # Factorized attention layers
        self.factorized_layers = nn.ModuleList([
            FactorizedBlock(self.embed_dim, self.num_heads)
            for _ in range(self.num_layers)
        ])

        # Output processing
        self.flatten = nn.Flatten()
        self.final_proj = nn.Linear(self.hidden_dims[0], self.hidden_dims[1])
        self.dropout = nn.Dropout(0.1)

    def forward(self, state: torch.Tensor):
        B, C, H, W = state.shape

        # Input projection and downsampling
        x = self.input_proj(state)  # [B, 64, 64, 32]
        x = self.downsample(x)      # [B, 64, 16, 8]

        # Convert to format expected by factorized attention: [B, H, W, C]
        x = x.permute(0, 2, 3, 1)  # [B, 16, 8, 64]

        # Create positional encodings
        pos_x = torch.linspace(0, 2 * np.pi, x.shape[1], device=state.device)
        pos_y = torch.linspace(0, 2 * np.pi, x.shape[2], device=state.device)

        # Apply factorized attention layers
        for layer in self.factorized_layers:
            x = layer(x, pos_x, pos_y)

        # Convert back to channel-first format and flatten
        x = x.permute(0, 3, 1, 2)  # [B, 64, 16, 8]
        x = self.flatten(x)
        x = self.dropout(x)
        z = self.final_proj(x)

        return z


class ERA5_K_S_preimage(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(ERA5_K_S_preimage, self).__init__(*args, **kwargs)
        self.input_dim, self.w, self.h = ERA5_settings["state_dim"]  # 5, 64, 32
        self.hidden_dims = ERA5_settings["state_feature_dim"]
        
        # Match the encoder's dimensions: 64 channels, 16x8 spatial size
        self.embed_dim = 64
        self.spatial_h, self.spatial_w = 16, 8

        # Linear layer to reconstruct the flattened feature map
        self.linear = nn.Linear(self.hidden_dims[1], self.embed_dim * self.spatial_h * self.spatial_w)
        
        # Progressive upsampling from 16x8 to 64x32
        self.upsample1 = nn.ConvTranspose2d(self.embed_dim, 128, kernel_size=4, stride=2, padding=1)  # 16x8 -> 32x16
        self.upsample2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 32x16 -> 64x32
        
        # Final output projection
        self.output_proj = nn.Conv2d(64, self.input_dim, kernel_size=3, stride=1, padding=1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        # Output refinement layers
        self.output_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=self.input_dim, kernel_size=3, stride=1, padding=1)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, z: torch.Tensor):
        batch_size = z.shape[0]
        
        # Linear transformation and reshape
        x = self.linear(z)  # [batch, embed_dim * 16 * 8]
        x = self.relu(x)
        x = self.dropout(x)
        
        # Reshape to feature map: [batch, embed_dim, 16, 8]
        x = x.view(batch_size, self.embed_dim, self.spatial_h, self.spatial_w)
        
        # Upsample to original resolution
        x = self.upsample1(x)  # [batch, 128, 32, 16]
        x = self.relu(x)
        
        x = self.upsample2(x)  # [batch, 64, 64, 32]
        x = self.relu(x)
        
        # Final output projection
        x = self.output_proj(x)  # [batch, 5, 64, 32]
        
        # Output refinement
        recon_s = self.output_conv(x)
        recon_s = self.sigmoid(recon_s)

        return recon_s


'''
=============================
Operators for ERA5 system with Factorized Attention
=============================
'''


class ERA5_C_FORWARD(base_forward_model):
    def __init__(self, *args, **kwargs) -> None:
        K_S = ERA5_K_S()
        K_S_preimage = ERA5_K_S_preimage()
        seq_length = ERA5_settings["seq_length"]
        super(ERA5_C_FORWARD, self).__init__(K_S=K_S,
                                            K_S_preimage=K_S_preimage, 
                                            seq_length=seq_length,
                                            *args, **kwargs)


if __name__ == "__main__":
    # Simple test
    model = ERA5_C_FORWARD()
    print(f"Model created successfully!")
    print(f"State dimension: {ERA5_settings['state_dim']}")
    print(f"Sequence length: {ERA5_settings['seq_length']}")
    
    # Test with dummy input
    dummy_input = torch.randn(1, 5, 64, 32)
    with torch.no_grad():
        encoded = model.K_S(dummy_input)
        decoded = model.K_S_preimage(encoded)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Encoded shape: {encoded.shape}")
    print(f"Decoded shape: {decoded.shape}")
    print("Test completed successfully!")