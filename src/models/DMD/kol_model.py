from torch.nn.modules import Module
from base import *
import torch.nn.functional as F

'''
================================
Features for Kolmogorov Flow system
================================
'''

class KOL_phi_S(phi_S_BASE):
    """
    State encoder for Kolmogorov flow
    Much deeper architecture with residual connections and higher capacity
    """
    def __init__(self, config, *args, **kwargs) -> None:
        self.input_channels = config.input_channels  # 1 for single field
        self.input_size = config.input_size  # 64 for 64x64 resolution
        self.latent_dim = config.latent_dim  # Increased latent dimension
        
        # For compatibility with base.py
        self.hidden_dims = [self.latent_dim]
        
        features = nn.ModuleList()
        
        # Initial convolution with larger kernel for global context
        features.append(nn.Conv2d(self.input_channels, 32, kernel_size=7, stride=1, padding=3))
        features.append(nn.BatchNorm2d(32))
        features.append(nn.ReLU(inplace=True))
        
        # Residual Block 1: 64x64 -> 32x32
        features.append(self._make_residual_block(32, 64, stride=2))
        
        # Residual Block 2: 32x32 -> 16x16  
        features.append(self._make_residual_block(64, 128, stride=2))
        
        # Residual Block 3: 16x16 -> 8x8
        features.append(self._make_residual_block(128, 256, stride=2))
        
        # Residual Block 4: 8x8 -> 4x4
        features.append(self._make_residual_block(256, 512, stride=2))
        
        # Additional residual block for more capacity
        features.append(self._make_residual_block(512, 512, stride=1))
        
        # Multi-scale feature aggregation
        features.append(nn.AdaptiveAvgPool2d(1))  # Global context
        features.append(nn.Flatten())
        
        # More sophisticated FC layers
        features.append(nn.Linear(512, 1024))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.Dropout(0.1))
        
        features.append(nn.Linear(1024, 512))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.Dropout(0.1))
        
        features.append(nn.Linear(512, self.latent_dim))
        
        features = nn.Sequential(*features)
        super(KOL_phi_S, self).__init__(features, *args, **kwargs)
        
        print(f'[INFO] KOL_phi_S: Input channels={self.input_channels}, '
              f'Input size={self.input_size}x{self.input_size}, '
              f'Output dim={self.latent_dim}')
        print(f'[INFO] architecture with residual connections and higher capacity')
    
    def _make_residual_block(self, in_channels, out_channels, stride=1):
        """Create a residual block"""
        layers = []
        
        # Main path
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            shortcut = nn.Identity()
        
        return ResidualBlock(nn.Sequential(*layers), shortcut)


class ResidualBlock(nn.Module):
    """Residual block implementation"""
    def __init__(self, main_path, shortcut):
        super(ResidualBlock, self).__init__()
        self.main_path = main_path
        self.shortcut = shortcut
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.main_path(x)
        out += residual
        out = self.relu(out)
        return out


class KOL_phi_inv_S(phi_inv_S_BASE):
    """
    State decoder for Kolmogorov flow
    """
    def __init__(self, config, *args, **kwargs) -> None:
        self.input_channels = config.input_channels
        self.input_size = config.input_size
        self.latent_dim = config.latent_dim
        
        # For compatibility with base.py
        self.hidden_dims = [self.latent_dim]
        
        features = nn.ModuleList()
        
        # FC layers
        features.append(nn.Linear(self.latent_dim, 512))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.Dropout(0.1))
        
        features.append(nn.Linear(512, 1024))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.Dropout(0.1))
        
        features.append(nn.Linear(1024, 512 * 4 * 4))
        features.append(nn.ReLU(inplace=True))
        
        # Reshape
        features.append(View((-1, 512, 4, 4)))
        
        # transposed convolutions with residual connections
        # 4x4 -> 8x8
        features.append(self._make_transpose_residual_block(512, 256, stride=2))
        
        # 8x8 -> 16x16
        features.append(self._make_transpose_residual_block(256, 128, stride=2))
        
        # 16x16 -> 32x32
        features.append(self._make_transpose_residual_block(128, 64, stride=2))
        
        # 32x32 -> 64x64
        features.append(nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1))
        features.append(nn.BatchNorm2d(32))
        features.append(nn.ReLU(inplace=True))
        
        # Final output layer
        features.append(nn.Conv2d(32, self.input_channels, kernel_size=3, stride=1, padding=1))
        
        features = nn.Sequential(*features)
        super(KOL_phi_inv_S, self).__init__(features, *args, **kwargs)
        
        print(f'[INFO] KOL_phi_inv_S: Output channels={self.input_channels}, '
              f'Output size={self.input_size}x{self.input_size}, '
              f'Input dim={self.latent_dim}')
    
    def _make_transpose_residual_block(self, in_channels, out_channels, stride=2):
        """Create a transpose residual block"""
        main_path = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        shortcut = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        return TransposeResidualBlock(main_path, shortcut)


class TransposeResidualBlock(nn.Module):
    """Transpose residual block"""
    def __init__(self, main_path, shortcut):
        super(TransposeResidualBlock, self).__init__()
        self.main_path = main_path
        self.shortcut = shortcut
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.main_path(x)
        out += residual
        out = self.relu(out)
        return out


# Helper class for reshaping
class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape
    
    def forward(self, x):
        return x.view(*self.shape)


'''
=============================
Forward Model for Kolmogorov Flow
=============================
'''

class KOL_forward_model(forward_model):
    """
    forward model for Kolmogorov flow using deeper architecture
    """
    def __init__(self, config, *args, **kwargs) -> None:
        phi_S = KOL_phi_S(config)
        phi_inv_S = KOL_phi_inv_S(config)
        seq_length = config.seq_length
        
        super(KOL_forward_model, self).__init__(
            phi_S=phi_S,
            phi_inv_S=phi_inv_S, 
            seq_length=seq_length,
            *args, **kwargs
        )
        
        print(f'[INFO] KOL_forward_model initialized')
        print(f'[INFO] Latent dimension: {self.hidden_dim}')
        print(f'[INFO] Model capacity significantly increased for chaotic dynamics')
    
    def verify_model(self, sample_input):
        """
        Verify the model architecture with a sample input
        Args:
            sample_input: torch.Tensor of shape [B, C, H, W]
        """
        print(f'[INFO] Verifying model with input shape: {sample_input.shape}')
        
        with torch.no_grad():
            # Test encoder
            z = self.phi_S(sample_input)
            print(f'[INFO] Encoded shape: {z.shape}')
            
            # Test decoder
            reconstructed = self.phi_inv_S(z)
            print(f'[INFO] Reconstructed shape: {reconstructed.shape}')
            
            # Check if reconstruction shape matches input
            if reconstructed.shape == sample_input.shape:
                print('[INFO] ✓ model architecture verified successfully!')
                
                # Calculate reconstruction error
                recon_error = F.mse_loss(reconstructed, sample_input)
                print(f'[INFO] Initial reconstruction error: {recon_error.item():.6f}')
            else:
                print(f'[ERROR] Shape mismatch! Input: {sample_input.shape}, '
                      f'Output: {reconstructed.shape}')
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        encoder_params = sum(p.numel() for p in self.phi_S.parameters())
        decoder_params = sum(p.numel() for p in self.phi_inv_S.parameters())
        
        print(f'[INFO] Total parameters: {total_params:,}')
        print(f'[INFO] Encoder parameters: {encoder_params:,}')
        print(f'[INFO] Decoder parameters: {decoder_params:,}')
                
        return z, reconstructed


# configuration class
class KolmogorovConfig:
    def __init__(self):
        # Data dimensions
        self.input_channels = 1      # Single channel
        self.input_size = 64         # 64x64 spatial resolution
        self.seq_length = 10         # Time sequence length
        
        # model architecture
        self.latent_dim = 256        # Significantly increased latent space
        
        # Training parameters
        self.batch_size = 4          # Smaller due to larger model
        self.learning_rate = 0.0005  # Lower learning rate for stability
        self.num_epochs = 100


if __name__ == "__main__":
    # Test the model
    config = KolmogorovConfig()
    
    # Create model
    model = KOL_forward_model(config)
    
    # Create sample data
    sample_batch = torch.randn(2, 1, 64, 64)
    
    # Verify the model
    print("Testing KOL_forward_model...")
    latent, reconstructed = model.verify_model(sample_batch)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n Model summary:")
    print(f"- Total parameters: {total_params:,}")
    print(f"- Input shape: [B, 1, 64, 64]")
    print(f"- Latent shape: [B, {config.latent_dim}]")
    print(f"- Output shape: [B, 1, 64, 64]")
    print(f"- Compression ratio: {4096/config.latent_dim:.1f}:1")