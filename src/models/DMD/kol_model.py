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
    Input: [B, C, H, W] where C=2 (u,v velocity components), H=W=64
    Output: [B, hidden_dim] latent representation
    Architecture: Conv → Conv → Conv → GlobalPool → FC (1 layer)
    """
    def __init__(self, config, *args, **kwargs) -> None:
        self.input_channels = config.input_channels  # 2 for u,v components
        self.input_size = config.input_size  # 64 for 64x64 resolution
        self.latent_dim = config.latent_dim  # Final latent dimension
        
        # For compatibility with base.py, set hidden_dims as a list
        self.hidden_dims = [self.latent_dim]
        
        features = nn.ModuleList()
        
        # Convolutional feature extraction for 2D flow fields
        # Conv block 1: 64x64 -> 32x32
        features.append(nn.Conv2d(self.input_channels, 64, kernel_size=4, stride=2, padding=1))
        features.append(nn.BatchNorm2d(64))
        features.append(nn.ReLU(inplace=True))
        
        # Conv block 2: 32x32 -> 16x16
        features.append(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1))
        features.append(nn.BatchNorm2d(128))
        features.append(nn.ReLU(inplace=True))
        
        # Conv block 3: 16x16 -> 8x8
        features.append(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1))
        features.append(nn.BatchNorm2d(256))
        features.append(nn.ReLU(inplace=True))
        
        # Conv block 4: 8x8 -> 4x4
        features.append(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1))
        features.append(nn.BatchNorm2d(512))
        features.append(nn.ReLU(inplace=True))
        
        # Global Average Pooling: 4x4 -> 1x1
        features.append(nn.AdaptiveAvgPool2d(1))  # [B, 512, 1, 1]
        features.append(nn.Flatten())             # [B, 512]
        
        # Single fully connected layer to latent space
        features.append(nn.Linear(512, self.latent_dim))  # [B, latent_dim]
        
        features = nn.Sequential(*features)
        super(KOL_phi_S, self).__init__(features, *args, **kwargs)
        
        print(f'[INFO] KOL_phi_S: Input channels={self.input_channels}, '
              f'Input size={self.input_size}x{self.input_size}, '
              f'Output dim={self.latent_dim}')
        print(f'[INFO] Architecture: Conv(2→64→128→256→512) → GlobalPool → FC(512→{self.latent_dim})')


class KOL_phi_inv_S(phi_inv_S_BASE):
    """
    State decoder for Kolmogorov flow
    Input: [B, latent_dim] latent representation
    Output: [B, C, H, W] where C=2, H=W=64
    Architecture: FC → Reshape → ConvTranspose → ConvTranspose → ConvTranspose → ConvTranspose
    """
    def __init__(self, config, *args, **kwargs) -> None:
        self.input_channels = config.input_channels  # 2
        self.input_size = config.input_size  # 64
        self.latent_dim = config.latent_dim  # Input latent dimension
        
        # For compatibility with base.py, set hidden_dims as a list  
        self.hidden_dims = [self.latent_dim]
        
        features = nn.ModuleList()
        
        # Single FC layer to prepare for reshaping: latent_dim -> 512*4*4
        features.append(nn.Linear(self.latent_dim, 512 * 4 * 4))
        features.append(nn.ReLU(inplace=True))
        
        # Reshape to [B, 512, 4, 4]
        features.append(View((-1, 512, 4, 4)))
        
        # Transposed convolutions (reverse of encoder)
        # 4x4 -> 8x8
        features.append(nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1))
        features.append(nn.BatchNorm2d(256))
        features.append(nn.ReLU(inplace=True))
        
        # 8x8 -> 16x16
        features.append(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1))
        features.append(nn.BatchNorm2d(128))
        features.append(nn.ReLU(inplace=True))
        
        # 16x16 -> 32x32
        features.append(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1))
        features.append(nn.BatchNorm2d(64))
        features.append(nn.ReLU(inplace=True))
        
        # 32x32 -> 64x64
        features.append(nn.ConvTranspose2d(64, self.input_channels, kernel_size=4, stride=2, padding=1))
        # No activation for the final layer to allow for any real values
        
        features = nn.Sequential(*features)
        super(KOL_phi_inv_S, self).__init__(features, *args, **kwargs)
        
        print(f'[INFO] KOL_phi_inv_S: Output channels={self.input_channels}, '
              f'Output size={self.input_size}x{self.input_size}, '
              f'Input dim={self.latent_dim}')
        print(f'[INFO] Architecture: FC({self.latent_dim}→512*4*4) → ConvTranspose(512→256→128→64→{self.input_channels})')


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
    Forward model for Kolmogorov flow using latent DMD
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
        
        print(f'[INFO] KOL_forward_model initialized with seq_length={seq_length}')
        print(f'[INFO] Latent dimension: {self.hidden_dim}')
        
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
                print('[INFO] ✓ Model architecture verified successfully!')
                
                # Calculate reconstruction error
                recon_error = F.mse_loss(reconstructed, sample_input)
                print(f'[INFO] Initial reconstruction error: {recon_error.item():.6f}')
            else:
                print(f'[ERROR] Shape mismatch! Input: {sample_input.shape}, '
                      f'Output: {reconstructed.shape}')
                
        return z, reconstructed


# Example configuration class for Kolmogorov flow
class KolmogorovConfig:
    def __init__(self):
        # Data dimensions
        self.input_channels = 2  # u and v velocity components
        self.input_size = 64     # 64x64 spatial resolution
        self.seq_length = 320    # Time sequence length
        
        # Model architecture - simplified to single latent dimension
        self.latent_dim = 64     # Final latent space dimension
        
        # Training parameters
        self.batch_size = 8      # Adjust based on GPU memory
        self.learning_rate = 1e-4
        self.num_epochs = 100


if __name__ == "__main__":
    # Test the model
    config = KolmogorovConfig()
    
    # Create model
    model = KOL_forward_model(config)
    
    # Create sample data: [batch_size, channels, height, width]
    sample_batch = torch.randn(4, 2, 64, 64)
    
    # Verify the model
    print("Testing KOL_forward_model...")
    latent, reconstructed = model.verify_model(sample_batch)
    
    print(f"\nModel summary:")
    print(f"- Input shape: [B, 2, 64, 64]")
    print(f"- Latent shape: [B, {config.latent_dim}]")
    print(f"- Output shape: [B, 2, 64, 64]")
    print(f"- DMD matrix shape will be: [{config.latent_dim}, {config.latent_dim}]")
    print(f"- Total encoder parameters: {sum(p.numel() for p in model.phi_S.parameters()):,}")
    print(f"- Total decoder parameters: {sum(p.numel() for p in model.phi_inv_S.parameters()):,}")