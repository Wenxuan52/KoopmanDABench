"""
Koopman Autoencoder Model
Basic implementation with simple physics constraints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np


class Encoder(nn.Module):
    """Encoder network φ_NN: X_t → z_t"""
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = [64, 128, 256],
        activation: str = 'relu',
        use_batch_norm: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Build encoder layers
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'elu':
                layers.append(nn.ELU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.2))
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            in_dim = h_dim
        
        # Final layer to latent space
        layers.append(nn.Linear(in_dim, latent_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, input_dim] or [batch, time, input_dim]
        Returns:
            z: Latent representation [batch, latent_dim] or [batch, time, latent_dim]
        """
        if x.dim() == 3:
            # Handle temporal sequences
            batch, time, dim = x.shape
            x_flat = x.reshape(-1, dim)
            z_flat = self.encoder(x_flat)
            z = z_flat.reshape(batch, time, -1)
        else:
            z = self.encoder(x)
        
        return z


class Decoder(nn.Module):
    """Decoder network H: z_t → Y_t"""
    
    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        activation: str = 'relu',
        use_batch_norm: bool = True,
        dropout: float = 0.0,
        output_activation: Optional[str] = None
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Build decoder layers
        layers = []
        in_dim = latent_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'elu':
                layers.append(nn.ELU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.2))
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            in_dim = h_dim
        
        # Final layer
        layers.append(nn.Linear(in_dim, output_dim))
        
        # Output activation
        if output_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif output_activation == 'tanh':
            layers.append(nn.Tanh())
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent tensor [batch, latent_dim] or [batch, time, latent_dim]
        Returns:
            y: Reconstructed output [batch, output_dim] or [batch, time, output_dim]
        """
        if z.dim() == 3:
            batch, time, dim = z.shape
            z_flat = z.reshape(-1, dim)
            y_flat = self.decoder(z_flat)
            y = y_flat.reshape(batch, time, -1)
        else:
            y = self.decoder(z)
        
        return y


class KoopmanOperator(nn.Module):
    """Koopman operator K for linear dynamics in latent space"""
    
    def __init__(
        self,
        latent_dim: int,
        operator_type: str = 'linear',
        use_bias: bool = False,
        init_scale: float = 0.9
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.operator_type = operator_type
        
        if operator_type == 'linear':
            # Simple linear matrix
            self.K = nn.Linear(latent_dim, latent_dim, bias=use_bias)
            
            # Initialize near identity for stability
            with torch.no_grad():
                self.K.weight.data = torch.eye(latent_dim) * init_scale + \
                                     0.1 * torch.randn(latent_dim, latent_dim)
                if use_bias:
                    self.K.bias.data.zero_()
                    
        elif operator_type == 'diagonal':
            # Diagonal matrix (even simpler dynamics)
            self.diag = nn.Parameter(torch.ones(latent_dim) * init_scale)
            
        elif operator_type == 'block_diagonal':
            # Block diagonal for capturing different timescales
            self.block_size = 2
            self.n_blocks = latent_dim // self.block_size
            self.blocks = nn.ModuleList([
                nn.Linear(self.block_size, self.block_size, bias=False)
                for _ in range(self.n_blocks)
            ])
            
            # Initialize blocks
            for block in self.blocks:
                block.weight.data = torch.eye(self.block_size) * init_scale + \
                                   0.1 * torch.randn(self.block_size, self.block_size)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply Koopman operator"""
        if self.operator_type == 'linear':
            return self.K(z)
            
        elif self.operator_type == 'diagonal':
            return z * self.diag
            
        elif self.operator_type == 'block_diagonal':
            # Apply block diagonal operation
            batch_shape = z.shape[:-1]
            z_blocks = z.reshape(*batch_shape, self.n_blocks, self.block_size)
            
            output_blocks = []
            for i, block in enumerate(self.blocks):
                output_blocks.append(block(z_blocks[..., i, :]))
            
            output = torch.stack(output_blocks, dim=-2)
            return output.reshape(*batch_shape, -1)
    
    def power(self, z: torch.Tensor, n: int) -> torch.Tensor:
        """Apply Koopman operator n times"""
        result = z
        for _ in range(n):
            result = self.forward(result)
        return result
    
    def get_eigenvalues(self) -> torch.Tensor:
        """Get eigenvalues of the Koopman operator (for analysis)"""
        if self.operator_type == 'linear':
            return torch.linalg.eigvals(self.K.weight)
        elif self.operator_type == 'diagonal':
            return self.diag
        else:
            # For block diagonal, compute eigenvalues per block
            eigenvals = []
            for block in self.blocks:
                eigenvals.append(torch.linalg.eigvals(block.weight))
            return torch.cat(eigenvals)


class KoopmanAE(nn.Module):
    """
    Koopman Autoencoder combining encoder, decoder, and Koopman operator
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims_encoder: List[int] = [64, 128, 256],
        hidden_dims_decoder: List[int] = [256, 128, 64],
        activation: str = 'relu',
        operator_type: str = 'linear',
        use_batch_norm: bool = True,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Initialize components
        self.encoder = Encoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims_encoder,
            activation=activation,
            use_batch_norm=use_batch_norm,
            dropout=dropout
        )
        
        self.decoder = Decoder(
            latent_dim=latent_dim,
            output_dim=input_dim,  # Assuming reconstruction to same dimension
            hidden_dims=hidden_dims_decoder,
            activation=activation,
            use_batch_norm=use_batch_norm,
            dropout=dropout
        )
        
        self.koopman = KoopmanOperator(
            latent_dim=latent_dim,
            operator_type=operator_type
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space"""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space"""
        return self.decoder(z)
    
    def forward(
        self, 
        x: torch.Tensor, 
        n_steps: int = 1,
        return_latent: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model
        
        Args:
            x: Input tensor [batch, input_dim] or [batch, time, input_dim]
            n_steps: Number of forward prediction steps
            return_latent: Whether to return latent representations
            
        Returns:
            Dictionary containing predictions and optionally latent states
        """
        # Encode to latent space
        z = self.encode(x)
        
        # Reconstruction
        x_recon = self.decode(z)
        
        # Multi-step predictions
        predictions = []
        z_pred = z
        
        for step in range(n_steps):
            z_pred = self.koopman(z_pred)
            x_pred = self.decode(z_pred)
            predictions.append(x_pred)
        
        outputs = {
            'reconstruction': x_recon,
            'predictions': torch.stack(predictions, dim=1) if predictions else None
        }
        
        if return_latent:
            outputs['z'] = z
            outputs['z_predictions'] = z_pred
        
        return outputs
    
    def predict_trajectory(
        self,
        x0: torch.Tensor,
        n_steps: int,
        return_latent: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Predict future trajectory from initial condition
        
        Args:
            x0: Initial condition [batch, input_dim]
            n_steps: Number of prediction steps
            return_latent: Whether to return latent trajectory
            
        Returns:
            Dictionary with trajectory predictions
        """
        # Encode initial condition
        z0 = self.encode(x0)
        
        # Generate latent trajectory
        z_trajectory = [z0]
        z_current = z0
        
        for _ in range(n_steps):
            z_current = self.koopman(z_current)
            z_trajectory.append(z_current)
        
        # Decode trajectory
        z_traj = torch.stack(z_trajectory, dim=1)
        x_trajectory = self.decode(z_traj)
        
        outputs = {
            'trajectory': x_trajectory,
            'initial_reconstruction': self.decode(z0)
        }
        
        if return_latent:
            outputs['latent_trajectory'] = z_traj
        
        return outputs
    
    def get_koopman_matrix(self) -> torch.Tensor:
        """Get the Koopman operator matrix (if applicable)"""
        if hasattr(self.koopman, 'K'):
            return self.koopman.K.weight.detach()
        elif hasattr(self.koopman, 'diag'):
            return torch.diag(self.koopman.diag.detach())
        else:
            raise NotImplementedError("Matrix extraction not implemented for this operator type")


# Utility function to create model from config
def create_koopman_ae(config: Dict) -> KoopmanAE:
    """Create KoopmanAE model from configuration dictionary"""
    model_config = config.get('model', {})
    
    # For 1D heat equation
    if config.get('dataset') == 'heat1d':
        input_dim = config.get('nx', 100)  # Number of spatial points
    # For 2D advection-diffusion
    elif config.get('dataset') == 'advection_diffusion_2d':
        nx = config.get('nx', 64)
        ny = config.get('ny', 64)
        input_dim = nx * ny  # Flattened spatial dimensions
    else:
        input_dim = model_config.get('input_dim', 100)
    
    model = KoopmanAE(
        input_dim=input_dim,
        latent_dim=model_config.get('latent_dim', 32),
        hidden_dims_encoder=model_config.get('hidden_dims_encoder', [64, 128, 256]),
        hidden_dims_decoder=model_config.get('hidden_dims_decoder', [256, 128, 64]),
        activation=model_config.get('activation', 'relu'),
        operator_type=model_config.get('operator_type', 'linear'),
        use_batch_norm=model_config.get('use_batch_norm', True),
        dropout=model_config.get('dropout', 0.0)
    )
    
    return model