import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict
import sys
import os

# Add DMD module to path
current_dir = os.path.dirname(os.path.abspath(__file__))
dmd_path = os.path.join(current_dir, '..')
sys.path.insert(0, dmd_path)
from DMD.dmd import DMD


class ConvBlock(nn.Module):
    """Convolutional block for encoder"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, use_bn: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.relu(x)
        return x


class DeconvBlock(nn.Module):
    """Deconvolutional block for decoder"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, output_padding: int = 0, 
                 use_bn: bool = True, use_relu: bool = True):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 
                                        stride, padding, output_padding)
        self.use_bn = use_bn
        self.use_relu = use_relu
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if use_relu:
            self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_relu:
            x = self.relu(x)
        return x


class Encoder(nn.Module):
    """Convolutional Encoder"""
    def __init__(self, input_channels: int, latent_dim: int, 
                 base_channels: int = 32, depth: int = 4, use_bn: bool = True):
        super().__init__()
        
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.depth = depth
        
        # Build encoder layers
        layers = []
        in_ch = input_channels
        out_ch = base_channels
        
        for i in range(depth):
            # Use stride 2 for downsampling
            stride = 2 if i > 0 else 1
            layers.append(ConvBlock(in_ch, out_ch, kernel_size=3, stride=stride, 
                                  padding=1, use_bn=use_bn))
            in_ch = out_ch
            out_ch = min(out_ch * 2, 512)  # Cap at 512 channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate size after convolutions (need to be set dynamically)
        self.flatten_size = None
        self.fc = None
    
    def _initialize_fc(self, x: torch.Tensor):
        """Initialize FC layer based on conv output size"""
        batch_size = x.shape[0]
        self.flatten_size = x.view(batch_size, -1).shape[1]
        self.fc = nn.Linear(self.flatten_size, self.latent_dim).to(x.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply convolutions
        x = self.conv_layers(x)
        
        # Initialize FC layer if needed
        if self.fc is None:
            self._initialize_fc(x)
        
        # Flatten and project to latent space
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    """Convolutional Decoder"""
    def __init__(self, latent_dim: int, output_channels: int, output_shape: Tuple[int, int],
                 base_channels: int = 32, depth: int = 4, use_bn: bool = True):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.output_shape = output_shape
        self.base_channels = base_channels
        self.depth = depth
        
        # Calculate initial shape after fc layer
        # Start with output shape and work backwards
        h, w = output_shape
        for _ in range(depth - 1):
            h = (h + 1) // 2
            w = (w + 1) // 2
        
        self.init_h, self.init_w = h, w
        self.init_channels = base_channels * (2 ** (depth - 1))
        self.init_channels = min(self.init_channels, 512)
        
        # FC layer to reshape from latent
        self.fc = nn.Linear(latent_dim, self.init_channels * self.init_h * self.init_w)
        
        # Build decoder layers
        layers = []
        in_ch = self.init_channels
        
        for i in range(depth):
            if i < depth - 1:
                out_ch = in_ch // 2
                out_ch = max(out_ch, base_channels)
                stride = 2
                output_padding = 1
            else:
                out_ch = output_channels
                stride = 1
                output_padding = 0
            
            use_bn_layer = use_bn and i < depth - 1
            use_relu_layer = i < depth - 1
            
            layers.append(DeconvBlock(in_ch, out_ch, kernel_size=3, stride=stride,
                                    padding=1, output_padding=output_padding,
                                    use_bn=use_bn_layer, use_relu=use_relu_layer))
            in_ch = out_ch
        
        self.deconv_layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project and reshape
        x = self.fc(x)
        x = x.view(x.shape[0], self.init_channels, self.init_h, self.init_w)
        
        # Apply deconvolutions
        x = self.deconv_layers(x)
        
        # Ensure output has correct spatial dimensions
        if x.shape[2:] != self.output_shape:
            x = F.interpolate(x, size=self.output_shape, mode='bilinear', align_corners=False)
        
        return x


class ConvAutoencoder(nn.Module):
    """Convolutional Autoencoder"""
    def __init__(self, input_channels: int, input_shape: Tuple[int, int], latent_dim: int,
                 base_channels: int = 32, depth: int = 4, use_bn: bool = True):
        super().__init__()
        
        self.input_channels = input_channels
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        
        self.encoder = Encoder(input_channels, latent_dim, base_channels, depth, use_bn)
        self.decoder = Decoder(latent_dim, input_channels, input_shape, base_channels, depth, use_bn)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        Args:
            x: Input tensor [batch, channels, height, width]
        Returns:
            recon: Reconstructed tensor
            latent: Latent representation
        """
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space"""
        return self.encoder(x)
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode from latent space"""
        return self.decoder(latent)


class LinearMLP(nn.Module):
    """Linear MLP for dynamics in latent space"""
    def __init__(self, latent_dim: int):
        super().__init__()
        self.linear = nn.Linear(latent_dim, latent_dim, bias=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
    
    def get_weight_matrix(self) -> torch.Tensor:
        """Get the weight matrix for linear constraint"""
        return self.linear.weight


class NonlinearMLP(nn.Module):
    """Nonlinear MLP for dynamics in latent space"""
    def __init__(self, latent_dim: int, hidden_dims: Optional[List[int]] = None,
                 activation: str = 'relu', use_bn: bool = True, dropout: float = 0.0):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [latent_dim * 2, latent_dim * 2]
        
        layers = []
        in_dim = latent_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, latent_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Store a linear approximation for weak constraint
        self.linear_approx = nn.Linear(latent_dim, latent_dim, bias=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
    
    def get_linear_approximation(self, x: torch.Tensor) -> torch.Tensor:
        """Get linear approximation for weak constraint"""
        return self.linear_approx(x)


class CAE_LinearMLP(nn.Module):
    """CAE + Linear MLP model"""
    def __init__(self, input_channels: int, input_shape: Tuple[int, int], 
                 latent_dim: int, cae_depth: int = 4, cae_base_channels: int = 32,
                 use_bn: bool = True):
        super().__init__()
        
        self.cae = ConvAutoencoder(input_channels, input_shape, latent_dim,
                                   cae_base_channels, cae_depth, use_bn)
        self.dynamics = LinearMLP(latent_dim)
    
    def forward(self, x_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for one time step prediction
        Args:
            x_t: Input at time t [batch, channels, height, width]
        Returns:
            x_t_recon: Reconstructed x_t
            x_next_pred: Predicted x_{t+1}
            z_t: Latent representation at time t
            z_next: Predicted latent at time t+1
        """
        # Encode current state
        z_t = self.cae.encode(x_t)
        
        # Reconstruct current state
        x_t_recon = self.cae.decode(z_t)
        
        # Predict next latent state
        z_next = self.dynamics(z_t)
        
        # Decode predicted latent
        x_next_pred = self.cae.decode(z_next)
        
        return x_t_recon, x_next_pred, z_t, z_next
    
    def predict_sequence(self, x_0: torch.Tensor, n_steps: int) -> List[torch.Tensor]:
        """
        Predict a sequence of future states
        Args:
            x_0: Initial state [batch, channels, height, width]
            n_steps: Number of steps to predict
        Returns:
            List of predicted states
        """
        predictions = []
        
        # Encode initial state
        z = self.cae.encode(x_0)
        
        # Predict sequence in latent space
        for _ in range(n_steps):
            z = self.dynamics(z)
            x_pred = self.cae.decode(z)
            predictions.append(x_pred)
        
        return predictions


class CAE_WeakLinearMLP(nn.Module):
    """CAE + Nonlinear MLP with weak linear constraint"""
    def __init__(self, input_channels: int, input_shape: Tuple[int, int], 
                 latent_dim: int, hidden_dims: Optional[List[int]] = None,
                 cae_depth: int = 4, cae_base_channels: int = 32,
                 mlp_activation: str = 'relu', use_bn: bool = True, 
                 mlp_dropout: float = 0.0):
        super().__init__()
        
        self.cae = ConvAutoencoder(input_channels, input_shape, latent_dim,
                                   cae_base_channels, cae_depth, use_bn)
        self.dynamics = NonlinearMLP(latent_dim, hidden_dims, mlp_activation, 
                                    use_bn, mlp_dropout)
    
    def forward(self, x_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for one time step prediction
        Args:
            x_t: Input at time t [batch, channels, height, width]
        Returns:
            x_t_recon: Reconstructed x_t
            x_next_pred: Predicted x_{t+1}
            z_t: Latent representation at time t
            z_next: Predicted latent at time t+1 (nonlinear)
            z_next_linear: Linear approximation of next latent (for constraint)
        """
        # Encode current state
        z_t = self.cae.encode(x_t)
        
        # Reconstruct current state
        x_t_recon = self.cae.decode(z_t)
        
        # Predict next latent state (nonlinear)
        z_next = self.dynamics(z_t)
        
        # Get linear approximation for constraint
        z_next_linear = self.dynamics.get_linear_approximation(z_t)
        
        # Decode predicted latent
        x_next_pred = self.cae.decode(z_next)
        
        return x_t_recon, x_next_pred, z_t, z_next, z_next_linear
    
    def predict_sequence(self, x_0: torch.Tensor, n_steps: int) -> List[torch.Tensor]:
        """
        Predict a sequence of future states
        Args:
            x_0: Initial state [batch, channels, height, width]
            n_steps: Number of steps to predict
        Returns:
            List of predicted states
        """
        predictions = []
        
        # Encode initial state
        z = self.cae.encode(x_0)
        
        # Predict sequence in latent space
        for _ in range(n_steps):
            z = self.dynamics(z)
            x_pred = self.cae.decode(z)
            predictions.append(x_pred)
        
        return predictions


class CAE_DMD(nn.Module):
    """CAE + DMD model for dynamics in latent space"""
    def __init__(self, input_channels: int, input_shape: Tuple[int, int], 
                 latent_dim: int, cae_depth: int = 4, cae_base_channels: int = 32,
                 use_bn: bool = True, dmd_svd_rank: Optional[int] = None, 
                 dmd_exact: bool = True):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.dmd_fitted = False
        
        # Convolutional Autoencoder
        self.cae = ConvAutoencoder(input_channels, input_shape, latent_dim,
                                   cae_base_channels, cae_depth, use_bn)
        
        # DMD model for dynamics in latent space
        self.dmd = DMD(svd_rank=dmd_svd_rank, exact=dmd_exact)
        
        # Buffer to store latent states for DMD training
        self.latent_buffer = []
        self.max_buffer_size = 10000  # Limit memory usage
    
    def forward(self, x_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for one time step prediction
        Args:
            x_t: Input at time t [batch, channels, height, width]
        Returns:
            x_t_recon: Reconstructed x_t
            x_next_pred: Predicted x_{t+1} (if DMD is fitted, otherwise None)
            z_t: Latent representation at time t
            z_next: Predicted latent at time t+1 (if DMD is fitted, otherwise None)
        """
        # Encode current state
        z_t = self.cae.encode(x_t)  # [batch, latent_dim]
        
        # Reconstruct current state
        x_t_recon = self.cae.decode(z_t)
        
        # Predict next latent state using DMD (if fitted)
        if self.dmd_fitted:
            with torch.no_grad():
                # Convert to numpy for DMD prediction
                z_t_np = z_t.detach().cpu().numpy()  # [batch, latent_dim]
                
                # Predict one step for each sample in batch
                z_next_list = []
                for i in range(z_t_np.shape[0]):
                    x0 = z_t_np[i]  # [latent_dim]
                    z_pred = self.dmd.predict(x0=x0, n_steps=1)  # [latent_dim, 1]
                    z_next_list.append(z_pred[:, 0])  # [latent_dim]
                
                # Stack and convert back to tensor
                z_next_np = np.stack(z_next_list, axis=0)  # [batch, latent_dim]
                z_next = torch.from_numpy(z_next_np).float().to(z_t.device)
            
            # Decode predicted latent
            x_next_pred = self.cae.decode(z_next)
        else:
            z_next = None
            x_next_pred = None
        
        return x_t_recon, x_next_pred, z_t, z_next
    
    def collect_latent_data(self, x_t: torch.Tensor, x_next: torch.Tensor):
        """
        Collect latent state pairs for DMD training
        Args:
            x_t: Current state [batch, channels, height, width]
            x_next: Next state [batch, channels, height, width]
        """
        with torch.no_grad():
            # Encode both states
            z_t = self.cae.encode(x_t).detach().cpu().numpy()      # [batch, latent_dim]
            z_next = self.cae.encode(x_next).detach().cpu().numpy() # [batch, latent_dim]
            
            # Add to buffer
            for i in range(z_t.shape[0]):
                if len(self.latent_buffer) >= self.max_buffer_size:
                    # Remove oldest entries to maintain buffer size
                    self.latent_buffer.pop(0)
                
                self.latent_buffer.append((z_t[i], z_next[i]))
    
    def fit_dmd(self, min_samples: int = 100):
        """
        Fit DMD model using collected latent data
        Args:
            min_samples: Minimum number of samples required to fit DMD
        """
        if len(self.latent_buffer) < min_samples:
            print(f"Not enough samples for DMD fitting. Have {len(self.latent_buffer)}, need {min_samples}")
            return False
        
        print(f"Fitting DMD with {len(self.latent_buffer)} latent state pairs...")
        
        # Convert buffer to training arrays
        X_train = np.array([pair[0] for pair in self.latent_buffer])  # [n_samples, latent_dim]
        Y_train = np.array([pair[1] for pair in self.latent_buffer])  # [n_samples, latent_dim]
        
        print(f"DMD training data shapes: X_train={X_train.shape}, Y_train={Y_train.shape}")
        
        # Fit DMD model
        try:
            self.dmd.fit(X_train, Y_train)
            self.dmd_fitted = True
            print("DMD fitting completed successfully")
            
            # Print DMD statistics
            if hasattr(self.dmd, 'Lambda') and self.dmd.Lambda is not None:
                eigenvalue_mags = np.abs(self.dmd.Lambda)
                growth_rates = np.real(np.log(self.dmd.Lambda + 1e-12))
                stable_modes = np.sum(growth_rates <= 0)
                unstable_modes = np.sum(growth_rates > 0)
                
                print(f"DMD Statistics:")
                print(f"  Number of modes: {len(self.dmd.Lambda)}")
                print(f"  Stable modes: {stable_modes}")
                print(f"  Unstable modes: {unstable_modes}")
                print(f"  Eigenvalue magnitudes range: [{eigenvalue_mags.min():.6f}, {eigenvalue_mags.max():.6f}]")
            
            return True
            
        except Exception as e:
            print(f"DMD fitting failed: {e}")
            return False
    
    def predict_sequence(self, x_0: torch.Tensor, n_steps: int) -> List[torch.Tensor]:
        """
        Predict a sequence of future states using DMD
        Args:
            x_0: Initial state [batch, channels, height, width]
            n_steps: Number of steps to predict
        Returns:
            List of predicted states
        """
        if not self.dmd_fitted:
            raise RuntimeError("DMD model must be fitted before sequence prediction")
        
        predictions = []
        batch_size = x_0.shape[0]
        
        with torch.no_grad():
            # Encode initial state
            z_0 = self.cae.encode(x_0)  # [batch, latent_dim]
            z_0_np = z_0.detach().cpu().numpy()  # [batch, latent_dim]
            
            # Predict sequence for each sample in batch
            for i in range(batch_size):
                # Get initial condition for this sample
                x0 = z_0_np[i]  # [latent_dim]
                
                # Predict sequence in latent space using DMD
                Z_pred = self.dmd.predict(x0=x0, n_steps=n_steps)  # [latent_dim, n_steps]
                
                # Convert back to tensor and decode each step
                sample_predictions = []
                for t in range(n_steps):
                    z_t = torch.from_numpy(Z_pred[:, t]).float().to(x_0.device).unsqueeze(0)  # [1, latent_dim]
                    x_t_pred = self.cae.decode(z_t)  # [1, channels, height, width]
                    sample_predictions.append(x_t_pred)
                
                # Stack predictions for this sample
                if i == 0:
                    # Initialize predictions list
                    for t in range(n_steps):
                        predictions.append(sample_predictions[t])
                else:
                    # Concatenate with existing predictions
                    for t in range(n_steps):
                        predictions[t] = torch.cat([predictions[t], sample_predictions[t]], dim=0)
        
        return predictions
    
    def get_dmd_reconstruction_error(self) -> Optional[Dict[str, float]]:
        """
        Get DMD reconstruction error on collected latent data
        Returns:
            Dictionary of error metrics if DMD is fitted, None otherwise
        """
        if not self.dmd_fitted or len(self.latent_buffer) == 0:
            return None
        
        # Create test data from buffer
        X_test = np.array([pair[0] for pair in self.latent_buffer[-100:]])  # Use last 100 samples
        Y_test = np.array([pair[1] for pair in self.latent_buffer[-100:]])
        
        # Transpose for DMD format [latent_dim, n_samples]
        X_test_t = X_test.T
        Y_test_t = Y_test.T
        
        # Get DMD reconstruction
        Y_pred_t = self.dmd.reconstruct(X_test_t)
        
        # Compute errors
        errors = self.dmd.compute_error(Y_test_t, Y_pred_t)
        return errors
    
    def save_dmd(self, filepath: str):
        """Save DMD model"""
        if self.dmd_fitted:
            self.dmd.save(filepath)
            print(f"DMD model saved to {filepath}")
        else:
            print("DMD model not fitted, nothing to save")
    
    def load_dmd(self, filepath: str):
        """Load DMD model"""
        try:
            self.dmd.load(filepath)
            self.dmd_fitted = True
            print(f"DMD model loaded from {filepath}")
        except Exception as e:
            print(f"Failed to load DMD model: {e}")
            self.dmd_fitted = False
    
    def clear_buffer(self):
        """Clear latent data buffer"""
        self.latent_buffer.clear()
        print("Latent data buffer cleared")


# Helper function to create models
def create_cae_linear_mlp(input_channels: int, input_shape: Tuple[int, int], 
                         latent_dim: int, **kwargs) -> CAE_LinearMLP:
    """Create CAE + Linear MLP model"""
    return CAE_LinearMLP(input_channels, input_shape, latent_dim, **kwargs)


def create_cae_weaklinear_mlp(input_channels: int, input_shape: Tuple[int, int], 
                              latent_dim: int, **kwargs) -> CAE_WeakLinearMLP:
    """Create CAE + Weak Linear MLP model"""
    return CAE_WeakLinearMLP(input_channels, input_shape, latent_dim, **kwargs)

def create_cae_dmd(input_channels: int, input_shape: Tuple[int, int], 
                   latent_dim: int, dmd_svd_rank: Optional[int] = None,
                   dmd_exact: bool = True, **kwargs) -> CAE_DMD:
    """
    Create CAE + DMD model
    """
    return CAE_DMD(input_channels, input_shape, latent_dim, 
                   dmd_svd_rank=dmd_svd_rank, dmd_exact=dmd_exact, **kwargs)