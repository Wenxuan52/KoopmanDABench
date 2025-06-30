import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, Any, Tuple, Union, Callable
from scipy.linalg import inv, solve
import os
import sys

# Add parent directory to path to import DMD
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from models.DMD.dmd import DMD


class DMD4DVAR:
    """
    DMD + 4D-Variational Data Assimilation
    
    This class combines Dynamic Mode Decomposition (DMD) with 4D-VAR data assimilation
    in latent space. The assimilation can be performed using either Kalman filter
    or variational optimization methods.
    """
    
    def __init__(self, 
                 encoder: Optional[nn.Module] = None,
                 decoder: Optional[nn.Module] = None,
                 dmd_params: Optional[Dict[str, Any]] = None,
                 device: str = 'cpu'):
        """
        Initialize DMD+4DVAR model
        
        Args:
            encoder: Neural network encoder E(·) to map from physical to latent space
            decoder: Neural network decoder D(·) to map from latent to physical space
            dmd_params: Dictionary of parameters for DMD model
            device: Device for computation ('cpu' or 'cuda')
        """
        self.device = device
        self.encoder = encoder
        self.decoder = decoder
        
        # Initialize DMD with provided parameters or defaults
        if dmd_params is None:
            dmd_params = {'svd_rank': None, 'exact': True}
        self.dmd = DMD(**dmd_params, device=device)
        
        # Data assimilation parameters
        self.B = None  # Background error covariance matrix
        self.R = None  # Observation error covariance matrix
        
        # Training state
        self._dmd_fitted = False
        self._latent_dim = None
        self._use_latent = False  # Initialize this attribute
        
    def train_dmd(self, X_train: np.ndarray, Y_train: np.ndarray, 
                  use_latent: bool = True) -> 'DMD4DVAR':
        """
        Train DMD model on training data
        
        Args:
            X_train: Training data at time t [n_samples, n_features]
            Y_train: Training data at time t+1 [n_samples, n_features]
            use_latent: If True, train DMD in latent space; if False, train in physical space
        
        Returns:
            Self
        """
        if use_latent and (self.encoder is None or self.decoder is None):
            raise ValueError("Encoder and decoder must be provided to train in latent space")
        
        if use_latent:
            # Convert to torch tensors
            X_train_torch = torch.tensor(X_train, dtype=torch.float32).to(self.device)
            Y_train_torch = torch.tensor(Y_train, dtype=torch.float32).to(self.device)
            
            # Encode to latent space
            with torch.no_grad():
                X_train_latent = self.encoder(X_train_torch).cpu().numpy()
                Y_train_latent = self.encoder(Y_train_torch).cpu().numpy()
            
            # Store latent dimension
            self._latent_dim = X_train_latent.shape[1]
            
            # Train DMD in latent space
            self.dmd.fit(X_train_latent, Y_train_latent)
            print(f"DMD trained in latent space with dimension {self._latent_dim}")
        else:
            # Train DMD directly in physical space
            self.dmd.fit(X_train, Y_train)
            self._latent_dim = X_train.shape[1]
            print(f"DMD trained in physical space with dimension {X_train.shape[1]}")
        
        self._dmd_fitted = True
        self._use_latent = use_latent
        return self
    
    def load_dmd(self, filepath: str, latent_dim: Optional[int] = None) -> 'DMD4DVAR':
        """
        Load pre-trained DMD model from file
        
        Args:
            filepath: Path to saved DMD model
            latent_dim: Dimension of latent space (required if DMD was trained in latent space)
        
        Returns:
            Self
        """
        self.dmd.load(filepath)
        self._dmd_fitted = True
        
        # Set latent dimension
        if latent_dim is not None:
            self._latent_dim = latent_dim
        elif hasattr(self.dmd, 'n_features'):
            self._latent_dim = self.dmd.n_features
        else:
            raise ValueError("Cannot determine latent dimension from loaded DMD model")
        
        print(f"DMD model loaded from {filepath}")
        print(f"Latent dimension: {self._latent_dim}")
        return self
    
    def set_error_covariances(self, B: np.ndarray, R: np.ndarray):
        """
        Set error covariance matrices for data assimilation
        
        Args:
            B: Background error covariance matrix [latent_dim, latent_dim]
            R: Observation error covariance matrix [obs_dim, obs_dim]
        """
        self.B = B
        self.R = R
        
        # Validate dimensions
        if B.shape[0] != B.shape[1]:
            raise ValueError(f"B must be square matrix, got shape {B.shape}")
        if R.shape[0] != R.shape[1]:
            raise ValueError(f"R must be square matrix, got shape {R.shape}")
        
        print(f"Error covariances set: B shape {B.shape}, R shape {R.shape}")
    
    def assimilate(self, 
                   x0: np.ndarray,
                   observations: Dict[int, np.ndarray],
                   H_operators: Dict[int, np.ndarray],
                   method: str = 'kalman',
                   window_size: int = 1,
                   max_iter: int = 100,
                   tol: float = 1e-6) -> Dict[str, Any]:
        """
        Perform 4D-VAR data assimilation
        
        Args:
            x0: Initial condition in physical space [n_features]
            observations: Dictionary mapping time index to observations {t: y_t}
            H_operators: Dictionary mapping time index to observation operators {t: H_t}
            method: 'kalman' for Kalman filter or 'var' for variational optimization
            window_size: Assimilation window size (number of time steps)
            max_iter: Maximum iterations for variational method
            tol: Convergence tolerance for variational method
        
        Returns:
            Dictionary containing:
                - 'analysis': Analyzed initial condition in physical space
                - 'trajectory': Analyzed trajectory in physical space
                - 'cost': Final cost function value
                - 'convergence_info': Convergence information (for var method)
        """
        if not self._dmd_fitted:
            raise RuntimeError("DMD model must be fitted or loaded before assimilation")
        
        if self.B is None or self.R is None:
            raise RuntimeError("Error covariances must be set before assimilation")
        
        # Convert initial condition to latent space if needed
        if self._use_latent and self.encoder is not None:
            x0_torch = torch.tensor(x0, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                x0_latent = self.encoder(x0_torch.unsqueeze(0)).squeeze(0).cpu().numpy()
        else:
            x0_latent = x0
        
        # Get background state (one-step DMD prediction)
        xb_latent = self.dmd.predict(x0=x0_latent, n_steps=1)[:, 0]
        
        print(f"Starting {method} assimilation with window size {window_size}")
        print(f"Number of observation times: {len(observations)}")
        
        if method == 'kalman':
            result = self._assimilate_kalman(
                x0_latent, xb_latent, observations, H_operators, window_size
            )
        elif method == 'var':
            result = self._assimilate_variational(
                x0_latent, xb_latent, observations, H_operators, 
                window_size, max_iter, tol
            )
        else:
            raise ValueError(f"Unknown method: {method}. Choose 'kalman' or 'var'")
        
        # Convert results back to physical space if needed
        if self._use_latent and self.decoder is not None:
            # Analysis
            xa_latent_torch = torch.tensor(result['analysis'], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                result['analysis'] = self.decoder(xa_latent_torch.unsqueeze(0)).squeeze(0).cpu().numpy()
            
            # Trajectory
            traj_latent_torch = torch.tensor(result['trajectory'], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                traj_physical = []
                for t in range(traj_latent_torch.shape[1]):
                    state_physical = self.decoder(traj_latent_torch[:, t].unsqueeze(0)).squeeze(0).cpu().numpy()
                    traj_physical.append(state_physical)
                result['trajectory'] = np.column_stack(traj_physical)
        
        return result
    
    def _assimilate_kalman(self, x0_latent: np.ndarray, xb_latent: np.ndarray,
                          observations: Dict[int, np.ndarray],
                          H_operators: Dict[int, np.ndarray],
                          window_size: int) -> Dict[str, Any]:
        """
        Kalman filter-based assimilation (exact for linear operators)
        """
        # Initialize
        xa = x0_latent.copy()  # Analysis state
        Pa = self.B.copy()     # Analysis error covariance
        
        trajectory = [xa]
        costs = []
        
        # Get sorted observation times
        obs_times = sorted(observations.keys())
        
        # Process observations in windows
        for window_start in range(0, len(obs_times), window_size):
            window_end = min(window_start + window_size, len(obs_times))
            window_obs_times = obs_times[window_start:window_end]
            
            # Forecast to first observation time in window
            if window_start > 0:
                n_forecast_steps = window_obs_times[0] - obs_times[window_start - 1]
                forecast_traj = self.dmd.predict(x0=xa, n_steps=n_forecast_steps)
                xa = forecast_traj[:, -1]
                
                # Evolve error covariance
                # Since we're working in potentially high-dimensional space, 
                # we use a simplified covariance propagation
                if n_forecast_steps > 0:
                    # Add process noise proportional to forecast steps
                    # This avoids dimension mismatch issues
                    process_noise = self.B * 0.1 * n_forecast_steps
                    Pa = Pa + process_noise
            
            # Assimilate observations in window
            for t in window_obs_times:
                # Get observation and operator
                y_t = observations[t]
                H_t = H_operators[t]
                
                # Innovation
                if self._use_latent and self.decoder is not None:
                    # Decode to physical space for observation operator
                    xa_torch = torch.tensor(xa, dtype=torch.float32).to(self.device)
                    with torch.no_grad():
                        xa_physical = self.decoder(xa_torch.unsqueeze(0)).squeeze(0).cpu().numpy()
                    innovation = y_t - H_t @ xa_physical
                else:
                    innovation = y_t - H_t @ xa
                
                # Kalman gain
                if self._use_latent and self.decoder is not None:
                    # Linearize decoder around current state
                    # For simplicity, we approximate with finite differences
                    D_linearized = self._linearize_decoder(xa)
                    H_eff = H_t @ D_linearized
                else:
                    H_eff = H_t
                
                S = H_eff @ Pa @ H_eff.T + self.R  # Innovation covariance
                K = Pa @ H_eff.T @ inv(S)          # Kalman gain
                
                # Update
                xa = xa + K @ innovation
                Pa = (np.eye(len(xa)) - K @ H_eff) @ Pa
                
                # Store trajectory
                trajectory.append(xa.copy())
                
                # Compute cost
                cost_b = 0.5 * (xa - xb_latent).T @ inv(self.B) @ (xa - xb_latent)
                cost_o = 0.5 * innovation.T @ inv(self.R) @ innovation
                costs.append(float(cost_b + cost_o))
        
        return {
            'background': xb_latent,
            'analysis': xa,
            'trajectory': np.column_stack(trajectory),
            'cost': costs[-1] if costs else 0.0,
            'cost_history': costs
        }
    
    def _assimilate_variational(self, x0_latent: np.ndarray, xb_latent: np.ndarray,
                               observations: Dict[int, np.ndarray],
                               H_operators: Dict[int, np.ndarray],
                               window_size: int, max_iter: int, tol: float) -> Dict[str, Any]:
        """
        Variational optimization-based assimilation
        """
        # Convert to torch for optimization
        x0_var = torch.tensor(x0_latent, dtype=torch.float32, requires_grad=True).to(self.device)
        xb_torch = torch.tensor(xb_latent, dtype=torch.float32).to(self.device)
        B_inv = torch.tensor(inv(self.B), dtype=torch.float32).to(self.device)
        R_inv = torch.tensor(inv(self.R), dtype=torch.float32).to(self.device)
        
        # Optimizer
        optimizer = optim.LBFGS([x0_var], lr=0.1, max_iter=20)
        
        # Get sorted observation times
        obs_times = sorted(observations.keys())
        max_time = max(obs_times)
        
        cost_history = []
        
        def cost_function():
            # Background term
            diff_b = x0_var - xb_torch
            cost = 0.5 * torch.matmul(torch.matmul(diff_b, B_inv), diff_b)
            
            # Forecast and compute observation terms
            current_state = x0_var.detach().cpu().numpy()
            forecast_traj = self.dmd.predict(x0=current_state, n_steps=max_time)
            
            for t in obs_times:
                if t < max_time:
                    x_t = forecast_traj[:, t]
                else:
                    x_t = forecast_traj[:, -1]
                
                # Convert to torch
                x_t_torch = torch.tensor(x_t, dtype=torch.float32).to(self.device)
                
                # Apply decoder if in latent space
                if self._use_latent and self.decoder is not None:
                    x_t_physical = self.decoder(x_t_torch.unsqueeze(0)).squeeze(0)
                else:
                    x_t_physical = x_t_torch
                
                # Observation operator
                H_t = torch.tensor(H_operators[t], dtype=torch.float32).to(self.device)
                y_t = torch.tensor(observations[t], dtype=torch.float32).to(self.device)
                
                # Innovation
                innovation = y_t - torch.matmul(H_t, x_t_physical)
                
                # Observation cost
                cost += 0.5 * torch.matmul(torch.matmul(innovation, R_inv), innovation)
            
            cost_history.append(float(cost.item()))
            return cost
        
        # Optimize
        for iter_count in range(max_iter):
            old_cost = cost_history[-1] if cost_history else float('inf')
            
            def closure():
                optimizer.zero_grad()
                cost = cost_function()
                cost.backward()
                return cost
            
            optimizer.step(closure)
            
            # Check convergence
            if len(cost_history) > 1:
                cost_change = abs(cost_history[-1] - old_cost)
                if cost_change < tol:
                    print(f"Converged after {iter_count + 1} iterations")
                    break
        
        # Get final analysis
        xa_latent = x0_var.detach().cpu().numpy()
        
        # Generate analyzed trajectory
        trajectory = self.dmd.predict(x0=xa_latent, n_steps=max_time)
        
        return {
            'background': xb_latent,
            'analysis': xa_latent,
            'trajectory': trajectory,
            'cost': cost_history[-1] if cost_history else 0.0,
            'cost_history': cost_history,
            'convergence_info': {
                'iterations': len(cost_history),
                'converged': len(cost_history) < max_iter,
                'final_gradient_norm': float(torch.norm(x0_var.grad).item()) if x0_var.grad is not None else 0.0
            }
        }
    
    def _linearize_decoder(self, x_latent: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
        """
        Linearize decoder around a point using finite differences
        
        Args:
            x_latent: Point in latent space to linearize around
            epsilon: Finite difference step size
        
        Returns:
            Jacobian matrix of decoder at x_latent
        """
        if self.decoder is None:
            raise ValueError("Decoder not available")
        
        x_torch = torch.tensor(x_latent, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            # Reference output
            y_ref = self.decoder(x_torch.unsqueeze(0)).squeeze(0)
            n_output = len(y_ref)
            n_input = len(x_latent)
            
            # Compute Jacobian
            jacobian = np.zeros((n_output, n_input))
            
            for i in range(n_input):
                # Perturb input
                x_perturb = x_torch.clone()
                x_perturb[i] += epsilon
                
                # Compute finite difference
                y_perturb = self.decoder(x_perturb.unsqueeze(0)).squeeze(0)
                jacobian[:, i] = (y_perturb - y_ref).cpu().numpy() / epsilon
        
        return jacobian
    
    def evaluate_forecast(self, x0: np.ndarray, true_trajectory: np.ndarray,
                         n_steps: int) -> Dict[str, Any]:
        """
        Evaluate forecast quality against true trajectory
        
        Args:
            x0: Initial condition in physical space
            true_trajectory: True trajectory in physical space [n_features, n_steps]
            n_steps: Number of steps to forecast
        
        Returns:
            Dictionary with forecast and error metrics
        """
        if not self._dmd_fitted:
            raise RuntimeError("DMD model must be fitted before evaluation")
        
        # Convert to latent space if needed
        if self._use_latent and self.encoder is not None:
            x0_torch = torch.tensor(x0, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                x0_latent = self.encoder(x0_torch.unsqueeze(0)).squeeze(0).cpu().numpy()
        else:
            x0_latent = x0
        
        # Forecast in latent space
        forecast_latent = self.dmd.predict(x0=x0_latent, n_steps=n_steps)
        
        # Convert back to physical space if needed
        if self._use_latent and self.decoder is not None:
            forecast_torch = torch.tensor(forecast_latent.T, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                forecast_physical = []
                for t in range(forecast_torch.shape[0]):
                    state_physical = self.decoder(forecast_torch[t].unsqueeze(0)).squeeze(0).cpu().numpy()
                    forecast_physical.append(state_physical)
                forecast = np.column_stack(forecast_physical)
        else:
            forecast = forecast_latent
        
        # Compute errors
        errors = self.dmd.compute_error(true_trajectory[:, :n_steps], forecast)
        
        return {
            'forecast': forecast,
            'errors': errors
        }


# Example usage
if __name__ == "__main__":
    # This is just a placeholder example
    # In practice, encoder/decoder would be loaded from trained models
    
    # Generate synthetic data
    t = np.linspace(0, 10, 100)
    x = np.vstack([np.sin(t), np.cos(t)])  # [2 features, 100 time steps]
    
    # Prepare training data
    X_train = x[:, :-1].T  # [99 samples, 2 features]
    Y_train = x[:, 1:].T   # [99 samples, 2 features]
    
    # Initialize DMD+4DVAR (without encoder/decoder for this example)
    dmd4dvar = DMD4DVAR(dmd_params={'svd_rank': 2})
    
    # Train DMD
    dmd4dvar.train_dmd(X_train, Y_train, use_latent=False)
    
    # Set error covariances
    B = np.eye(2) * 0.1  # Background error
    R = np.eye(1) * 0.01  # Observation error (1D observations)
    dmd4dvar.set_error_covariances(B, R)
    
    # Create synthetic observations
    true_x0 = x[:, 50]  # True initial condition
    H = np.array([[1.0, 0.0]])  # Observe only first component
    
    observations = {
        1: H @ x[:, 51] + np.random.normal(0, 0.1, 1),
        3: H @ x[:, 53] + np.random.normal(0, 0.1, 1),
        5: H @ x[:, 55] + np.random.normal(0, 0.1, 1)
    }
    
    H_operators = {1: H, 3: H, 5: H}
    
    # Perform assimilation
    result_kalman = dmd4dvar.assimilate(
        x0=true_x0,
        observations=observations,
        H_operators=H_operators,
        method='kalman',
        window_size=2
    )
    
    print("\nKalman Filter Results:")
    print(f"Analysis: {result_kalman['analysis']}")
    print(f"Final cost: {result_kalman['cost']:.6f}")
    
    # Try variational method
    result_var = dmd4dvar.assimilate(
        x0=true_x0,
        observations=observations,
        H_operators=H_operators,
        method='var',
        window_size=3,
        max_iter=50
    )
    
    print("\nVariational Results:")
    print(f"Analysis: {result_var['analysis']}")
    print(f"Final cost: {result_var['cost']:.6f}")
    print(f"Converged: {result_var['convergence_info']['converged']}")
    print(f"Iterations: {result_var['convergence_info']['iterations']}")