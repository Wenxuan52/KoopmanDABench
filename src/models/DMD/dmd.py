import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any
from scipy.linalg import svd, pinv


class DMD:
    """
    Dynamic Mode Decomposition (DMD) implementation
    
    DMD extracts dynamic modes from sequential data by finding the best-fit 
    linear operator that advances the system state forward in time.
    """
    
    def __init__(self, svd_rank: Optional[int] = None, tlsq_rank: Optional[int] = None,
                 exact: bool = True, opt: bool = False, device: str = 'cpu'):
        """
        Initialize DMD model
        
        Args:
            svd_rank: Truncation rank for SVD. If None, no truncation
            tlsq_rank: Rank for total least squares DMD. If None, standard DMD
            exact: If True, use exact DMD; if False, use projected DMD
            opt: If True, use optimized DMD (not implemented yet)
            device: Device for computation ('cpu' or 'cuda')
        """
        self.svd_rank = svd_rank
        self.tlsq_rank = tlsq_rank
        self.exact = exact
        self.opt = opt
        self.device = device
        
        # Model components (to be computed during fit)
        self.A_tilde = None  # Reduced dynamics matrix
        self.Phi = None      # DMD modes
        self.Lambda = None   # DMD eigenvalues
        self.omega = None    # Continuous-time eigenvalues
        self.b = None        # Mode amplitudes
        
        # Data statistics
        self.mean = None
        self.std = None
        
        # SVD components
        self.U = None
        self.S = None
        self.V = None
        
        self._fitted = False
    
    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> 'DMD':
        """
        Fit DMD model to training data
        
        Args:
            X_train: Input data matrix [n_samples, n_features] - states at time t
            Y_train: Target data matrix [n_samples, n_features] - states at time t+1
        
        Returns:
            Self
        """
        # Convert to numpy if needed
        if isinstance(X_train, torch.Tensor):
            X_train = X_train.detach().cpu().numpy()
        if isinstance(Y_train, torch.Tensor):
            Y_train = Y_train.detach().cpu().numpy()
        
        # Validate input dimensions
        if X_train.ndim != 2:
            raise ValueError(f"Expected 2D X_train, got {X_train.ndim}D array with shape {X_train.shape}")
        if Y_train.ndim != 2:
            raise ValueError(f"Expected 2D Y_train, got {Y_train.ndim}D array with shape {Y_train.shape}")
        
        # Check consistency between X_train and Y_train
        if X_train.shape != Y_train.shape:
            raise ValueError(f"X_train and Y_train must have same shape: X_train={X_train.shape}, Y_train={Y_train.shape}")
        
        n_samples, n_features = X_train.shape
        print(f"Training DMD with {n_samples} samples and {n_features} features")
        
        # Transpose to match DMD convention: [n_features, n_time_steps]
        # Each column is a state vector at one time step
        X1 = X_train.T  # [n_features, n_samples]
        X2 = Y_train.T  # [n_features, n_samples]
        
        # Store data statistics for normalization
        self.mean = np.mean(X1, axis=1, keepdims=True)  # [n_features, 1]
        self.std = np.std(X1, axis=1, keepdims=True)    # [n_features, 1]
        self.std[self.std < 1e-8] = 1.0  # Avoid division by zero
        
        # Store the feature dimension
        self.n_features = n_features
        self.n_samples = n_samples
        
        print(f"Data statistics - Mean range: [{self.mean.min():.6f}, {self.mean.max():.6f}]")
        print(f"Data statistics - Std range: [{self.std.min():.6f}, {self.std.max():.6f}]")
        
        # Perform DMD decomposition
        if self.tlsq_rank is not None:
            print(f"Using Total Least Squares DMD with rank {self.tlsq_rank}")
            self._fit_tlsq(X1, X2)
        else:
            print("Using standard DMD")
            self._fit_standard(X1, X2)
        
        # Validate decomposition results
        if hasattr(self, 'Phi') and self.Phi is not None:
            print(f"DMD fit complete. Phi shape: {self.Phi.shape}")
            print(f"Reconstruction rank: {self.Phi.shape[1]}")
        else:
            print("Warning: DMD decomposition may have failed - Phi not found")
        
        # Store training data shapes for reference
        self._X_train_shape = X_train.shape
        self._Y_train_shape = Y_train.shape
        
        self._fitted = True
        return self
    
    def _fit_standard(self, X1: np.ndarray, X2: np.ndarray):
        """Standard DMD algorithm"""
        # Step 1: SVD of X1
        U, S, Vh = svd(X1, full_matrices=False)
        V = Vh.T.conj()
        
        # Truncate if needed
        if self.svd_rank is not None:
            U = U[:, :self.svd_rank]
            S = S[:self.svd_rank]
            V = V[:, :self.svd_rank]
        
        self.U = U
        self.S = S
        self.V = V
        
        # Step 2: Build reduced matrix A_tilde
        self.A_tilde = U.T.conj() @ X2 @ V @ np.diag(1/S)
        
        # Step 3: Eigendecomposition of A_tilde
        eigvals, eigvecs = np.linalg.eig(self.A_tilde)
        
        # Sort by magnitude
        idx = np.argsort(np.abs(eigvals))[::-1]
        self.Lambda = eigvals[idx]
        W = eigvecs[:, idx]
        
        # Step 4: Compute DMD modes
        if self.exact:
            # Exact DMD modes
            self.Phi = X2 @ V @ np.diag(1/S) @ W
        else:
            # Projected DMD modes
            self.Phi = U @ W
        
        # Step 5: Compute mode amplitudes
        # Use least squares to find initial amplitudes
        self.b = np.linalg.lstsq(self.Phi, X1[:, 0], rcond=None)[0]
        
        # Compute continuous-time eigenvalues (for time scaling)
        self.omega = np.log(self.Lambda)
    
    def _fit_tlsq(self, X1: np.ndarray, X2: np.ndarray):
        """Total Least Squares DMD"""
        # Stack data matrices
        Z = np.vstack([X1, X2])
        
        # SVD of stacked matrix
        U, S, Vh = svd(Z, full_matrices=False)
        V = Vh.T.conj()
        
        # Truncate
        if self.tlsq_rank is not None:
            U = U[:, :self.tlsq_rank]
            S = S[:self.tlsq_rank]
            V = V[:, :self.tlsq_rank]
        
        # Split U matrix
        n = X1.shape[0]
        U1 = U[:n, :]
        U2 = U[n:, :]
        
        # Compute A_tilde
        self.A_tilde = U2 @ U1.T.conj()
        
        # Continue with standard DMD steps
        eigvals, eigvecs = np.linalg.eig(self.A_tilde)
        
        # Sort by magnitude
        idx = np.argsort(np.abs(eigvals))[::-1]
        self.Lambda = eigvals[idx]
        W = eigvecs[:, idx]
        
        # Compute modes
        self.Phi = U2 @ W
        
        # Compute amplitudes
        self.b = np.linalg.lstsq(self.Phi, X1[:, 0], rcond=None)[0]
        self.omega = np.log(self.Lambda)
        
        # Store for reconstruction
        self.U = U1
        self.S = S
        self.V = V
    
    def predict(self, x0: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Predict future states using DMD starting from initial condition
        
        Args:
            x0: Initial condition [n_features] - starting state vector
            n_steps: Number of time steps to predict
        
        Returns:
            Predicted states [n_features, n_steps] - each column is a predicted state
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Convert to numpy if needed
        if isinstance(x0, torch.Tensor):
            x0 = x0.detach().cpu().numpy()
        
        # Ensure x0 is 1D array
        x0 = np.asarray(x0).flatten()
        
        # Validate initial condition dimensions
        if x0.shape[0] != self.n_features:
            raise ValueError(f"Initial condition dimension {x0.shape[0]} doesn't match "
                            f"DMD feature dimension {self.n_features}")
        
        # Validate n_steps
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")
        
        print(f"Predicting {n_steps} steps from initial condition with {x0.shape[0]} features")
        
        # Normalize initial condition using training statistics
        x0_normalized = (x0.reshape(-1, 1) - self.mean) / self.std
        x0_normalized = x0_normalized.flatten()
        
        # Project initial condition onto DMD modes
        # Solve: Phi @ b = x0_normalized for b (mode amplitudes)
        try:
            b, residuals, rank, s = np.linalg.lstsq(self.Phi, x0_normalized, rcond=None)
            
            # Check if projection is reasonable
            if len(residuals) > 0:
                relative_error = np.sqrt(residuals[0]) / np.linalg.norm(x0_normalized)
                print(f"Initial condition projection relative error: {relative_error:.6f}")
                if relative_error > 0.1:
                    print("Warning: Initial condition may not be well-represented by DMD modes")
            
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"Failed to project initial condition onto DMD modes: {e}")
        
        # Generate time dynamics
        times = np.arange(n_steps)
        time_dynamics = np.zeros((len(self.Lambda), n_steps), dtype=complex)
        
        for i, lam in enumerate(self.Lambda):
            # Each mode evolves as: b[i] * lambda[i]^t
            time_dynamics[i, :] = b[i] * (lam ** times)
        
        # Reconstruct states: X_pred = Phi @ time_dynamics
        X_pred_normalized = self.Phi @ time_dynamics
        
        # Denormalize predictions
        X_pred = X_pred_normalized.real * self.std + self.mean
        
        print(f"Prediction complete. Output shape: {X_pred.shape}")
        print(f"Prediction range: [{X_pred.min():.6f}, {X_pred.max():.6f}]")
        
        return X_pred
    
    
    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """
        Reconstruct data by predicting one step forward from each time step
        
        Args:
            X: Input data [n_features, n_time_steps] - each column is a state vector
        
        Returns:
            Reconstructed data [n_features, n_time_steps] - one-step predictions from each input state
            Note: X_reconstructed[:, t] = predict(X[:, t], n_steps=1)[:, 0]
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before reconstruction")
        
        # Convert to numpy if needed
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        
        # Validate input dimensions
        if X.ndim != 2:
            raise ValueError(f"Expected 2D input, got {X.ndim}D array with shape {X.shape}")
        
        if X.shape[0] != self.n_features:
            raise ValueError(f"Feature dimension mismatch: expected {self.n_features}, got {X.shape[0]}")
        
        n_features, n_time_steps = X.shape
        print(f"Reconstructing {n_time_steps} time steps, each with {n_features} features")
        
        # Initialize output array
        X_reconstructed = np.zeros_like(X)
        
        # For each time step, predict one step forward
        for t in range(n_time_steps):
            # Get current state as initial condition
            x0 = X[:, t]  # [n_features]
            
            # Predict one step forward
            x_next = self.predict(x0=x0, n_steps=1)  # [n_features, 1]
            
            # Store the predicted next state
            X_reconstructed[:, t] = x_next[:, 0]  # Take the single predicted step
        
        print(f"Reconstruction complete. Output shape: {X_reconstructed.shape}")
        
        return X_reconstructed
    
    def compute_error(self, X_true: np.ndarray, X_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute reconstruction/prediction errors including SSIM
        
        Args:
            X_true: True data [n_features, n_time_steps]
            X_pred: Predicted/reconstructed data [n_features, n_time_steps]
        
        Returns:
            Dictionary of error metrics including SSIM
        """
        # Validate inputs
        if X_true.ndim != 2 or X_pred.ndim != 2:
            raise ValueError("Both inputs must be 2D arrays [n_features, n_time_steps]")
        
        # Ensure same shape
        min_steps = min(X_true.shape[1], X_pred.shape[1])
        X_true = X_true[:, :min_steps]
        X_pred = X_pred[:, :min_steps]
        
        # Compute basic errors
        mse = np.mean((X_true - X_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(X_true - X_pred))
        
        # Relative errors
        norm_true = np.linalg.norm(X_true, 'fro')
        rel_err = np.linalg.norm(X_true - X_pred, 'fro') / norm_true if norm_true > 0 else np.inf
        
        # Compute SSIM
        ssim_value = self._compute_ssim(X_true, X_pred)
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'relative_error': float(rel_err),
            'ssim': float(ssim_value)
        }

    def _compute_ssim(self, X_true: np.ndarray, X_pred: np.ndarray) -> float:
        """
        Compute Structural Similarity Index (SSIM) between two matrices
        
        Args:
            X_true: True data [n_features, n_time_steps]
            X_pred: Predicted data [n_features, n_time_steps]
        
        Returns:
            SSIM value (higher is better, max = 1.0)
        """
        # Flatten matrices to 1D for SSIM computation
        x_true = X_true.flatten()
        x_pred = X_pred.flatten()
        
        # SSIM parameters (similar to image SSIM)
        # Dynamic range of the data
        data_range = np.max([np.max(x_true) - np.min(x_true), 
                            np.max(x_pred) - np.min(x_pred)])
        
        # Small constants to avoid division by zero
        k1, k2 = 0.01, 0.03
        c1 = (k1 * data_range) ** 2
        c2 = (k2 * data_range) ** 2
        
        # Compute means
        mu_x = np.mean(x_true)
        mu_y = np.mean(x_pred)
        
        # Compute variances and covariance
        var_x = np.var(x_true)
        var_y = np.var(x_pred)
        cov_xy = np.mean((x_true - mu_x) * (x_pred - mu_y))
        
        # SSIM formula
        numerator = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
        denominator = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
        
        ssim = numerator / denominator if denominator > 0 else 0.0
        
        return ssim
    
    def get_mode_dynamics(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get DMD mode information
        
        Returns:
            modes: DMD modes (Phi)
            eigenvalues: DMD eigenvalues (Lambda)
            amplitudes: Mode amplitudes (b)
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted first")
        
        return self.Phi, self.Lambda, self.b
    
    def save(self, filepath: str):
        """Save DMD model to file"""
        model_dict = {
            # Original hyperparameters
            'svd_rank': self.svd_rank,
            'tlsq_rank': self.tlsq_rank,
            'exact': self.exact,
            
            # Core DMD components
            'A_tilde': self.A_tilde,
            'Phi': self.Phi,
            'Lambda': self.Lambda,
            'omega': self.omega,
            'b': self.b,
            
            # Data statistics (for normalization)
            'mean': self.mean,
            'std': self.std,
            
            # SVD components
            'U': self.U,
            'S': self.S,
            'V': self.V,
            
            # Training data information (NEW)
            'n_features': getattr(self, 'n_features', None),
            'n_samples': getattr(self, 'n_samples', None),
            '_X_train_shape': getattr(self, '_X_train_shape', None),
            '_Y_train_shape': getattr(self, '_Y_train_shape', None),
            
            # Model state
            '_fitted': self._fitted
        }
        
        # Remove None values to avoid issues
        model_dict = {k: v for k, v in model_dict.items() if v is not None}
        
        np.savez(filepath, **model_dict)
        print(f"DMD model saved to {filepath}")

    def load(self, filepath: str):
        """Load DMD model from file"""
        data = np.load(filepath, allow_pickle=True)
        
        # Restore hyperparameters
        self.svd_rank = data.get('svd_rank', None)
        self.tlsq_rank = data.get('tlsq_rank', None)
        self.exact = data.get('exact', True)
        
        # Restore core DMD components
        self.A_tilde = data.get('A_tilde', None)
        self.Phi = data.get('Phi', None)
        self.Lambda = data.get('Lambda', None)
        self.omega = data.get('omega', None)
        self.b = data.get('b', None)
        
        # Restore data statistics
        self.mean = data.get('mean', None)
        self.std = data.get('std', None)
        
        # Restore SVD components
        self.U = data.get('U', None)
        self.S = data.get('S', None)
        self.V = data.get('V', None)
        
        # Restore training data information (NEW)
        self.n_features = data.get('n_features', None)
        self.n_samples = data.get('n_samples', None)
        self._X_train_shape = data.get('_X_train_shape', None)
        self._Y_train_shape = data.get('_Y_train_shape', None)
        
        # Restore model state
        self._fitted = data.get('_fitted', False)
        
        data.close()
        print(f"DMD model loaded from {filepath}")
        
        # Validate loaded model
        if self._fitted:
            print(f"Loaded fitted DMD model:")
            if self.n_features is not None:
                print(f"  Features: {self.n_features}")
            if self.n_samples is not None:
                print(f"  Training samples: {self.n_samples}")
            if self.Phi is not None:
                print(f"  DMD modes shape: {self.Phi.shape}")
            if self.Lambda is not None:
                print(f"  Eigenvalues: {len(self.Lambda)}")
        else:
            print("Loaded unfitted DMD model")


# Example usage
if __name__ == "__main__":
    
    # Generate synthetic data
    t = np.linspace(0, 10, 100)
    x = np.vstack([np.sin(t), np.cos(t), np.sin(2*t)])  # [3 features, 100 time steps]
    
    print(f"Original data shape: {x.shape}")
    
    # Prepare training data in the new format [n_samples, n_features]
    # Convert from [n_features, n_time_steps] to time-delay embedded format
    n_features, n_time_steps = x.shape
    
    # Create X_train and Y_train for time-delay embedding
    X_train = x[:, :-1].T  # [99 samples, 3 features] - states at time t
    Y_train = x[:, 1:].T   # [99 samples, 3 features] - states at time t+1
    
    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")
    
    # Create and fit DMD with new interface
    dmd = DMD(svd_rank=3, exact=True)
    dmd.fit(X_train, Y_train)
    
    # Test predict function with initial condition
    x0 = x[:, 0]  # Use first time step as initial condition
    n_steps = 50
    x_pred = dmd.predict(x0=x0, n_steps=n_steps)
    print(f"Prediction shape: {x_pred.shape}")  # Should be [3 features, 50 time steps]
    
    # Test reconstruct function
    x_test = x[:, :20]  # Use first 20 time steps for reconstruction test
    x_reconstructed = dmd.reconstruct(x_test)
    print(f"Reconstruction input shape: {x_test.shape}")
    print(f"Reconstruction output shape: {x_reconstructed.shape}")
    
    # Compute reconstruction errors
    errors_recon = dmd.compute_error(x_test, x_reconstructed)
    print("Reconstruction errors:", errors_recon)
    
    # Compute prediction errors (compare with true continuation)
    if n_steps <= x.shape[1] - 1:  # Make sure we have enough true data
        x_true_continuation = x[:, 1:n_steps+1]  # True states from t=1 to t=n_steps
        errors_pred = dmd.compute_error(x_true_continuation, x_pred)
        print("Prediction errors:", errors_pred)
    
    # Get mode information (if this method exists)
    try:
        modes, eigenvalues, amplitudes = dmd.get_mode_dynamics()
        print(f"Number of modes: {modes.shape[1]}")
        print(f"First 5 eigenvalues: {eigenvalues[:5]}")
        print(f"Eigenvalue magnitudes: {np.abs(eigenvalues[:5])}")
    except AttributeError:
        print("get_mode_dynamics method not available")
        # Alternative: access DMD components directly
        if hasattr(dmd, 'Phi') and dmd.Phi is not None:
            print(f"DMD modes shape: {dmd.Phi.shape}")
        if hasattr(dmd, 'Lambda') and dmd.Lambda is not None:
            print(f"Number of eigenvalues: {len(dmd.Lambda)}")
            print(f"First 5 eigenvalues: {dmd.Lambda[:5]}")
            print(f"Eigenvalue magnitudes: {np.abs(dmd.Lambda[:5])}")
    
    # Test save/load functionality
    try:
        # Save model
        dmd.save("test_dmd_model.npz")
        
        # Create new DMD instance and load
        dmd_loaded = DMD(svd_rank=3, exact=True)
        dmd_loaded.load("test_dmd_model.npz")
        
        # Test loaded model
        x_pred_loaded = dmd_loaded.predict(x0=x0, n_steps=10)
        print(f"Loaded model prediction shape: {x_pred_loaded.shape}")
        
        # Compare predictions
        x_pred_original = dmd.predict(x0=x0, n_steps=10)
        diff = np.max(np.abs(x_pred_original - x_pred_loaded))
        print(f"Max difference between original and loaded model: {diff}")
        
    except Exception as e:
        print(f"Save/load test failed: {e}")
    
    print("\nDMD test completed successfully!")