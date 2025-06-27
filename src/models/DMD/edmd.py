import numpy as np
import torch
from typing import Callable, List, Optional, Dict, Any, Tuple
from scipy.linalg import svd, pinv
from dmd import DMD


class EDMD(DMD):
    """
    Extended Dynamic Mode Decomposition (EDMD) implementation
    
    EDMD extends DMD by applying dictionary functions to lift the state space
    into a higher-dimensional observable space before applying DMD.
    """
    
    def __init__(self,
                 dictionary_funcs: List[Callable[[np.ndarray], np.ndarray]],
                 svd_rank: Optional[int] = None,
                 tlsq_rank: Optional[int] = None,
                 exact: bool = True,
                 opt: bool = False,
                 device: str = 'cpu'):
        """
        Initialize EDMD model
        
        Args:
            dictionary_funcs: List of dictionary functions to apply to state vectors
                             Each function should take a 1D array of shape [n_features] 
                             and return a 1D array of observables
            svd_rank: Truncation rank for SVD. If None, no truncation
            tlsq_rank: Rank for total least squares DMD. If None, standard DMD
            exact: If True, use exact DMD; if False, use projected DMD
            opt: If True, use optimized DMD (not implemented yet)
            device: Device for computation ('cpu' or 'cuda')
        """
        # Initialize parent DMD class
        super().__init__(svd_rank=svd_rank, tlsq_rank=tlsq_rank, 
                        exact=exact, opt=opt, device=device)
        
        # EDMD-specific parameters
        self.dictionary_funcs = dictionary_funcs
        self.n_observables = None  # Will be determined after applying dictionary functions
        
        # Store lifted data statistics
        self.lifted_mean = None
        self.lifted_std = None
        
        print(f"Initialized EDMD with {len(dictionary_funcs)} dictionary functions")
    
    def _apply_dictionary(self, X: np.ndarray) -> np.ndarray:
        """
        Apply dictionary functions to transform state vectors into observables
        
        Args:
            X: State data [n_features, n_time_steps] - each column is a state vector
        
        Returns:
            Transformed data [n_observables, n_time_steps] - lifted to observable space
        """
        if X.ndim != 2:
            raise ValueError(f"Expected 2D input, got {X.ndim}D array with shape {X.shape}")
        
        n_features, n_time_steps = X.shape
        
        # Apply dictionary functions to each time step
        lifted_data = []
        
        for t in range(n_time_steps):
            state_vector = X[:, t]  # [n_features]
            
            # Apply each dictionary function
            observables = []
            for func in self.dictionary_funcs:
                obs = func(state_vector)
                # Ensure output is 1D array
                obs = np.asarray(obs).flatten()
                observables.append(obs)
            
            # Concatenate all observables for this time step
            full_observables = np.concatenate(observables)  # [n_observables]
            lifted_data.append(full_observables)
        
        # Stack time steps as columns
        lifted_X = np.column_stack(lifted_data)  # [n_observables, n_time_steps]
        
        # Store observable dimension on first call
        if self.n_observables is None:
            self.n_observables = lifted_X.shape[0]
            print(f"Observable space dimension: {self.n_observables}")
        
        return lifted_X
    
    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> 'EDMD':
        """
        Fit EDMD model to training data
        
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
        print(f"Training EDMD with {n_samples} samples and {n_features} features")
        
        # Transpose to match DMD convention: [n_features, n_time_steps]
        X1 = X_train.T  # [n_features, n_samples]
        X2 = Y_train.T  # [n_features, n_samples]
        
        # Apply dictionary functions to lift to observable space
        print("Applying dictionary functions...")
        Psi1 = self._apply_dictionary(X1)  # [n_observables, n_samples]
        Psi2 = self._apply_dictionary(X2)  # [n_observables, n_samples]
        
        print(f"Lifted data shape: {Psi1.shape} -> {Psi2.shape}")
        
        # Store statistics for the LIFTED data (not original data)
        self.lifted_mean = np.mean(Psi1, axis=1, keepdims=True)  # [n_observables, 1]
        self.lifted_std = np.std(Psi1, axis=1, keepdims=True)    # [n_observables, 1]
        self.lifted_std[self.lifted_std < 1e-8] = 1.0  # Avoid division by zero
        
        # Also store original data statistics for prediction interface
        self.mean = np.mean(X1, axis=1, keepdims=True)  # [n_features, 1]
        self.std = np.std(X1, axis=1, keepdims=True)    # [n_features, 1]
        self.std[self.std < 1e-8] = 1.0
        
        # Store dimensions
        self.n_features = n_features
        self.n_samples = n_samples
        
        print(f"Lifted data statistics - Mean range: [{self.lifted_mean.min():.6f}, {self.lifted_mean.max():.6f}]")
        print(f"Lifted data statistics - Std range: [{self.lifted_std.min():.6f}, {self.lifted_std.max():.6f}]")
        
        # Apply DMD in the lifted observable space
        if self.tlsq_rank is not None:
            print(f"Using Total Least Squares EDMD with rank {self.tlsq_rank}")
            self._fit_tlsq(Psi1, Psi2)
        else:
            print("Using standard EDMD")
            self._fit_standard(Psi1, Psi2)
        
        # Validate decomposition results
        if hasattr(self, 'Phi') and self.Phi is not None:
            print(f"EDMD fit complete. Phi shape: {self.Phi.shape}")
            print(f"Reconstruction rank: {self.Phi.shape[1]}")
        else:
            print("Warning: EDMD decomposition may have failed - Phi not found")
        
        # Store training data shapes for reference
        self._X_train_shape = X_train.shape
        self._Y_train_shape = Y_train.shape
        
        self._fitted = True
        return self
    
    def predict(self, x0: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Predict future states using EDMD starting from initial condition
        
        Args:
            x0: Initial condition [n_features] - starting state vector in original space
            n_steps: Number of time steps to predict
        
        Returns:
            Predicted states [n_features, n_steps] - each column is a predicted state in original space
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
                            f"EDMD feature dimension {self.n_features}")
        
        # Validate n_steps
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")
        
        print(f"Predicting {n_steps} steps from initial condition with {x0.shape[0]} features")
        
        # For EDMD, we need to:
        # 1. Lift initial condition to observable space
        # 2. Evolve in observable space using DMD dynamics
        # 3. Project back to original state space (if possible) or return observables
        
        # Lift initial condition to observable space
        psi0 = self._apply_dictionary(x0.reshape(-1, 1))  # [n_observables, 1]
        psi0 = psi0.flatten()  # [n_observables]
        
        # Normalize in lifted space
        psi0_normalized = (psi0.reshape(-1, 1) - self.lifted_mean) / self.lifted_std
        psi0_normalized = psi0_normalized.flatten()
        
        # Project initial condition onto DMD modes in observable space
        try:
            b, residuals, rank, s = np.linalg.lstsq(self.Phi, psi0_normalized, rcond=None)
            
            # Check if projection is reasonable
            if len(residuals) > 0:
                relative_error = np.sqrt(residuals[0]) / np.linalg.norm(psi0_normalized)
                print(f"Initial condition projection relative error: {relative_error:.6f}")
                if relative_error > 0.1:
                    print("Warning: Initial condition may not be well-represented by EDMD modes")
            
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"Failed to project initial condition onto EDMD modes: {e}")
        
        # Generate time dynamics in observable space
        times = np.arange(n_steps)
        time_dynamics = np.zeros((len(self.Lambda), n_steps), dtype=complex)
        
        for i, lam in enumerate(self.Lambda):
            time_dynamics[i, :] = b[i] * (lam ** times)
        
        # Reconstruct observables: Psi_pred = Phi @ time_dynamics
        Psi_pred_normalized = self.Phi @ time_dynamics
        
        # Denormalize predictions in observable space
        Psi_pred = Psi_pred_normalized.real * self.lifted_std + self.lifted_mean
        
        # WARNING: For full EDMD, we would need an inverse mapping from observables back to states
        # This is generally not possible unless we have specific structure
        # For now, we return the observables as the "prediction"
        # In practice, you might want to:
        # 1. Use only the first n_features observables if they correspond to the original states
        # 2. Learn an inverse mapping
        # 3. Use optimization to find states that produce the predicted observables
        
        # Simple heuristic: if first n_features observables are identity functions, use them
        if (self.n_observables >= self.n_features and 
            len(self.dictionary_funcs) > 0):
            
            # Check if first dictionary function might be identity-like
            # This is a heuristic - in practice you'd design this more carefully
            try:
                # Try to extract state-like observables (first n_features components)
                X_pred = Psi_pred[:self.n_features, :]
                
                print(f"EDMD prediction complete. Extracted state prediction shape: {X_pred.shape}")
                print(f"State prediction range: [{X_pred.min():.6f}, {X_pred.max():.6f}]")
                
                return X_pred
                
            except Exception as e:
                print(f"Could not extract state prediction: {e}")
                print("Returning full observable prediction")
                
        # Fallback: return full observable space prediction
        print(f"EDMD prediction complete. Observable prediction shape: {Psi_pred.shape}")
        print(f"Observable prediction range: [{Psi_pred.min():.6f}, {Psi_pred.max():.6f}]")
        print("Note: Returning observables - implement inverse mapping for state space predictions")
        
        return Psi_pred
    
    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """
        Reconstruct data by predicting one step forward from each time step in observable space
        
        Args:
            X: Input data [n_features, n_time_steps] - each column is a state vector in original space
        
        Returns:
            Reconstructed data [n_features or n_observables, n_time_steps] 
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
        
        # Lift to observable space
        Psi = self._apply_dictionary(X)  # [n_observables, n_time_steps]
        
        # Initialize output in observable space
        Psi_reconstructed = np.zeros_like(Psi)
        
        # For each time step, predict one step forward in observable space
        for t in range(n_time_steps):
            # Get current observables
            psi_t = Psi[:, t]  # [n_observables]
            
            # Normalize
            psi_t_normalized = (psi_t.reshape(-1, 1) - self.lifted_mean) / self.lifted_std
            psi_t_normalized = psi_t_normalized.flatten()
            
            # Project onto DMD modes
            b_t = np.linalg.lstsq(self.Phi, psi_t_normalized, rcond=None)[0]
            
            # Evolve one step: Psi(t+1) = Phi @ Lambda @ b_t
            psi_next_normalized = self.Phi @ (self.Lambda * b_t)
            
            # Denormalize
            psi_next = (psi_next_normalized.real.reshape(-1, 1) * self.lifted_std + 
                       self.lifted_mean).flatten()
            
            Psi_reconstructed[:, t] = psi_next
        
        # Try to extract state space reconstruction if possible
        if self.n_observables >= self.n_features:
            X_reconstructed = Psi_reconstructed[:self.n_features, :]
            print(f"State space reconstruction complete. Output shape: {X_reconstructed.shape}")
            return X_reconstructed
        else:
            print(f"Observable space reconstruction complete. Output shape: {Psi_reconstructed.shape}")
            print("Note: Returning observables - implement inverse mapping for state space reconstruction")
            return Psi_reconstructed
    
    def save(self, filepath: str):
        """Save EDMD model to file"""
        # Get base DMD components
        model_dict = {
            # Original hyperparameters
            'svd_rank': self.svd_rank,
            'tlsq_rank': self.tlsq_rank,
            'exact': self.exact,
            
            # Core DMD components (these work in observable space for EDMD)
            'A_tilde': self.A_tilde,
            'Phi': self.Phi,
            'Lambda': self.Lambda,
            'omega': self.omega,
            'b': self.b,
            
            # Original data statistics
            'mean': self.mean,
            'std': self.std,
            
            # EDMD-specific: Lifted data statistics
            'lifted_mean': self.lifted_mean,
            'lifted_std': self.lifted_std,
            
            # SVD components
            'U': self.U,
            'S': self.S,
            'V': self.V,
            
            # Dimensions
            'n_features': getattr(self, 'n_features', None),
            'n_samples': getattr(self, 'n_samples', None),
            'n_observables': getattr(self, 'n_observables', None),
            '_X_train_shape': getattr(self, '_X_train_shape', None),
            '_Y_train_shape': getattr(self, '_Y_train_shape', None),
            
            # Model state
            '_fitted': self._fitted,
            
            # EDMD identifier
            'model_type': 'EDMD'
        }
        
        # Remove None values
        model_dict = {k: v for k, v in model_dict.items() if v is not None}
        
        np.savez(filepath, **model_dict)
        print(f"EDMD model saved to {filepath}")
        print("Note: Dictionary functions are not saved - you must provide them when loading")
    
    def load(self, filepath: str, dictionary_funcs: List[Callable[[np.ndarray], np.ndarray]]):
        """
        Load EDMD model from file
        
        Args:
            filepath: Path to saved model
            dictionary_funcs: List of dictionary functions (must match those used during training)
        """
        data = np.load(filepath, allow_pickle=True)
        
        # Verify this is an EDMD model
        if data.get('model_type', None) != 'EDMD':
            print("Warning: Loading non-EDMD model file into EDMD class")
        
        # Restore dictionary functions
        self.dictionary_funcs = dictionary_funcs
        
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
        self.lifted_mean = data.get('lifted_mean', None)
        self.lifted_std = data.get('lifted_std', None)
        
        # Restore SVD components
        self.U = data.get('U', None)
        self.S = data.get('S', None)  
        self.V = data.get('V', None)
        
        # Restore dimensions
        self.n_features = data.get('n_features', None)
        self.n_samples = data.get('n_samples', None)
        self.n_observables = data.get('n_observables', None)
        self._X_train_shape = data.get('_X_train_shape', None)
        self._Y_train_shape = data.get('_Y_train_shape', None)
        
        # Restore model state
        self._fitted = data.get('_fitted', False)
        
        data.close()
        print(f"EDMD model loaded from {filepath}")
        
        # Validate loaded model
        if self._fitted:
            print(f"Loaded fitted EDMD model:")
            if self.n_features is not None:
                print(f"  Original features: {self.n_features}")
            if self.n_observables is not None:
                print(f"  Observable features: {self.n_observables}")
            if self.n_samples is not None:
                print(f"  Training samples: {self.n_samples}")
            if self.Phi is not None:
                print(f"  DMD modes shape: {self.Phi.shape}")
            if self.Lambda is not None:
                print(f"  Eigenvalues: {len(self.Lambda)}")
            print(f"  Dictionary functions: {len(self.dictionary_funcs)}")
        else:
            print("Loaded unfitted EDMD model")


# Example dictionary functions that can be used with EDMD
def create_polynomial_basis(degree: int = 2):
    """
    Create polynomial basis functions up to given degree
    
    Args:
        degree: Maximum polynomial degree
    
    Returns:
        List of polynomial basis functions
    """
    def identity(x):
        """Identity function (degree 1)"""
        return x.copy()
    
    def quadratic_monomials(x):
        """All quadratic monomials x_i * x_j"""
        n = len(x)
        result = []
        for i in range(n):
            for j in range(i, n):  # j >= i to avoid duplicates
                result.append(x[i] * x[j])
        return np.array(result)
    
    def cubic_monomials(x):
        """Selected cubic monomials (x_i^3 for simplicity)"""
        return x**3
    
    basis_funcs = [identity]
    
    if degree >= 2:
        basis_funcs.append(quadratic_monomials)
    
    if degree >= 3:
        basis_funcs.append(cubic_monomials)
    
    return basis_funcs


def create_rbf_basis(centers: np.ndarray, sigma: float = 1.0):
    """
    Create Radial Basis Function (RBF) dictionary
    
    Args:
        centers: Centers for RBF functions [n_centers, n_features]
        sigma: RBF width parameter
    
    Returns:
        List containing RBF basis functions
    """
    def rbf_functions(x):
        """RBF evaluations at all centers"""
        # x is [n_features], centers is [n_centers, n_features]
        diffs = centers - x.reshape(1, -1)  # [n_centers, n_features]
        distances_sq = np.sum(diffs**2, axis=1)  # [n_centers]
        return np.exp(-distances_sq / (2 * sigma**2))
    
    return [rbf_functions]


def create_fourier_basis(frequencies: np.ndarray):
    """
    Create Fourier basis functions
    
    Args:
        frequencies: Frequencies for sine/cosine functions [n_freq]
    
    Returns:
        List of Fourier basis functions
    """
    def fourier_functions(x):
        """Fourier features: sin and cos at various frequencies"""
        result = []
        for freq in frequencies:
            # Apply frequency to all state components
            for xi in x:
                result.append(np.sin(freq * xi))
                result.append(np.cos(freq * xi))
        return np.array(result)
    
    return [fourier_functions]


# Example usage
if __name__ == "__main__":
    
    # Generate synthetic nonlinear data
    t = np.linspace(0, 10, 200)
    # Nonlinear system: Duffing oscillator-like
    x1 = np.sin(t) * np.exp(-0.1 * t)
    x2 = np.cos(t) * np.exp(-0.1 * t)  
    x3 = 0.1 * (x1**2 + x2**2)  # Nonlinear coupling
    
    x = np.vstack([x1, x2, x3])  # [3 features, 200 time steps]
    
    print(f"Original nonlinear data shape: {x.shape}")
    
    # Prepare training data
    n_features, n_time_steps = x.shape
    X_train = x[:, :-1].T  # [199 samples, 3 features]
    Y_train = x[:, 1:].T   # [199 samples, 3 features]
    
    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")
    
    # Create dictionary functions
    # Option 1: Polynomial basis
    poly_basis = create_polynomial_basis(degree=2)
    
    # Option 2: RBF basis (example)
    # centers = np.random.randn(5, 3)  # 5 RBF centers in 3D space
    # rbf_basis = create_rbf_basis(centers, sigma=1.0)
    
    # Option 3: Combined basis
    combined_basis = poly_basis  # Start with polynomial
    
    print(f"Using {len(combined_basis)} dictionary functions")
    
    # Test dictionary functions
    test_state = X_train[0]  # First state vector
    print(f"Test state: {test_state}")
    
    for i, func in enumerate(combined_basis):
        obs = func(test_state)
        print(f"Dictionary function {i} output shape: {obs.shape}")
        print(f"Dictionary function {i} output sample: {obs[:5]}")  # First 5 elements
    
    # Create and fit EDMD
    edmd = EDMD(dictionary_funcs=combined_basis, svd_rank=10, exact=True)
    edmd.fit(X_train, Y_train)
    
    # Test prediction
    x0 = x[:, 0]  # Initial condition
    n_steps = 50
    
    print(f"\nTesting prediction from initial condition: {x0}")
    x_pred = edmd.predict(x0=x0, n_steps=n_steps)
    print(f"Prediction shape: {x_pred.shape}")
    
    # Test reconstruction
    x_test = x[:, :20]  # First 20 time steps
    x_reconstructed = edmd.reconstruct(x_test)
    print(f"Reconstruction input shape: {x_test.shape}")
    print(f"Reconstruction output shape: {x_reconstructed.shape}")
    
    # Compute errors if dimensions match
    if x_reconstructed.shape == x_test.shape:
        errors_recon = edmd.compute_error(x_test, x_reconstructed)
        print("Reconstruction errors:", errors_recon)
    else:
        print("Cannot compute reconstruction errors - dimension mismatch")
    
    # Test save/load
    try:
        # Save model
        edmd.save("test_edmd_model.npz")
        
        # Create new EDMD instance and load
        edmd_loaded = EDMD(dictionary_funcs=combined_basis, svd_rank=10, exact=True)
        edmd_loaded.load("test_edmd_model.npz", dictionary_funcs=combined_basis)
        
        # Test loaded model
        x_pred_loaded = edmd_loaded.predict(x0=x0, n_steps=10)
        print(f"Loaded model prediction shape: {x_pred_loaded.shape}")
        
        # Compare predictions if shapes match
        x_pred_original = edmd.predict(x0=x0, n_steps=10)
        if x_pred_original.shape == x_pred_loaded.shape:
            diff = np.max(np.abs(x_pred_original - x_pred_loaded))
            print(f"Max difference between original and loaded model: {diff}")
        else:
            print("Cannot compare predictions - shape mismatch")
        
    except Exception as e:
        print(f"Save/load test failed: {e}")
    
    print("\nEDMD test completed!")
