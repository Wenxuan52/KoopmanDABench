"""
TorchDMD: GPU-accelerated Dynamic Mode Decomposition using PyTorch
"""

import torch
import numpy as np


class TorchDMD:
    """
    Dynamic Mode Decomposition implemented with PyTorch for GPU acceleration.
    
    Parameters
    ----------
    svd_rank : int or float, optional
        Truncation rank for SVD. If 0, optimal rank is computed.
        If -1, no truncation. If float in (0,1), rank is determined
        by the cumulative energy. Default is 0.
    device : str, optional
        Device to run computations on ('cuda' or 'cpu'). Default is 'cuda'.
    """
    
    def __init__(self, svd_rank=0, device='cuda'):
        self.svd_rank = svd_rank
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # DMD matrices and results
        self.modes = None
        self.eigenvalues = None
        self.amplitudes = None
        self._Atilde = None
        
        # Original data info
        self.original_time = {'t0': 0, 'tend': 0, 'dt': 1}
        self.n_snapshots = 0
        
    def fit(self, X, Y=None):
        """
        Fit the DMD model to the data.
        
        Parameters
        ----------
        X : array-like, shape (n_features, n_snapshots)
            Input snapshots matrix
        Y : array-like, shape (n_features, n_snapshots), optional
            Target snapshots matrix. If None, uses X[:, 1:] and X[:, :-1]
            
        Returns
        -------
        self : TorchDMD
            Fitted DMD model
        """
        # Convert to torch tensors and move to device
        if isinstance(X, torch.Tensor):
            X = X.to(dtype=torch.float32, device=self.device)
        else:
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        if Y is None:
            X_data = X[:, :-1]
            Y_data = X[:, 1:]
            self.n_snapshots = X.shape[1]
        else:
            if isinstance(Y, torch.Tensor):
                Y = Y.to(dtype=torch.float32, device=self.device)
            else:
                Y = torch.tensor(Y, dtype=torch.float32, device=self.device)
            X_data = X
            Y_data = Y
            self.n_snapshots = X.shape[1]
            
        # Compute SVD of X_data
        U, s, Vh = torch.linalg.svd(X_data, full_matrices=False)
        V = Vh.T
        
        # Determine truncation rank
        r = self._compute_rank(s)
        U_r = U[:, :r]
        s_r = s[:r]
        V_r = V[:, :r]
        
        # Compute Atilde (low-rank approximation of A)
        self._Atilde = U_r.T @ Y_data @ V_r @ torch.diag(1.0 / s_r)
        
        # Compute eigendecomposition of Atilde
        eigenvalues, W = torch.linalg.eig(self._Atilde)
        self.eigenvalues = eigenvalues
        
        # Compute DMD modes
        # Convert to complex for compatibility with eigenvectors
        Y_complex = Y_data.to(torch.complex64)
        V_complex = V_r.to(torch.complex64)
        s_inv_complex = torch.diag(1.0 / s_r).to(torch.complex64)
        
        self.modes = Y_complex @ V_complex @ s_inv_complex @ W
        
        # Compute amplitudes using least squares
        # b = Phi^+ * x0 where Phi^+ is pseudoinverse of modes
        x0 = X_data[:, 0].to(torch.complex64)
        self.amplitudes = torch.linalg.lstsq(self.modes, x0).solution
        
        # Update time info
        self.original_time['tend'] = self.n_snapshots - 1
        
        return self
        
    def predict(self, X):
        """
        Predict one timestep ahead using the DMD model.
        
        Parameters
        ----------
        X : array-like, shape (n_features,) or (n_features, n_samples)
            Input state vector(s)
            
        Returns
        -------
        Y : torch.Tensor
            Predicted state at next timestep
        """
        if self.modes is None:
            raise RuntimeError("Model must be fitted before prediction")
            
        # Convert to tensor if needed
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(self.device)
        
        # Ensure X is 2D
        if X.dim() == 1:
            X = X.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False
            
        # Project onto DMD modes
        X_complex = X.to(torch.complex64)
        b = torch.linalg.lstsq(self.modes, X_complex).solution
        
        # Evolve one timestep
        b_next = torch.diag(self.eigenvalues) @ b
        
        # Reconstruct
        Y = self.modes @ b_next
        
        # Return real part
        Y_real = Y.real
        
        return Y_real.squeeze() if squeeze_output else Y_real
    
    def predict_sequence(self, x0, time_steps):
        """
        Predict multiple timesteps ahead using the DMD model.
        
        Parameters
        ----------
        x0 : array-like, shape (n_features,) or (n_features, n_samples)
            Initial state vector(s)
        time_steps : int
            Number of timesteps to predict
            
        Returns
        -------
        predictions : torch.Tensor, shape (n_features, time_steps) or (n_features, time_steps, n_samples)
            Predicted states for each timestep
        """
        if self.modes is None:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Convert to tensor if needed
        if not isinstance(x0, torch.Tensor):
            x0 = torch.tensor(x0, dtype=torch.float32, device=self.device)
        else:
            x0 = x0.to(self.device)
        
        # Ensure x0 is 2D
        if x0.dim() == 1:
            x0 = x0.unsqueeze(1)
            squeeze_batch = True
        else:
            squeeze_batch = False
            
        batch_size = x0.shape[1]
        n_features = x0.shape[0]
        
        # Project initial condition onto DMD modes
        x0_complex = x0.to(torch.complex64)
        b0 = torch.linalg.lstsq(self.modes, x0_complex).solution
        
        # Prepare output tensor
        predictions = torch.zeros((n_features, time_steps, batch_size), 
                                dtype=torch.float32, device=self.device)
        
        # Compute predictions for each timestep
        for t in range(time_steps):
            # Evolve the modal coefficients
            b_t = torch.diag(torch.pow(self.eigenvalues, t)) @ b0
            
            # Reconstruct state
            x_t = self.modes @ b_t
            predictions[:, t, :] = x_t.real
        
        # Reshape output
        if squeeze_batch:
            return predictions.squeeze(2)  # (n_features, time_steps)
        else:
            return predictions  # (n_features, time_steps, n_samples)
        
    def reconstruct(self, n_steps=None):
        """
        Reconstruct the data using DMD modes.
        
        Parameters
        ----------
        n_steps : int, optional
            Number of timesteps to reconstruct. If None, uses original length.
            
        Returns
        -------
        X_dmd : torch.Tensor
            Reconstructed data matrix
        """
        if self.modes is None:
            raise RuntimeError("Model must be fitted before reconstruction")
            
        if n_steps is None:
            n_steps = self.n_snapshots
            
        # Time evolution
        time_dynamics = torch.zeros((len(self.eigenvalues), n_steps), 
                                   dtype=torch.complex64, device=self.device)
        
        for i in range(n_steps):
            time_dynamics[:, i] = torch.pow(self.eigenvalues, i) * self.amplitudes
            
        # Reconstruct
        X_dmd = self.modes @ time_dynamics
        
        return X_dmd.real
        
    def save_dmd(self, filepath):
        """
        Save the DMD model to file.
        
        Parameters
        ----------
        filepath : str
            Path to save the model
        """
        state_dict = {
            'svd_rank': self.svd_rank,
            'modes': self.modes.cpu() if self.modes is not None else None,
            'eigenvalues': self.eigenvalues.cpu() if self.eigenvalues is not None else None,
            'amplitudes': self.amplitudes.cpu() if self.amplitudes is not None else None,
            '_Atilde': self._Atilde.cpu() if self._Atilde is not None else None,
            'original_time': self.original_time,
            'n_snapshots': self.n_snapshots
        }
        torch.save(state_dict, filepath)
        
    def load_dmd(self, filepath):
        """
        Load a DMD model from file.
        
        Parameters
        ----------
        filepath : str
            Path to load the model from
        """
        state_dict = torch.load(filepath, weights_only=True)
        
        self.svd_rank = state_dict['svd_rank']
        self.original_time = state_dict['original_time']
        self.n_snapshots = state_dict['n_snapshots']
        
        # Move tensors to device
        if state_dict['modes'] is not None:
            self.modes = state_dict['modes'].to(self.device)
        if state_dict['eigenvalues'] is not None:
            self.eigenvalues = state_dict['eigenvalues'].to(self.device)
        if state_dict['amplitudes'] is not None:
            self.amplitudes = state_dict['amplitudes'].to(self.device)
        if state_dict['_Atilde'] is not None:
            self._Atilde = state_dict['_Atilde'].to(self.device)
            
    def _compute_rank(self, s):
        """
        Compute the truncation rank based on singular values.
        
        Parameters
        ----------
        s : torch.Tensor
            Singular values
            
        Returns
        -------
        r : int
            Truncation rank
        """
        if self.svd_rank == -1:
            # No truncation
            return len(s)
        elif self.svd_rank == 0:
            # Optimal rank (keeping 99.99% energy)
            cumsum = torch.cumsum(s**2, dim=0)
            total_energy = cumsum[-1]
            r = torch.where(cumsum >= 0.9999 * total_energy)[0][0].item() + 1
            return min(r, len(s))
        elif 0 < self.svd_rank < 1:
            # Energy-based truncation
            cumsum = torch.cumsum(s**2, dim=0)
            total_energy = cumsum[-1]
            r = torch.where(cumsum >= self.svd_rank * total_energy)[0][0].item() + 1
            return min(r, len(s))
        else:
            # Fixed rank
            return min(int(self.svd_rank), len(s))


# Test case
if __name__ == "__main__":
    # Create synthetic data: traveling wave
    x = np.linspace(0, 10, 100)
    t = np.linspace(0, 4*np.pi, 200)
    dt = t[1] - t[0]
    c = 0.5  # wave speed
    
    # Generate snapshots matrix (traveling wave)
    X = np.array([np.sin(x - c*dt*i) for i in range(len(t))]).T
    
    print(f"Data shape: {X.shape}")
    
    # Initialize and fit DMD
    dmd = TorchDMD(svd_rank=10, device='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {dmd.device}")
    
    dmd.fit(X)
    
    print(f"Number of modes: {len(dmd.eigenvalues)}")
    print(f"First 5 eigenvalues magnitude: {torch.abs(dmd.eigenvalues[:5])}")
    
    # Test prediction
    x0 = X[:, 0]
    x1_pred = dmd.predict(x0)
    x1_true = X[:, 1]
    
    # Convert prediction to numpy for comparison
    x1_pred_np = x1_pred.cpu().numpy()
    error = np.linalg.norm(x1_true - x1_pred_np)
    print(f"One-step prediction error: {error:.6f}")
    
    # Test reconstruction
    X_reconstructed = dmd.reconstruct()
    X_tensor = torch.tensor(X, dtype=torch.float32, device=dmd.device)
    reconstruction_error = torch.norm(X_tensor - X_reconstructed) / torch.norm(X_tensor)
    print(f"Relative reconstruction error: {reconstruction_error.item():.6f}")
    
    # Test save/load
    dmd.save_dmd('test_dmd_model.pth')
    
    dmd2 = TorchDMD(device=dmd.device)
    dmd2.load_dmd('test_dmd_model.pth')
    
    # Verify loaded model
    x1_pred2 = dmd2.predict(x0)
    x1_pred2_np = x1_pred2.cpu().numpy()
    load_error = np.linalg.norm(x1_pred_np - x1_pred2_np)
    print(f"Load verification error: {load_error:.6f}")
    
    # Clean up
    import os
    if os.path.exists('test_dmd_model.pth'):
        os.remove('test_dmd_model.pth')
    
    print("\nTest completed successfully!")