import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import matplotlib.pyplot as plt
from torch import Tensor

import os
import sys

current_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", ".."))
sys.path.append(src_directory)

from src.utils.utils import is_symmetric, weighted_MSELoss


# Features Base Classes (same as Koopman)
class K_O_BASE(nn.Module):
    def __init__(self, nn_featrues:nn.Module, *args, **kwargs) -> None:
        super(K_O_BASE, self).__init__(*args, **kwargs)
        self.nn_features = nn_featrues
    
    def forward(self, x: torch.Tensor):
        return self.nn_features(x)

class K_H_BASE(nn.Module):
    def __init__(self, nn_featrues:nn.Module, hist_w:int, *args, **kwargs) -> None:
        # hist_w is the history window length
        super(K_H_BASE, self).__init__(*args, **kwargs)
        self.nn_features = nn_featrues
        self.hist_w = hist_w
    
    def forward(self, x: torch.Tensor):
        return self.nn_features(x)

class K_S_BASE(nn.Module):
    def __init__(self, nn_featrues:nn.Module, *args, **kwargs) -> None:
        super(K_S_BASE, self).__init__(*args, **kwargs)
        self.nn_features = nn_featrues
    
    def forward(self, x: torch.Tensor):
        return self.nn_features(x)

class K_S_preimage_BASE(nn.Module):
    def __init__(self, nn_featrues:nn.Module, *args, **kwargs) -> None:
        super(K_S_preimage_BASE, self).__init__(*args, **kwargs)
        self.nn_features = nn_featrues
    
    def forward(self, x: torch.Tensor):
        return self.nn_features(x)


class base_dmd_model(nn.Module):
    def __init__(self, K_S:nn.Module, K_S_preimage:nn.Module, 
                       seq_length: int, rank: int = None, *args, **kwargs) -> None:
        super(base_dmd_model, self).__init__(*args, **kwargs)
        self.K_S = K_S
        self.K_S_preimage = K_S_preimage
        self.hidden_dim = K_S.hidden_dims[-1]
        self.seq_length = seq_length
        self.rank = rank if rank is not None else self.hidden_dim  # DMD rank for truncation
        
        # DMD matrices
        self.A_dmd = None  # DMD matrix
        self.U = None      # Left singular vectors
        self.S = None      # Singular values
        self.V = None      # Right singular vectors
        self.Phi = None    # DMD modes
        self.Lambda = None # DMD eigenvalues
        self.b = None      # Initial amplitudes
    
    def forward(self, state: torch.Tensor):
        """Forward prediction using DMD"""
        z = self.K_S(state)
        z_next = self.batch_dmd_forward(z)
        pred_s_next = self.K_S_preimage(z_next)
        return pred_s_next

    def dmd_forward(self, z: torch.Tensor):
        """Apply DMD transformation to latent states (backward compatibility)"""
        return self.batch_dmd_forward(z)

    def batch_dmd_forward(self, z_batch: torch.Tensor):
        """
        Efficient batch DMD forward pass
        z_batch: [B, T, hidden_dim] or [B, hidden_dim]
        """
        if self.A_dmd is None:
            raise ValueError("DMD matrix not computed. Call compute_dmd_matrices first.")
        
        original_shape = z_batch.shape
        
        if z_batch.dim() == 3:
            # Batch sequence: [B, T, hidden_dim] -> [B*T, hidden_dim]
            B, T = z_batch.shape[:2]
            z_flat = z_batch.view(-1, self.hidden_dim)
            z_next_flat = torch.mm(z_flat, self.A_dmd.T)
            z_next = z_next_flat.view(B, T, self.hidden_dim)
        elif z_batch.dim() == 2:
            # Batch: [B, hidden_dim]
            z_next = torch.mm(z_batch, self.A_dmd.T)
        else:
            raise ValueError(f"Unsupported tensor dimension: {z_batch.dim()}")
            
        return z_next

    def iterative_prediction(self, initial_z: torch.Tensor, steps: int):
        """
        Iterative prediction using DMD matrix (more stable for long sequences)
        initial_z: [B, hidden_dim]
        steps: number of future steps to predict
        """
        if self.A_dmd is None:
            raise ValueError("DMD matrix not computed.")
            
        device = initial_z.device
        batch_size = initial_z.shape[0]
        
        predictions = []
        current_z = initial_z.clone()
        
        for _ in range(steps):
            current_z = self.batch_dmd_forward(current_z)
            predictions.append(current_z)
        
        return torch.stack(predictions, dim=1)  # [B, steps, hidden_dim]

    def compute_dmd_matrices(self, X: torch.Tensor, Y: torch.Tensor):
        """
        Compute DMD matrices from data matrices X and Y using PyTorch GPU acceleration
        X: [hidden_dim, N] - current states
        Y: [hidden_dim, N] - next states
        """
        device = X.device
        dtype = X.dtype
        
        # SVD of X using PyTorch
        U, S, Vt = torch.linalg.svd(X, full_matrices=False)
        V = Vt.T
        
        # Truncate to rank-r approximation
        r = min(self.rank, len(S))
        
        # Handle numerical stability for small singular values
        tolerance = 1e-10
        valid_indices = S > tolerance
        r = min(r, torch.sum(valid_indices).item())
        
        U_r = U[:, :r]
        S_r = S[:r]
        V_r = V[:, :r]
        
        # Compute DMD matrix A_tilde = U_r^T @ Y @ V_r @ diag(1/S_r)
        # More numerically stable: A_tilde = U_r^T @ Y @ V_r @ diag(1/S_r)
        S_r_inv = 1.0 / S_r
        A_tilde = torch.mm(torch.mm(U_r.T, Y), V_r * S_r_inv.unsqueeze(0))
        
        # Eigendecomposition of A_tilde using PyTorch
        try:
            Lambda, W = torch.linalg.eig(A_tilde)
        except RuntimeError as e:
            print(f"[WARNING] Eigendecomposition failed: {e}")
            print("[INFO] Using real-valued approximation")
            # Fallback: use real-valued approximation
            Lambda, W = torch.linalg.eig(A_tilde.real)
            Lambda = Lambda.to(torch.complex64)
            W = W.to(torch.complex64)
        
        # Ensure W is complex for consistent computation
        if W.dtype != torch.complex64:
            W = W.to(torch.complex64)
        if Lambda.dtype != torch.complex64:
            Lambda = Lambda.to(torch.complex64)
        
        # Compute DMD modes: Phi = Y @ V_r @ diag(1/S_r) @ W
        # Convert intermediate results to complex for matrix multiplication
        Y_complex = Y.to(torch.complex64)
        V_r_complex = V_r.to(torch.complex64)
        S_r_inv_complex = S_r_inv.to(torch.complex64)
        
        intermediate = torch.mm(Y_complex, V_r_complex * S_r_inv_complex.unsqueeze(0))
        Phi = torch.mm(intermediate, W)
        
        # Compute initial amplitudes: b = pinv(Phi) @ X[:, 0]
        try:
            X_first_complex = X[:, 0].to(torch.complex64)
            b = torch.linalg.solve(Phi, X_first_complex)
        except RuntimeError:
            # Use pseudoinverse if solve fails
            Phi_pinv = torch.linalg.pinv(Phi)
            X_first_complex = X[:, 0].to(torch.complex64)
            b = torch.mv(Phi_pinv, X_first_complex)
        
        # Full DMD matrix for direct application: A_dmd = Y @ pinv(X)
        try:
            X_pinv = torch.linalg.pinv(X)
            A_dmd = torch.mm(Y, X_pinv)
        except RuntimeError:
            # Alternative computation using SVD components
            X_pinv = torch.mm(V, torch.mm(torch.diag(1.0 / S), U.T))
            A_dmd = torch.mm(Y, X_pinv)
        
        # Store all components
        self.A_dmd = A_dmd.to(dtype)
        self.U = U_r.to(dtype)
        self.S = S_r.to(dtype)
        self.V = V_r.to(dtype)
        self.Phi = Phi
        self.Lambda = Lambda
        self.b = b
        
        return self.A_dmd

    def compute_loss(self, state_seq: torch.Tensor, state_next_seq: torch.Tensor, weight_matrix=None):
        """Compute loss and update DMD matrices"""
        B = state_seq.shape[0]
        device = state_seq.device
        z_seq = torch.zeros(B, self.seq_length, self.hidden_dim).to(device)
        z_next_seq = torch.zeros(B, self.seq_length, self.hidden_dim).to(device)

        loss_fwd = 0
        loss_identity = 0

        # Encode all states
        for i in range(self.seq_length):
            z_seq[:, i, :] = self.K_S(state_seq[:, i, :])
            z_next_seq[:, i, :] = self.K_S(state_next_seq[:, i, :])

        # Prepare data matrices for DMD
        # X: [hidden_dim, B*seq_length], Y: [hidden_dim, B*seq_length]
        X = z_seq.view(-1, self.hidden_dim).T  # [hidden_dim, B*T]
        Y = z_next_seq.view(-1, self.hidden_dim).T  # [hidden_dim, B*T]
        
        # Compute DMD matrices
        self.compute_dmd_matrices(X, Y)
        
        # Apply DMD forward
        pred_z_next = self.batch_dmd_forward(z_seq)
        
        # Compute losses
        for i in range(self.seq_length):
            recon_s = self.K_S_preimage(z_seq[:, i, :])
            recon_s_next = self.K_S_preimage(pred_z_next[:, i, :])

            if weight_matrix is not None:
                loss_fwd += weighted_MSELoss()(recon_s_next, state_next_seq[:, i, :], weight_matrix).sum()
                loss_identity += weighted_MSELoss()(recon_s, state_seq[:, i, :], weight_matrix).sum()
            else:
                loss_fwd += F.mse_loss(recon_s_next, state_next_seq[:, i, :])
                loss_identity += F.mse_loss(recon_s, state_seq[:, i, :])
                
        return loss_fwd, loss_identity, self.A_dmd

    def compute_loss_multi_step(self, state_seq: torch.Tensor, state_next_seq: torch.Tensor, 
                               multi_step: int, weight_matrix=None):
        """Compute loss with multi-step prediction using DMD"""
        B = state_seq.shape[0]
        device = state_seq.device
        
        original_state_shape = state_seq.shape[2:]
        
        if len(state_seq.shape) > 3:
            state_seq_flat = state_seq.view(B, self.seq_length, -1)
            state_next_seq_flat = state_next_seq.view(B, self.seq_length, -1)
        else:
            state_seq_flat = state_seq
            state_next_seq_flat = state_next_seq
        
        z_seq = torch.zeros(B, self.seq_length, self.hidden_dim).to(device)
        z_next_seq = torch.zeros(B, self.seq_length, self.hidden_dim).to(device)

        loss_fwd = 0
        loss_identity = 0
        loss_multi_step = 0

        # Encode states
        for i in range(self.seq_length):
            if len(original_state_shape) > 1:
                state_reshaped = state_seq_flat[:, i, :].view(B, *original_state_shape)
                state_next_reshaped = state_next_seq_flat[:, i, :].view(B, *original_state_shape)
                z_seq[:, i, :] = self.K_S(state_reshaped)
                z_next_seq[:, i, :] = self.K_S(state_next_reshaped)
            else:
                z_seq[:, i, :] = self.K_S(state_seq_flat[:, i, :])
                z_next_seq[:, i, :] = self.K_S(state_next_seq_flat[:, i, :])

        # Compute DMD matrices
        X = z_seq.view(-1, self.hidden_dim).T
        Y = z_next_seq.view(-1, self.hidden_dim).T
        self.compute_dmd_matrices(X, Y)

        # Single-step prediction
        pred_z_next = self.batch_dmd_forward(z_seq)
        
        for i in range(self.seq_length):
            recon_s = self.K_S_preimage(z_seq[:, i, :])
            recon_s_next = self.K_S_preimage(pred_z_next[:, i, :])
            
            if recon_s.shape != state_seq_flat[:, i, :].shape:
                recon_s = recon_s.view(B, -1)
            if recon_s_next.shape != state_next_seq_flat[:, i, :].shape:
                recon_s_next = recon_s_next.view(B, -1)

            if weight_matrix is not None:
                loss_fwd += weighted_MSELoss()(recon_s_next, state_next_seq_flat[:, i, :], weight_matrix).sum()
                loss_identity += weighted_MSELoss()(recon_s, state_seq_flat[:, i, :], weight_matrix).sum()
            else:
                loss_fwd += F.mse_loss(recon_s_next, state_next_seq_flat[:, i, :])
                loss_identity += F.mse_loss(recon_s, state_seq_flat[:, i, :])

        # Multi-step prediction
        if multi_step > 1 and self.seq_length > multi_step:
            for start_idx in range(self.seq_length - multi_step):
                current_z = z_seq[:, start_idx, :].clone()  # [B, hidden_dim]
                
                pred_z_sequence = []
                for step in range(multi_step):
                    current_z = self.batch_dmd_forward(current_z)
                    pred_z_sequence.append(current_z)

                pred_z_batch = torch.stack(pred_z_sequence, dim=1)  # [B, multi_step, hidden_dim]
                pred_z_flat = pred_z_batch.view(-1, self.hidden_dim)  # [B*multi_step, hidden_dim]
                pred_states_decoded = self.K_S_preimage(pred_z_flat)  # [B*multi_step, ...]

                if pred_states_decoded.dim() > 2:
                    pred_states_decoded = pred_states_decoded.view(pred_states_decoded.shape[0], -1)
                
                pred_states = pred_states_decoded.view(B, multi_step, -1)  # [B, multi_step, state_dim]
                target_states = state_seq_flat[:, start_idx+1:start_idx+multi_step+1, :]  # [B, multi_step, state_dim]

                for step in range(multi_step):
                    pred_state = pred_states[:, step, :]  # [B, state_dim]
                    target_state = target_states[:, step, :]  # [B, state_dim]
                    
                    if weight_matrix is not None:
                        loss_multi_step += weighted_MSELoss()(pred_state, target_state, weight_matrix).sum()
                    else:
                        loss_multi_step += F.mse_loss(pred_state, target_state)
            
            loss_multi_step = loss_multi_step / (self.seq_length - multi_step)
        
        return loss_fwd, loss_identity, loss_multi_step, self.A_dmd

    def predict_future(self, initial_state: torch.Tensor, steps: int):
        """Predict future states using DMD eigenvalue/eigenvector decomposition"""
        if self.Phi is None or self.Lambda is None or self.b is None:
            raise ValueError("DMD eigendecomposition not computed.")
            
        device = initial_state.device
        batch_size = initial_state.shape[0] if initial_state.dim() > 1 else 1
        
        # Encode initial state
        if initial_state.dim() == 1:
            initial_state = initial_state.unsqueeze(0)
        
        z_0 = self.K_S(initial_state)  # [B, hidden_dim]
        
        # For batch processing, we need to compute initial amplitudes for each sample
        if batch_size > 1:
            # Compute amplitudes for each sample in the batch
            try:
                # b_batch = pinv(Phi) @ z_0.T  -> [modes, B]
                Phi_pinv = torch.linalg.pinv(self.Phi)
                b_batch = torch.mm(Phi_pinv, z_0.T.to(torch.complex64))  # [modes, B]
            except RuntimeError:
                # Fallback: use the pre-computed b for all samples
                b_batch = self.b.unsqueeze(1).repeat(1, batch_size)  # [modes, B]
        else:
            b_batch = self.b.unsqueeze(1)  # [modes, 1]
        
        # Predict future states
        future_states = []
        
        for t in range(1, steps + 1):
            # Lambda^t for all eigenvalues
            lambda_t = torch.pow(self.Lambda, t)  # [modes]
            
            # Broadcast for batch computation: [modes, B] * [modes, 1] -> [modes, B]
            lambda_t_expanded = lambda_t.unsqueeze(1).expand(-1, batch_size)
            
            # DMD prediction: z(t) = Phi @ diag(lambda^t) @ b
            # [hidden_dim, modes] @ ([modes, B] * [modes, B]) -> [hidden_dim, B]
            z_t_complex = torch.mm(self.Phi, lambda_t_expanded * b_batch)
            
            # Take real part and convert to appropriate dtype
            z_t = torch.real(z_t_complex).T.to(z_0.dtype)  # [B, hidden_dim]
            
            # Decode to original space
            state_t = self.K_S_preimage(z_t)
            future_states.append(state_t)
            
        return torch.stack(future_states, dim=1)  # [B, steps, state_dim]

    def get_dmd_modes(self):
        """Return DMD modes and eigenvalues for analysis"""
        return self.Phi, self.Lambda

    def get_growth_rates_and_frequencies(self):
        """Extract growth rates and frequencies from DMD eigenvalues"""
        if self.Lambda is None:
            raise ValueError("DMD eigenvalues not computed.")
            
        # Growth rates (real part of log(lambda))
        # Handle numerical stability for log computation
        lambda_log = torch.log(self.Lambda + 1e-15)  # Add small epsilon for stability
        growth_rates = torch.real(lambda_log)
        
        # Frequencies (imaginary part of log(lambda))
        frequencies = torch.imag(lambda_log)
        
        return growth_rates, frequencies

    def save_A_dmd(self, path, A_dmd):
        """Save DMD matrix"""
        A_dmd_filename = path + '/' + 'A_dmd.pt'
        print('[INFO] Saving A_dmd matrix to:', A_dmd_filename)
        torch.save(A_dmd, A_dmd_filename)
        
    def save_dmd_components(self, path):
        """Save all DMD components"""
        components = {
            'A_dmd': self.A_dmd,
            'U': self.U,
            'S': self.S, 
            'V': self.V,
            'Phi': self.Phi,
            'Lambda': self.Lambda,
            'b': self.b,
            'rank': self.rank
        }
        
        dmd_filename = path + '/' + 'dmd_components.pt'
        print('[INFO] Saving DMD components to:', dmd_filename)
        torch.save(components, dmd_filename)
    
    def load_dmd_components(self, path, device):
        """Load all DMD components"""
        dmd_filename = path + 'dmd_components.pt'
        print('[INFO] Loading DMD components from:', dmd_filename)
        
        components = torch.load(dmd_filename, weights_only=True, map_location=device)
        self.A_dmd = components['A_dmd']
        self.U = components['U']
        self.S = components['S']
        self.V = components['V'] 
        self.Phi = components['Phi']
        self.Lambda = components['Lambda']
        self.b = components['b']
        self.rank = components['rank']
        
    def save_model(self, path):
        """Save the model"""
        self.to('cpu')
        filename = path + '/dmd_model.pt'
        print('[INFO] Saving DMD model weights to:', filename)
        torch.save(self.state_dict(), filename)

    def analyze_stability(self):
        """Analyze DMD stability based on eigenvalues"""
        if self.Lambda is None:
            raise ValueError("DMD eigenvalues not computed.")
            
        # Compute magnitude of eigenvalues
        eigenvalue_magnitudes = torch.abs(self.Lambda)
        
        # System is stable if all eigenvalues have magnitude <= 1
        is_stable = torch.all(eigenvalue_magnitudes <= 1.0)
        
        max_magnitude = torch.max(eigenvalue_magnitudes)
        unstable_modes = torch.sum(eigenvalue_magnitudes > 1.0)
        
        stability_info = {
            'is_stable': is_stable.item(),
            'max_eigenvalue_magnitude': max_magnitude.item(),
            'num_unstable_modes': unstable_modes.item(),
            'eigenvalue_magnitudes': eigenvalue_magnitudes
        }
        
        return stability_info