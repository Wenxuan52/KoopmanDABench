import torch 
import torch.nn as nn
import torch.nn.functional as F
# import tltorch
import time
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


# Featrues 
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


class base_forward_model(nn.Module):
    def __init__(self, K_S:nn.Module, K_S_preimage:nn.Module, 
                       seq_length: int, *args, **kwargs) -> None:
        super(base_forward_model, self).__init__(*args, **kwargs)
        self.K_S = K_S
        self.K_S_preimage = K_S_preimage
        self.hidden_dim = K_S.hidden_dims[-1]
        self.seq_length = seq_length
        self.C_forward = None
    
    def forward(self, state: torch.Tensor):
        z = self.K_S(state)
        z_next = torch.mm(z, self.C_forward)
        pred_s_next = self.K_S_preimage(z_next)

        
        return pred_s_next

    def batch_latent_forward(self, batch_z: torch.Tensor):
        B = batch_z.shape[0]
        if self.C_forward.dim() == 2:
            C_forward = self.C_forward.unsqueeze(0).repeat(B, 1, 1)
        else:
            C_forward = self.C_forward
        batch_z_next = torch.bmm(batch_z, C_forward)
        return batch_z_next

    def latent_forward(self, z: torch.Tensor):
        z_next = torch.mm(z, self.C_forward)
        return z_next
    
    def compute_loss(self, state_seq, state_next_seq, 
                 weight_matrix=None,
                 mi_k: int = 2, mi_temperature: float = 0.1):
        B = state_seq.shape[0]
        device = state_seq.device
        z_seq = torch.zeros(B, self.seq_length, self.hidden_dim).to(device)
        z_next_seq = torch.zeros(B, self.seq_length, self.hidden_dim).to(device)

        loss_fwd = 0.0
        loss_identity = 0.0
        mi_loss = torch.tensor(0.0, device=device)
        entropy_loss = torch.tensor(0.0, device=device)

        # decode
        for i in range(self.seq_length):
            z_seq[:, i, :] = self.K_S(state_seq[:, i, :])
            z_next_seq[:, i, :] = self.K_S(state_next_seq[:, i, :])

        # Koopman
        z_seq_pinv = self.batch_pinv(z_seq, I_factor=1e-1)
        forward_weights = torch.bmm(z_seq_pinv, z_next_seq).mean(dim=0).repeat(B, 1, 1)
        self.C_forward = forward_weights.detach().clone()
        pred_z_next = self.batch_latent_forward(z_seq)

        # MSE
        for i in range(self.seq_length):
            recon_s = self.K_S_preimage(z_seq[:, i, :])
            recon_s_next = self.K_S_preimage(pred_z_next[:, i, :])

            if weight_matrix is not None:
                loss_fwd += weighted_MSELoss()(recon_s_next, state_next_seq[:, i, :], weight_matrix).sum()
                loss_identity += weighted_MSELoss()(recon_s, state_seq[:, i, :], weight_matrix).sum()
            else:
                loss_fwd += F.mse_loss(recon_s_next, state_next_seq[:, i, :])
                loss_identity += F.mse_loss(recon_s, state_seq[:, i, :])

        mi_loss = self.info_nce(z_seq, k=mi_k, temperature=mi_temperature, use_cosine=True)
        
        entropy_loss, Sigma, evals  = self.von_neumann_entropy(z_seq, center=True)
        
        return {
            "loss_fwd": loss_fwd,
            "loss_identity": loss_identity,
            "loss_mi": mi_loss,
            "loss_entropy": entropy_loss,
            "C_forward": self.C_forward.mean(dim=0),
            "Entropy": {'Sigma': Sigma, 'evals': evals},
        }

    def compute_loss_multi_step(self,
                                state_seq: torch.Tensor,
                                state_next_seq: torch.Tensor,
                                multi_step: int,
                                weight_matrix=None,
                                mi_k: int = 2,
                                mi_temperature: float = 0.1):
        B = state_seq.shape[0]
        device = state_seq.device
        original_state_shape = state_seq.shape[2:]

        if len(state_seq.shape) > 3:
            state_seq_flat = state_seq.reshape(B, self.seq_length, -1)
            state_next_seq_flat = state_next_seq.reshape(B, self.seq_length, -1)
        else:
            state_seq_flat = state_seq
            state_next_seq_flat = state_next_seq

        weight_matrix_flat = None
        if weight_matrix is not None:
            if weight_matrix.dim() == 0:
                weight_matrix_flat = weight_matrix.view(1, 1).repeat(B, state_seq_flat.shape[-1])
            elif weight_matrix.dim() == 1:
                weight_matrix_flat = weight_matrix.view(1, -1).repeat(B, 1)
            else:
                w = weight_matrix.reshape(weight_matrix.shape[0], -1)
                if w.shape[0] != B:
                    weight_matrix_flat = w.reshape(1, -1).repeat(B, 1)
                else:
                    weight_matrix_flat = w

        z_seq = torch.zeros(B, self.seq_length, self.hidden_dim, device=device)
        z_next_seq = torch.zeros(B, self.seq_length, self.hidden_dim, device=device)

        loss_fwd = torch.tensor(0.0, device=device)
        loss_identity = torch.tensor(0.0, device=device)
        loss_multi_step = torch.tensor(0.0, device=device)

        for i in range(self.seq_length):
            if len(original_state_shape) > 1:
                state_reshaped = state_seq_flat[:, i, :].reshape(B, *original_state_shape)
                state_next_reshaped = state_next_seq_flat[:, i, :].reshape(B, *original_state_shape)
                z_seq[:, i, :] = self.K_S(state_reshaped)
                z_next_seq[:, i, :] = self.K_S(state_next_reshaped)
            else:
                z_seq[:, i, :] = self.K_S(state_seq_flat[:, i, :])
                z_next_seq[:, i, :] = self.K_S(state_next_seq_flat[:, i, :])

        z_seq_pinv = self.batch_pinv(z_seq, I_factor=1e-2)
        forward_weights = torch.bmm(z_seq_pinv, z_next_seq).mean(dim=0).repeat(B, 1, 1)
        self.C_forward = forward_weights.detach().clone()

        pred_z_next = self.batch_latent_forward(z_seq)

        for i in range(self.seq_length):
            recon_s = self.K_S_preimage(z_seq[:, i, :])
            recon_s_next = self.K_S_preimage(pred_z_next[:, i, :])

            if recon_s.dim() > 2:
                recon_s = recon_s.reshape(B, -1)
            if recon_s_next.dim() > 2:
                recon_s_next = recon_s_next.reshape(B, -1)

            target_s = state_seq_flat[:, i, :]
            target_s_next = state_next_seq_flat[:, i, :]

            if weight_matrix_flat is not None:
                loss_fwd += weighted_MSELoss()(recon_s_next, target_s_next, weight_matrix_flat).sum()
                loss_identity += weighted_MSELoss()(recon_s, target_s, weight_matrix_flat).sum()
            else:
                loss_fwd += F.mse_loss(recon_s_next, target_s_next)
                loss_identity += F.mse_loss(recon_s, target_s)

        if multi_step > 1 and self.seq_length > multi_step:
            for start_idx in range(self.seq_length - multi_step):
                current_z = z_seq[:, start_idx, :].clone()
                pred_z_sequence = []

                for _ in range(multi_step):
                    current_z = self.batch_latent_forward(current_z.unsqueeze(1)).squeeze(1)
                    pred_z_sequence.append(current_z)

                pred_z_batch = torch.stack(pred_z_sequence, dim=1)
                pred_z_flat = pred_z_batch.reshape(-1, self.hidden_dim)
                pred_states_decoded = self.K_S_preimage(pred_z_flat)

                if pred_states_decoded.dim() > 2:
                    pred_states_decoded = pred_states_decoded.reshape(pred_states_decoded.shape[0], -1)

                pred_states = pred_states_decoded.reshape(B, multi_step, -1)
                target_states = state_seq_flat[:, start_idx + 1:start_idx + multi_step + 1, :]

                for step in range(multi_step):
                    pred_state = pred_states[:, step, :]
                    target_state = target_states[:, step, :]
                    if weight_matrix_flat is not None:
                        loss_multi_step += weighted_MSELoss()(pred_state, target_state, weight_matrix_flat).sum()
                    else:
                        loss_multi_step += F.mse_loss(pred_state, target_state)

            loss_multi_step = loss_multi_step / (self.seq_length - multi_step)

        t0 = time.time()
        mi_loss = self.info_nce(z_seq, k=mi_k, temperature=mi_temperature, use_cosine=True)
        mi_time = time.time() - t0
        print(f"[Timing] loss_mi (info_nce) time: {mi_time * 1000:.2f} ms")

        t1 = time.time()
        entropy_loss = self.von_neumann_entropy(z_seq, center=True)
        entropy_time = time.time() - t1
        print(f"[Timing] loss_entropy (von_neumann_entropy) time: {entropy_time * 1000:.2f} ms")


        return {
            "loss_fwd": loss_fwd,
            "loss_identity": loss_identity,
            "loss_ms": loss_multi_step,
            "loss_mi": mi_loss,
            "loss_entropy": entropy_loss,
            "C_forward": self.C_forward.mean(dim=0),
        }

    def info_nce(self, z_seq: torch.Tensor, k: int = 2, 
                                  temperature: float = 0.1, use_cosine: bool = True):
        
        B, T, H = z_seq.shape
        device = z_seq.device
        
        if T <= 2 * k:
            return torch.tensor(0.0, device=device)
        
        # Only compute [k, T-k-1] for time steps with complete neighborhoods
        valid_indices = torch.arange(k, T - k, device=device)
        n_valid = len(valid_indices)
        
        if n_valid == 0:
            return torch.tensor(0.0, device=device)
        
        # Normalize the entire sequence (if using cosine similarity)
        if use_cosine:
            z_normalized = F.normalize(z_seq, dim=-1)  # [B, T, H]
        else:
            z_normalized = z_seq
        
        total_loss = 0.0
        
        # Calculate InfoNCE for each valid anchor position
        for n in valid_indices:
            # anchor: z_n [B, H]
            anchor = z_normalized[:, n, :]
            
            # Positive sample: neighborhood P_n = {z_{n±i} | 1≤i≤k}
            pos_indices = (torch.cat([
                torch.arange(n-k, n, device=device),  # past k
                torch.arange(n+1, n+k+1, device=device)  # k future
            ]))
            
            # Negative samples: all non-neighborhood positions (excluding themselves and neighbors)
            all_indices = torch.arange(T, device=device)
            neighbor_and_self = torch.cat([
                torch.tensor([n], device=device),  # self
                pos_indices  # Neighborhood
            ])
            neg_mask = ~torch.isin(all_indices, neighbor_and_self)
            neg_indices = all_indices[neg_mask]
            
            if len(neg_indices) == 0:
                continue
            
            # Construct positive and negative samples
            positives = z_normalized[:, pos_indices, :]  # [B, 2k, H]  
            negatives = z_normalized[:, neg_indices, :]   # [B, n_neg, H]
            
            # Calculate similarity score
            pos_scores = torch.bmm(
                anchor.unsqueeze(1),  # [B, 1, H]
                positives.transpose(1, 2)  # [B, H, 2k]
            ).squeeze(1) / temperature  # [B, 2k]
            
            neg_scores = torch.bmm(
                anchor.unsqueeze(1),  # [B, 1, H]
                negatives.transpose(1, 2)  # [B, H, n_neg]
            ).squeeze(1) / temperature  # [B, n_neg]
            
            # Multiple positive variant InfoNCE loss
            # log(sum(exp(pos_scores)) / (sum(exp(pos_scores)) + sum(exp(neg_scores))))
            
            # Numerically stable calculations
            max_pos = pos_scores.max(dim=1, keepdim=True)[0]  # [B, 1]
            max_neg = neg_scores.max(dim=1, keepdim=True)[0]  # [B, 1]
            max_all = torch.max(max_pos, max_neg)  # [B, 1]
            
            pos_scores_stable = pos_scores - max_all
            neg_scores_stable = neg_scores - max_all
            
            pos_exp_sum = torch.exp(pos_scores_stable).sum(dim=1)  # [B]
            neg_exp_sum = torch.exp(neg_scores_stable).sum(dim=1)  # [B]
            
            # InfoNCE loss: -log(positive sample probability)
            loss = -torch.log(pos_exp_sum / (pos_exp_sum + neg_exp_sum + 1e-8))
            total_loss += loss.mean()
        
        return total_loss / n_valid

    def von_neumann_entropy(self, z_seq: torch.Tensor, center: bool = True, eps: float = 1e-8):
        B, T, H = z_seq.shape
        Z = z_seq.reshape(B * T, H)
        
        if center:
            Z = Z - Z.mean(dim=0, keepdim=True)
        
        # Use more stable covariance calculation
        N = Z.shape[0]
        Sigma = torch.mm(Z.t(), Z) / (N - 1)  # Unbiased estimation
        
        # Ensure positive definiteness
        Sigma = Sigma + eps * torch.eye(H, device=Z.device)
        trace = torch.trace(Sigma)
        rho = Sigma / trace.clamp_min(eps)
        
        # Using eigvalsh is more stable for real symmetric matrices
        evals = torch.linalg.eigvalsh(rho)
        evals = evals.clamp_min(eps)
        
        S = -(evals * torch.log(evals)).sum()
        return S, Sigma.detach(), evals.detach()

    @staticmethod
    def batch_pinv(z_seq: torch.Tensor, I_factor:float=1e-2):
        # inverse of z_seq
        # za_seq: [B, T, Dim_s]
        # I_factor: Identity factor
        B, T, D = z_seq.size()
        device = z_seq.device

        trans = T < D
        if trans:
            z_seq = torch.transpose(z_seq, 1, 2)
            T, D = D, T

        if not z_seq.is_cuda:
            z_seq = z_seq.to('cpu')
            I = torch.eye(D)[None, :, :].repeat(B, 1, 1).to('cpu')
        else:
            I = torch.eye(D)[None, :, :].repeat(B, 1, 1).to(device)
            
        z_seq_T = torch.transpose(z_seq, 1, 2)
        z_seq_pinv = torch.linalg.solve(
            torch.bmm(z_seq_T, z_seq) + I_factor * I,
            z_seq_T
        )
        if trans:
            z_seq_pinv = torch.transpose(z_seq_pinv, 1, 2)

        return z_seq_pinv.to(device)
    
    def save_C_forward(self, path, C_forward):
        C_forward_filename = path + '/' + 'C_forward.pt'
        print('[INFO] Saving C_forward weights to:', C_forward_filename)
        torch.save(C_forward, C_forward_filename)
        
    def save_model(self, path):
        # Save the model
        self.to('cpu')
        filename = path + '/forward_model.pt'
        print('[INFO] Saving forward_model weights to:', filename)
        torch.save(self.state_dict(), filename)
    
