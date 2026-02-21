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

from src.utils.utils import weighted_MSELoss

# Features 
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
        self.C_forward = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
    
    def forward(self, state: torch.Tensor):
        z = self.K_S(state)
        z_next = self.C_forward(z)
        pred_s_next = self.K_S_preimage(z_next)
        return pred_s_next

    def batch_latent_forward(self, batch_z: torch.Tensor):
        return self.C_forward(batch_z)

    def latent_forward(self, z: torch.Tensor):
        return self.C_forward(z)
    
    def compute_loss(self, state_seq: torch.Tensor, state_next_seq: torch.Tensor, weight_matrix=None):
        B = state_seq.shape[0]
        device = state_seq.device
        z_seq = torch.zeros(B, self.seq_length, self.hidden_dim).to(device)
        z_next_seq = torch.zeros(B, self.seq_length, self.hidden_dim).to(device)

        loss_fwd = 0
        loss_identity = 0

        for i in range(self.seq_length):
            z_seq[:, i, :] = self.K_S(state_seq[:, i, :])
            z_next_seq[:, i, :] = self.K_S(state_next_seq[:, i, :])

        pred_z_next = self.batch_latent_forward(z_seq)
        
        for i in range(self.seq_length):
            recon_s = self.K_S_preimage(z_seq[:, i, :])
            recon_s_next = self.K_S_preimage(pred_z_next[:, i, :])

            if weight_matrix is not None:
                loss_fwd += weighted_MSELoss()(recon_s_next, state_next_seq[:, i, :], weight_matrix).sum()
                loss_identity += weighted_MSELoss()(recon_s, state_seq[:, i, :], weight_matrix).sum()
            else:
                loss_fwd += F.mse_loss(recon_s_next, state_next_seq[:, i, :])
                loss_identity += F.mse_loss(recon_s, state_seq[:, i, :])
                
        return loss_fwd, loss_identity, self.C_forward.weight
    
    def compute_loss_ms_latent_linear(self,
                                      state_seq: torch.Tensor,
                                      state_next_seq: torch.Tensor,
                                      multi_step: int,
                                      weight_matrix=None):
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
                weight_matrix_flat = weight_matrix.reshape(1, 1).repeat(B, state_seq_flat.shape[-1])
            else:
                weight_matrix_flat = weight_matrix.reshape(weight_matrix.shape[0], -1)
                if weight_matrix_flat.shape[0] != B:
                    weight_matrix_flat = weight_matrix_flat.reshape(1, -1).repeat(B, 1)

        z_seq = torch.zeros(B, self.seq_length, self.hidden_dim).to(device)
        z_next_seq = torch.zeros(B, self.seq_length, self.hidden_dim).to(device)

        loss_fwd = torch.tensor(0.0, device=device)
        loss_identity = torch.tensor(0.0, device=device)
        loss_fwd_latent = torch.tensor(0.0, device=device)
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

        pred_z_next = self.batch_latent_forward(z_seq)

        for i in range(self.seq_length):
            recon_s = self.K_S_preimage(z_seq[:, i, :])
            recon_s_next = self.K_S_preimage(pred_z_next[:, i, :])

            if len(original_state_shape) > 1:
                target_state = state_seq[:, i, :]
                target_state_next = state_next_seq[:, i, :]
                if recon_s.shape != target_state.shape:
                    recon_s = recon_s.reshape(B, *original_state_shape)
                if recon_s_next.shape != target_state_next.shape:
                    recon_s_next = recon_s_next.reshape(B, *original_state_shape)
            else:
                target_state = state_seq_flat[:, i, :]
                target_state_next = state_next_seq_flat[:, i, :]
                if recon_s.dim() > 2:
                    recon_s = recon_s.reshape(B, -1)
                if recon_s_next.dim() > 2:
                    recon_s_next = recon_s_next.reshape(B, -1)

            if weight_matrix is not None:
                loss_fwd += weighted_MSELoss()(recon_s_next, target_state_next, weight_matrix_flat).sum()
                loss_identity += weighted_MSELoss()(recon_s, target_state, weight_matrix_flat).sum()
            else:
                loss_fwd += F.mse_loss(recon_s_next, target_state_next)
                loss_identity += F.mse_loss(recon_s, target_state)

            loss_fwd_latent += F.mse_loss(pred_z_next[:, i, :], z_next_seq[:, i, :])

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

                    if weight_matrix is not None:
                        loss_multi_step += weighted_MSELoss()(pred_state, target_state, weight_matrix_flat).sum()
                    else:
                        loss_multi_step += F.mse_loss(pred_state, target_state)

            loss_multi_step = loss_multi_step / (self.seq_length - multi_step)

        return loss_fwd, loss_identity, loss_multi_step, loss_fwd_latent, self.C_forward.weight

    def save_C_forward(self, path, C_forward):
        C_forward_filename = path + '/' + 'C_forward.pt'
        print('[INFO] Saving C_forward weights to:', C_forward_filename)
        torch.save(C_forward, C_forward_filename)
        
    def save_model(self, path):
        # Save the model
        self.to('cpu')
        if not os.path.exists(path):
            os.makedirs(path)
        filename = path + '/forward_model.pt'
        print('[INFO] Saving forward_model weights to:', filename)
        torch.save(self.state_dict(), filename)
