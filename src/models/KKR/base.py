import torch 
import torch.nn as nn
import torch.nn.functional as F
# import tltorch
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
    
    def compute_loss(self, state_seq: torch.Tensor, state_next_seq: torch.Tensor, weight_matrix=None):
        B = state_seq.shape[0]
        device = state_seq.device
        z_seq = torch.zeros(B, self.seq_length, self.hidden_dim).to(device)

        z_next_seq = torch.zeros(B, self.seq_length, self.hidden_dim).to(device)

        loss_fwd = 0
        loss_identity = 0
        # loss_distance = torch.tensor(0.0).to(device)

        for i in range(self.seq_length):
            z_seq[:, i, :] = self.K_S(state_seq[:, i, :])

            z_next_seq[:, i, :] = self.K_S(state_next_seq[:, i, :])


        z_seq_pinv = self.batch_pinv(z_seq, I_factor=1e-1)
        forward_weights = torch.bmm(z_seq_pinv, z_next_seq).mean(dim=0).repeat(B, 1, 1)

        self.C_forward = forward_weights.detach().clone()
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
                
        return loss_fwd, loss_identity, self.C_forward.mean(dim=0)

    def compute_loss_multi_step(self, state_seq: torch.Tensor, state_next_seq: torch.Tensor, 
                            multi_step: int, weight_matrix=None):
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
            else:
                weight_matrix_flat = weight_matrix.reshape(weight_matrix.shape[0], -1)
                if weight_matrix_flat.shape[0] != B:
                    weight_matrix_flat = weight_matrix_flat.reshape(1, -1).repeat(B, 1)

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

            if recon_s.shape != state_seq_flat[:, i, :].shape:
                recon_s = recon_s.reshape(B, -1)
            if recon_s_next.shape != state_next_seq_flat[:, i, :].shape:
                recon_s_next = recon_s_next.reshape(B, -1)

            if weight_matrix_flat is not None:
                loss_fwd += weighted_MSELoss()(recon_s_next, state_next_seq_flat[:, i, :], weight_matrix_flat).sum()
                loss_identity += weighted_MSELoss()(recon_s, state_seq_flat[:, i, :], weight_matrix_flat).sum()
            else:
                loss_fwd += F.mse_loss(recon_s_next, state_next_seq_flat[:, i, :])
                loss_identity += F.mse_loss(recon_s, state_seq_flat[:, i, :])

        if multi_step > 1 and self.seq_length > multi_step:
            for start_idx in range(self.seq_length - multi_step):
                start_z = z_seq[:, start_idx, :].clone()
                current_z = start_z
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

        return loss_fwd, loss_identity, loss_multi_step, self.C_forward.mean(dim=0)


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
                weight_matrix_flat = weight_matrix.view(1, 1).repeat(B, state_seq_flat.shape[-1])
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

        z_seq_pinv = self.batch_pinv(z_seq, I_factor=1e-2)
        forward_weights = torch.bmm(z_seq_pinv, z_next_seq).mean(dim=0).repeat(B, 1, 1)

        self.C_forward = forward_weights.detach().clone()
        pred_z_next = self.batch_latent_forward(z_seq)

        for i in range(self.seq_length):
            recon_s = self.K_S_preimage(z_seq[:, i, :])
            recon_s_next = self.K_S_preimage(pred_z_next[:, i, :])

            if recon_s.shape != state_seq_flat[:, i, :].shape:
                recon_s = recon_s.reshape(B, -1)
            if recon_s_next.shape != state_next_seq_flat[:, i, :].shape:
                recon_s_next = recon_s_next.reshape(B, -1)

            if weight_matrix is not None:
                loss_fwd += weighted_MSELoss()(recon_s_next, state_next_seq_flat[:, i, :], weight_matrix_flat).sum()
                loss_identity += weighted_MSELoss()(recon_s, state_seq_flat[:, i, :], weight_matrix_flat).sum()
            else:
                loss_fwd += F.mse_loss(recon_s_next, state_next_seq_flat[:, i, :])
                loss_identity += F.mse_loss(recon_s, state_seq_flat[:, i, :])

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

        return loss_fwd, loss_identity, loss_multi_step, loss_fwd_latent, self.C_forward.mean(dim=0)

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

    def compute_Q_B(self, dynamics_dataset:torch.utils.data.Dataset, device:str='cpu', save_path:str=None):
        # Compute the Covariance Matrix Cov(s_t,s_{t+1})
        # Compute the Covariance Matrix Cov(s_t,s_t)
        N = len(dynamics_dataset)
        Q = torch.zeros((N, self.hidden_dim)).to(device)
        B = torch.zeros((N, self.hidden_dim)).to(device)
        
        BS = 2048
        assert dynamics_dataset.seq_length == 1, "The sequence length of the dataset should be 1"
        
        dataloader = torch.utils.data.DataLoader(dynamics_dataset, batch_size=BS, shuffle=False)
        
        for i, batch_data in enumerate(dataloader):
            state, next_state = batch_data
            
            B = state.shape[0]
            
            state = state.to(device)
            next_state = next_state.to(device)
            state_feature = self.K_S(state)
            next_state_feature = self.K_S(next_state)
            forward_error = next_state_feature - self.batch_latent_forward(state_feature)
            for j in range(B):
                Q[i*B+j] = forward_error[j]
                B[i*B+j] = state_feature[j]
            
        Q = torch.cov(Q.T)
        B = torch.cov(B.T)
        
        # MPS does not support float64
        Q = Q.to("cpu")
        B = B.to("cpu")
        
        Q = torch.pinverse(Q + 0.1*torch.eye(self.hidden_dim))
        B = torch.pinverse(B + torch.eye(self.hidden_dim))
        
        if not is_symmetric(Q):
            print('[INFO] Q is not symmetric, using symmetriziation')
            Q = 0.5*(Q + Q.T)
        else:
            print('[INFO] Q is symmetric')
            
        if not is_symmetric(B):
            print('[INFO] B is not symmetric, using symmetriziation')
            B = 0.5*(B + B.T)
        else:
            print('[INFO] B is symmetric')
            
        # plt.imshow(Q.cpu().detach().numpy())
        # plt.colorbar()
        # plt.show()
        
        # plt.imshow(B.cpu().detach().numpy())
        # plt.colorbar()
        # plt.show()
          
        if save_path is not None:
            save_path_Q = save_path + '/' + 'Q.pt'   
            print('[INFO] save Q to: ', save_path_Q)
            torch.save(Q, save_path_Q)
            
            save_path_B = save_path + '/' + 'B.pt'
            print('[INFO] save B to: ', save_path_B)
            torch.save(B, save_path_B)
        
        return Q, B
            
    def compute_z_b(self, dynamics_dataset:torch.utils.data.Dataset, device:str='cpu', save_path:str=None):
        N = len(dynamics_dataset)
        z_b = torch.zeros((N, self.hidden_dim)).to(device)
        BS = 32
        assert dynamics_dataset.seq_length == 1, "The sequence length of the dataset should be 1"
        
        dataloader = torch.utils.data.DataLoader(dynamics_dataset, batch_size=BS, shuffle=False)
        
        with torch.no_grad():
            for i, batch_data in enumerate(dataloader):
                state, _ = batch_data
                state = state.squeeze(1)
                bs = state.shape[0]
                state = state.to(device)
                state_feature = self.K_S(state)
                for j in range(bs):
                    z_b[i*bs+j] = state_feature[j]
        z_b = z_b.mean(dim=0)
        if save_path is not None:
            save_path_z_b = save_path + '/' + 'z_b.pt'
            print('[INFO] save z_b to: ', save_path_z_b)
            torch.save(z_b, save_path_z_b)
        
        return z_b
