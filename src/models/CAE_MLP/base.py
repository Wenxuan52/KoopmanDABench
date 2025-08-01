import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import matplotlib.pyplot as plt
from torch import Tensor

import os
import sys
current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", ".."))
sys.path.append(src_directory)

from src.utils.utils import is_symmetric, weighted_MSELoss

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
                       seq_length: int, hidden_dim: int = None, *args, **kwargs) -> None:
        super(base_forward_model, self).__init__(*args, **kwargs)
        self.K_S = K_S
        self.K_S_preimage = K_S_preimage
        self.hidden_dim = hidden_dim if hidden_dim is not None else K_S.hidden_dims[-1]
        self.seq_length = seq_length
        
        # Replace linear layer with MLP for weak nonlinearity
        self.C_forward = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        )
    
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
                
        return loss_fwd, loss_identity, self.C_forward
    
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