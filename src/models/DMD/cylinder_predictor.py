import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import yaml
from typing import Optional
import matplotlib.pyplot as plt

from cylinder_model import CYLINDER_C_FORWARD

class CylinderDynamicsDataset(Dataset):
    """Memory-efficient Dataset for Cylinder flow dynamics data"""
    
    def __init__(self, data_path: str, seq_length: int = 10, normalize: bool = True, 
                 train_ratio: float = 0.8, random_seed: int = 42, subsample_ratio: float = 0.1):
        self.data_path = data_path
        self.seq_length = seq_length
        self.normalize = normalize
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        self.subsample_ratio = subsample_ratio
        
        # Load and process data
        self._load_data()
        self._create_indices()
        
        if self.normalize:
            self._compute_normalization_stats()
    
    def _load_data(self):
        """Load raw data without creating all sequences in memory"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Cylinder dataset not found at {self.data_path}")
        
        # Load data: [49, 1000, 2, 64, 64]
        self.raw_data = np.load(self.data_path)
        print(f"[INFO] Loaded Cylinder data with shape: {self.raw_data.shape}")
        
        if len(self.raw_data.shape) != 5:
            raise ValueError(f"Expected 5D data [samples, time, channels, height, width], got {self.raw_data.shape}")
        
        self.n_samples, self.n_time, self.n_channels, self.height, self.width = self.raw_data.shape
    
    def _create_indices(self):
        """Create indices for valid sequences with subsampling"""
        valid_indices = []
        
        for sample_idx in range(self.n_samples):
            for t in range(0, self.n_time - self.seq_length, max(1, int(1/self.subsample_ratio))):
                valid_indices.append((sample_idx, t))
        
        # Random split
        np.random.seed(self.random_seed)
        np.random.shuffle(valid_indices)
        
        train_size = int(len(valid_indices) * self.train_ratio)
        self.train_indices = valid_indices[:train_size]
        self.val_indices = valid_indices[train_size:]
        
        # Use only training indices for now
        self.indices = self.train_indices
        
        print(f"[INFO] Created {len(self.train_indices)} training sequences")
        print(f"[INFO] Created {len(self.val_indices)} validation sequences")
        print(f"[INFO] Subsampling ratio: {self.subsample_ratio}")
    
    def _compute_normalization_stats(self):
        """Compute normalization statistics from a subset of training data"""
        if not self.normalize:
            return
            
        # Sample a subset of training data for computing stats
        sample_size = min(1000, len(self.train_indices))
        sample_indices = np.random.choice(len(self.train_indices), sample_size, replace=False)
        
        sample_data = []
        for idx in sample_indices:
            sample_idx, t = self.train_indices[idx]
            seq_current = self.raw_data[sample_idx, t:t+self.seq_length]
            sample_data.append(seq_current)
        
        sample_data = np.array(sample_data)  # [sample_size, seq_length, channels, height, width]
        
        # Compute mean and std across all dimensions except channels
        # Result shape: [channels]
        self.mean = np.mean(sample_data, axis=(0, 1, 3, 4))  # [channels]
        self.std = np.std(sample_data, axis=(0, 1, 3, 4))    # [channels]
        
        # Reshape for broadcasting: [channels] -> [channels, 1, 1]
        self.mean = self.mean.reshape(-1, 1, 1)
        self.std = self.std.reshape(-1, 1, 1)
        
        # Avoid division by zero
        self.std[self.std < 1e-8] = 1.0
        
        print(f"[INFO] Normalization computed from {sample_size} samples")
        print(f"[INFO] Mean shape: {self.mean.shape}, Std shape: {self.std.shape}")
        print(f"[INFO] Mean: {self.mean.flatten()}, Std: {self.std.flatten()}")
    
    def set_split(self, split='train'):
        """Set which split to use: 'train' or 'val'"""
        if split == 'train':
            self.indices = self.train_indices
        elif split == 'val':
            self.indices = self.val_indices
        else:
            raise ValueError("Split must be 'train' or 'val'")
        print(f"[INFO] Dataset split set to: {split} ({len(self.indices)} samples)")

    def __len__(self):
        """Return the number of samples in the current split"""
        return len(self.indices)
    
    def __getitem__(self, idx):
        sample_idx, t = self.indices[idx]
        
        # Extract sequences on-the-fly
        seq_current = self.raw_data[sample_idx, t:t+self.seq_length].copy()  # [seq_length, channels, height, width]
        seq_next = self.raw_data[sample_idx, t+1:t+self.seq_length+1].copy()  # [seq_length, channels, height, width]
        
        # Apply normalization if needed
        if self.normalize and hasattr(self, 'mean'):
            seq_current = (seq_current - self.mean) / self.std
            seq_next = (seq_next - self.mean) / self.std
        
        return torch.tensor(seq_current, dtype=torch.float32), torch.tensor(seq_next, dtype=torch.float32)
    
    def set_split(self, split='train'):
        """Set which split to use: 'train' or 'val'"""
        if split == 'train':
            self.indices = self.train_indices
        elif split == 'val':
            self.indices = self.val_indices
        else:
            raise ValueError("Split must be 'train' or 'val'")
        print(f"[INFO] DA Dataset split set to: {split} ({len(self.indices)} samples)")


def plot_comparisons(raw_data, reconstruct, onestep, rollout, time_indices=[0, 3, 6, 9], save_dir="figures"):
    """
    绘制四行四列的对比图和误差图，并保存为 PNG 文件
    参数:
        raw_data, reconstruct, onestep, rollout: torch.Tensor of shape [T, C, H, W]
        time_indices: 选择哪几个时间步进行绘图
        save_dir: 图片保存的文件夹
    """
    os.makedirs(save_dir, exist_ok=True)

    # 转为 numpy
    raw = raw_data
    recon = reconstruct.detach().cpu().numpy()
    one = onestep.detach().cpu().numpy()
    roll = rollout.detach().cpu().numpy()

    titles = ['Raw', 'Reconstruct', 'One-step', 'Rollout']
    datas = [raw, recon, one, roll]

    # ------ 图 1: 对比图 -------
    fig1, axes1 = plt.subplots(nrows=4, ncols=4, figsize=(16, 12))
    for row in range(4):
        for col, t in enumerate(time_indices):
            img = datas[row][t, 0]  # 取第 0 个通道
            ax = axes1[row, col]
            im = ax.imshow(img, cmap='viridis')
            ax.axis('off')
            if row == 0:
                ax.set_title(f"Time {t}")
    fig1.suptitle("Prediction Comparison", fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig1.savefig(os.path.join(save_dir, "comparison.png"))
    plt.close(fig1)

    # ------ 图 2: 误差图 -------
    fig2, axes2 = plt.subplots(nrows=4, ncols=4, figsize=(16, 12))
    for col, t in enumerate(time_indices):
        # raw image
        img_raw = raw[t, 0]
        axes2[0, col].imshow(img_raw, cmap='viridis')
        axes2[0, col].axis('off')
        if col == 0:
            axes2[0, col].set_ylabel('Raw')

        # abs error images
        for i, pred in enumerate([recon, one, roll], start=1):
            err = np.abs(pred[t, 0] - img_raw)
            axes2[i, col].imshow(err, cmap='magma')
            axes2[i, col].axis('off')
            if col == 0:
                axes2[i, col].set_ylabel(f"Error {titles[i]}")
    fig2.suptitle("Error Map (vs. Raw)", fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig2.savefig(os.path.join(save_dir, "error.png"))
    plt.close(fig2)


if __name__ == '__main__':
    foward_step = 12

    data_path = "../../../data/cylinder/cylinder_data.npy"
    
    dynamics_dataset = CylinderDynamicsDataset(data_path=data_path, 
                                              seq_length=foward_step,
                                              normalize=True,
                                              subsample_ratio=1.0)

    sample_idx, t_start = dynamics_dataset.val_indices[4]
    raw_data = dynamics_dataset.raw_data[sample_idx, t_start:t_start+foward_step]

    print(raw_data.shape)
    
    forward_model = CYLINDER_C_FORWARD()
    forward_model.load_state_dict(torch.load('model_weights/forward_model.pt', map_location='cpu'))
    forward_model.C_forward = torch.load('model_weights/C_forward.pt', map_location='cpu')
    forward_model.eval()

    print(forward_model.C_forward.shape)

    state = torch.tensor(raw_data, dtype=torch.float32)
    
    with torch.no_grad():
        z, encode_list = forward_model.K_S(state, return_encode_list=True)
        reconstruct = forward_model.K_S_preimage(z, encode_list)

    # print(reconstruct.shape)

    with torch.no_grad():
        z_current = forward_model.K_S(state)
        print(z_current.shape)
        z_next = forward_model.latent_forward(z_current)
        onestep = forward_model.K_S_preimage(z_next)

    # print(onestep.shape)

    predictions = []
    current_state = state[0, ...].unsqueeze(0)
    n_steps = foward_step
    
    with torch.no_grad():
        for step in range(n_steps):
            z_current = forward_model.K_S(current_state)
            z_next = forward_model.latent_forward(z_current)
            next_state = forward_model.K_S_preimage(z_next)
            
            predictions.append(next_state)
            current_state = next_state
            
    rollout = torch.cat(predictions, dim=0)
    # print(rollout.shape)

    plot_comparisons(raw_data, reconstruct, onestep, rollout)
