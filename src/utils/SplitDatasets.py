import numpy as np
import torch
from torch.utils.data import Dataset
import os
import json
from typing import Tuple, Optional, Dict, Any, List
import glob


class BaseDataset(Dataset):
    """Base dataset class for DMD models"""
    
    def __init__(self, data_path: str, normalize: bool = True, 
                 train_ratio: float = 0.8, random_seed: int = 42):
        self.data_path = data_path
        self.normalize = normalize
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        
        # To be filled by child classes
        self.data = None
        self.train_data = None
        self.val_data = None
        self.mean = None
        self.std = None
        
        # Load data
        self._load_data()
        
        # Split train/val
        self._split_data()
        
        # Normalize if needed
        if self.normalize:
            self._normalize_data()
    
    def _load_data(self):
        """To be implemented by child classes"""
        raise NotImplementedError
    
    def _split_data(self):
        """Split data into train/val sets sequentially"""
        n_samples = len(self.data)
        train_size = int(n_samples * self.train_ratio)

        sample_idx = [i for i in range(n_samples)]
        np.random.seed(self.random_seed)
        np.random.shuffle(sample_idx)

        print(sample_idx)
        
        train_idx = sample_idx[:train_size]
        val_idx = sample_idx[train_size:]

        self.train_data = self.data[train_idx]
        self.val_data = self.data[val_idx]
    
    def _normalize_data(self):
        """Normalize data using mean and std from training set"""
        # Calculate mean and std from training data
        # Handle both 3D (samples, time, features) and 4D (samples, time, channels, features) data
        if self.data.ndim == 4:
            self.mean = np.mean(self.train_data, axis=(0, 1, 2, 3), keepdims=True)
            self.std = np.std(self.train_data, axis=(0, 1, 2, 3), keepdims=True)
        elif self.data.ndim == 5:
            # For 4D data: normalize per channel
            self.mean = np.mean(self.train_data, axis=(0, 1, 3, 4), keepdims=True)
            self.std = np.std(self.train_data, axis=(0, 1, 3, 4), keepdims=True)
        # Avoid division by zero
        self.std[self.std < 1e-8] = 1.0
        
        # Apply normalization
        if self.mean is not None and self.std is not None:
            self.train_data = (self.train_data - self.mean) / self.std
            self.val_data = (self.val_data - self.mean) / self.std
    
    def set_normalization_params(self, mean: np.ndarray, std: np.ndarray):
        """Set normalization parameters (for validation set)"""
        self.mean = mean
        self.std = std
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        return sample


class DatasetKol(BaseDataset):
    """Dataset for Kolmogorov flow data"""
    
    def _load_data(self):
        """Load Kolmogorov dataset"""
        file_path = os.path.join(self.data_path)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Kolmogorov dataset not found at {file_path}")
        
        # Load data: [120, 320, 64, 64]
        data = np.load(file_path)
        print(f"Loaded Kolmogorov data with shape: {data.shape}")
        
        # Flatten spatial dimensions: 64x64 -> 4096
        self.data = data


class DatasetCylinder(BaseDataset):
    """Dataset for Cylinder flow data"""
    
    def _load_data(self):
        """Load Cylinder dataset"""
        file_path = os.path.join(self.data_path, "cylinder_data.npy")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Cylinder dataset not found at {file_path}")
        
        data = np.load(file_path)
        print(f"Loaded Cylinder data with shape: {data.shape}")
        
        self.data = data
    
    # def _split_data(self):
    #     """Split data into train/val sets using evenly spaced sampling for val"""
    #     n_samples = len(self.data)
    #     val_size = int(n_samples * (1 - self.train_ratio))
        
    #     # Compute validation indices: start from 0, step evenly to cover val_size samples
    #     step = n_samples // val_size
    #     val_idx = list(range(0, n_samples, step))[:val_size]

    #     val_idx.append(n_samples - 1)

    #     # Compute training indices: all others not in val_idx
    #     val_idx_set = set(val_idx)
    #     train_idx = [i for i in range(n_samples) if i not in val_idx_set]

    #     print(f"Train indices: {train_idx}")
    #     print(f"Val indices: {val_idx}")

    #     self.train_data = self.data[train_idx]
    #     self.val_data = self.data[val_idx]


class DatasetDam(BaseDataset):
    """Dataset for Dam flow data"""

    def _load_data(self):
        """Load Dam dataset"""
        file_path = os.path.join(self.data_path, "dam_data.npy")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dam dataset not found at {file_path}")
        
        data = np.load(file_path)
        print(f"Loaded Dam data with shape: {data.shape}")
        
        self.data = data

    def _split_data(self):
        """Split data into train/val sets using evenly spaced sampling for val"""
        n_samples = len(self.data)
        val_size = int(n_samples * (1 - self.train_ratio))
        
        # Compute validation indices: start from 0, step evenly to cover val_size samples
        step = n_samples // val_size + 1
        val_idx = list(range(0, n_samples, step))[:val_size]

        # Compute training indices: all others not in val_idx
        val_idx_set = set(val_idx)
        train_idx = [i for i in range(n_samples) if i not in val_idx_set]

        print(f"Train indices: {train_idx}")
        print(f"Val indices: {val_idx}")

        self.train_data = self.data[train_idx]
        self.val_data = self.data[val_idx]

def denormalize_data(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return data * std + mean

if __name__ == "__main__":
    koldata = DatasetKol("data/kol/kf_2d_re1000_T20.0_data.npy", normalize=False, train_ratio=0.8, random_seed=42)
    print(koldata.data.shape)
    
    print(koldata.mean)
    print(koldata.std)
    
    print(koldata.train_data.shape)
    print(koldata.val_data.shape)

    print(koldata.train_data.min())
    print(koldata.train_data.max())

    print(koldata.val_data.min())
    print(koldata.val_data.max())

    # de_train_data = denormalize_data(koldata.train_data, koldata.mean, koldata.std)
    # de_val_data = denormalize_data(koldata.val_data, koldata.mean, koldata.std)
    # print(de_train_data.min())
    # print(de_train_data.max())
    # print(de_val_data.min())
    # print(de_val_data.max())

    # temp_train = koldata.train_data[:, 500:, ...]
    # temp_val = koldata.val_data[:, 500:, ...]

    # print(temp_train.shape)
    # print(temp_val.shape)

    np.save("data/kol/kolmogorov_train_data.npy", koldata.train_data)
    np.save("data/kol/kolmogorov_val_data.npy", koldata.val_data)

    # cylinderdata = DatasetCylinder("data/cylinder", normalize=False, train_ratio=0.8, random_seed=42)
    # print(cylinderdata.mean)
    # print(cylinderdata.std)

    # print(cylinderdata.train_data.shape)
    # print(cylinderdata.val_data.shape)

    # print(cylinderdata.train_data.min())
    # print(cylinderdata.train_data.max())

    # print(cylinderdata.val_data.min())
    # print(cylinderdata.val_data.max())

    # # de_train_data = denormalize_data(cylinderdata.train_data, cylinderdata.mean, cylinderdata.std)
    # # de_val_data = denormalize_data(cylinderdata.val_data, cylinderdata.mean, cylinderdata.std)
    # # print(de_train_data.min())
    # # print(de_train_data.max())
    # # print(de_val_data.min())
    # # print(de_val_data.max())

    # # temp_train = cylinderdata.train_data[:, 500:, ...]
    # # temp_val = cylinderdata.val_data[:, 500:, ...]

    # np.save("data/cylinder/cylinder_train_data.npy", cylinderdata.train_data)
    # np.save("data/cylinder/cylinder_val_data.npy", cylinderdata.val_data)

    # damdata = DatasetDam("data/dam", normalize=False, train_ratio=0.8, random_seed=42)
    # print(damdata.mean)
    # print(damdata.std)

    # print(damdata.train_data.shape)
    # print(damdata.val_data.shape)

    # print(damdata.train_data.min())
    # print(damdata.train_data.max())

    # print(damdata.val_data.min())
    # print(damdata.val_data.max())

    # # de_train_data = denormalize_data(damdata.train_data, damdata.mean, damdata.std)
    # # de_val_data = denormalize_data(damdata.val_data, damdata.mean, damdata.std)
    # # print(de_train_data.min())
    # # print(de_train_data.max())
    # # print(de_val_data.min())
    # # print(de_val_data.max())

    # np.save("data/dam/dam_train_data.npy", damdata.train_data)
    # np.save("data/dam/dam_val_data.npy", damdata.val_data)
