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


class DatasetCHAP(BaseDataset):
    """Dataset for CHAP multi-physics data"""
    
    def __init__(self, data_path: str, chemical: str = "Cl", normalize: bool = True, 
                 train: bool = True, train_ratio: float = 0.8, random_seed: int = 42):
        """
        Args:
            chemical: Which chemical to use ("Cl", "NH4", "NO3", "SO4")
        """
        self.chemical = chemical
        super().__init__(data_path, normalize, train, train_ratio, random_seed)
    
    def _load_data(self):
        """Load CHAP dataset for specified chemical"""
        chem_path = os.path.join(self.data_path, self.chemical)
        
        if not os.path.exists(chem_path):
            raise FileNotFoundError(f"Chemical folder not found at {chem_path}")
        
        # Load observation data
        obs_path = os.path.join(chem_path, f"CHAP_{self.chemical}_observation.npy")
        lat_path = os.path.join(chem_path, f"CHAP_{self.chemical}_lat.npy")
        lon_path = os.path.join(chem_path, f"CHAP_{self.chemical}_lon.npy")
        
        if not all(os.path.exists(p) for p in [obs_path, lat_path, lon_path]):
            raise FileNotFoundError(f"Missing data files for {self.chemical}")
        
        # Load data
        observations = np.load(obs_path)  # [8, 365, lat_dim, lon_dim]
        print(f"Loaded CHAP {self.chemical} data with shape: {observations.shape}")
        
        # Reshape: [years, days, lat*lon] -> [years, days, features]
        n_years, n_days, lat_dim, lon_dim = observations.shape
        observations_flat = observations.reshape(n_years, n_days, -1)
        
        # Handle NaN values
        observations_flat = np.nan_to_num(observations_flat, nan=0.0)
        
        self.data = observations_flat
        print(f"Reshaped data to: {self.data.shape}")


def denormalize_data(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return data * std + mean

if __name__ == "__main__":
    koldata = DatasetKol("data/kolmogorov/RE450_n4/kf_2d_re450_n4_data.npy", normalize=False, train_ratio=0.8, random_seed=42)
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

    np.save("data/kolmogorov/RE450_n4/kolmogorov_train_data.npy", koldata.train_data)
    np.save("data/kolmogorov/RE450_n4/kolmogorov_val_data.npy", koldata.val_data)

    # cylinderdata = DatasetCylinder("data/cylinder", normalize=True, train_ratio=0.8, random_seed=42)
    # print(cylinderdata.mean)
    # print(cylinderdata.std)

    # print(cylinderdata.train_data.shape)
    # print(cylinderdata.val_data.shape)

    # print(cylinderdata.train_data.min())
    # print(cylinderdata.train_data.max())

    # print(cylinderdata.val_data.min())
    # print(cylinderdata.val_data.max())

    # de_train_data = denormalize_data(cylinderdata.train_data, cylinderdata.mean, cylinderdata.std)
    # de_val_data = denormalize_data(cylinderdata.val_data, cylinderdata.mean, cylinderdata.std)
    # print(de_train_data.min())
    # print(de_train_data.max())
    # print(de_val_data.min())
    # print(de_val_data.max())
