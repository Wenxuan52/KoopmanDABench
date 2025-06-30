import numpy as np
import torch
from torch.utils.data import Dataset
import os
import json
from typing import Tuple, Optional, Dict, Any, List
import glob


class BaseDataset(Dataset):
    """Base dataset class for DMD models"""
    
    def __init__(self, data_path: str, normalize: bool = True, train: bool = True, 
                 train_ratio: float = 0.8, random_seed: int = 42):
        """
        Args:
            data_path: Path to the dataset
            normalize: Whether to normalize the data
            train: Whether this is training set or validation set
            train_ratio: Ratio of training data
            random_seed: Random seed for train/val split
        """
        self.data_path = data_path
        self.normalize = normalize
        self.train = train
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        
        # To be filled by child classes
        self.data = None
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
        """Split data into train/val sets"""
        np.random.seed(self.random_seed)
        n_samples = len(self.data)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        train_size = int(n_samples * self.train_ratio)
        if self.train:
            self.data = self.data[indices[:train_size]]
        else:
            self.data = self.data[indices[train_size:]]
    
    def _normalize_data(self):
        """Normalize data using mean and std from training set"""
        if self.train:
            # Calculate mean and std from training data
            # Handle both 3D (samples, time, features) and 4D (samples, time, channels, features) data
            if self.data.ndim == 3:
                self.mean = np.mean(self.data, axis=(0, 1), keepdims=True)
                self.std = np.std(self.data, axis=(0, 1), keepdims=True)
            elif self.data.ndim == 4:
                # For 4D data: normalize per channel
                self.mean = np.mean(self.data, axis=(0, 1, 3), keepdims=True)
                self.std = np.std(self.data, axis=(0, 1, 3), keepdims=True)
            # Avoid division by zero
            self.std[self.std < 1e-8] = 1.0
        
        # Apply normalization
        if self.mean is not None and self.std is not None:
            self.data = (self.data - self.mean) / self.std
    
    def set_normalization_params(self, mean: np.ndarray, std: np.ndarray):
        """Set normalization parameters (for validation set)"""
        self.mean = mean
        self.std = std
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Return a sample for DMD: (x_t, x_{t+1})"""
        sample = self.data[idx]
        # For DMD, we need consecutive time steps
        if sample.shape[0] < 2:
            raise ValueError("Time series must have at least 2 time steps")
        
        # Random time step selection
        t = np.random.randint(0, sample.shape[0] - 1)
        x_t = sample[t]
        x_next = sample[t + 1]
        
        # Ensure proper shape
        # For standard dataset (3D): sample is [time, features]
        # For Cylinder dataset (4D): sample is [time, channels, features]
        
        return torch.FloatTensor(x_t), torch.FloatTensor(x_next)


class DatasetKol(BaseDataset):
    """Dataset for Kolmogorov flow data"""
    
    def _load_data(self):
        """Load Kolmogorov dataset"""
        file_path = os.path.join(self.data_path, "RE1000", "kf_2d_re1000_64_120seed.npy")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Kolmogorov dataset not found at {file_path}")
        
        # Load data: [120, 320, 64, 64]
        data = np.load(file_path)
        print(f"Loaded Kolmogorov data with shape: {data.shape}")
        
        # Reshape to [samples, time, features]
        # Flatten spatial dimensions: 64x64 -> 4096
        self.data = data.reshape(data.shape[0], data.shape[1], -1)
        print(f"Reshaped data to: {self.data.shape}")


class DatasetCylinder(BaseDataset):
    """Dataset for Cylinder flow data"""
    
    def __init__(self, data_path: str, normalize: bool = True, train: bool = True, 
                 train_ratio: float = 0.8, random_seed: int = 42,
                 target_resolution: Optional[Tuple[int, int]] = None,
                 interpolation_mode: str = 'bilinear'):
        """
        Args:
            target_resolution: Target resolution (H, W) for interpolation. If None, no interpolation
            interpolation_mode: Interpolation mode ('bilinear', 'bicubic', 'nearest')
        """
        self.target_resolution = target_resolution
        self.interpolation_mode = interpolation_mode
        super().__init__(data_path, normalize, train, train_ratio, random_seed)
    
    def _load_data(self):
        """Load Cylinder dataset"""
        case_folders = sorted(glob.glob(os.path.join(self.data_path, "case*")))
        
        if not case_folders:
            raise FileNotFoundError(f"No case folders found in {self.data_path}")
        
        all_data = []
        all_metadata = []
        min_time_steps = float('inf')
        
        # First pass: load all data and find minimum time steps
        temp_data = []
        for case_folder in case_folders:
            # Load u and v components
            u_path = os.path.join(case_folder, "u.npy")
            v_path = os.path.join(case_folder, "v.npy")
            json_path = os.path.join(case_folder, "case.json")
            
            if not os.path.exists(u_path) or not os.path.exists(v_path):
                print(f"Warning: Missing data in {case_folder}, skipping...")
                continue
            
            u = np.load(u_path)  # [time, height, width]
            v = np.load(v_path)  # [time, height, width]
            
            # Load metadata if available
            metadata = {}
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
            
            # Check original resolution
            # print(f"Case {os.path.basename(case_folder)}: u shape = {u.shape}, v shape = {v.shape}")
            
            # Stack u and v as channels: [time, 2, height, width]
            uv = np.stack([u, v], axis=1)
            
            # Apply interpolation if target resolution is specified
            if self.target_resolution is not None:
                uv = self._interpolate_data(uv, self.target_resolution)
            
            # Reshape to [time, channels, height*width]
            # Keep channels separate: [time, 2, spatial_features]
            n_time, n_channels, h, w = uv.shape
            uv_reshaped = uv.reshape(n_time, n_channels, h * w)
            
            temp_data.append(uv_reshaped)
            all_metadata.append(metadata)
            min_time_steps = min(min_time_steps, uv_reshaped.shape[0])
        
        if not temp_data:
            raise ValueError("No valid data found in Cylinder dataset")
        
        print(f"Minimum time steps across all cases: {min_time_steps}")
        
        # Second pass: truncate all sequences to minimum length
        for i, data in enumerate(temp_data):
            # Take only the first min_time_steps
            truncated_data = data[:min_time_steps]
            all_data.append(truncated_data)
        
        # Now we can safely stack all cases: [cases, time, features]
        self.data = np.array(all_data)
        self.metadata = all_metadata
        
        print(f"Loaded Cylinder data from {len(all_data)} cases")
        print(f"Final shape: {self.data.shape}")
        if self.target_resolution:
            print(f"Interpolated to resolution: {self.target_resolution}")
    
    def _interpolate_data(self, data: np.ndarray, target_resolution: Tuple[int, int]) -> np.ndarray:
        """Interpolate data to target resolution
        
        Args:
            data: Input data [time, channels, height, width]
            target_resolution: Target (height, width)
        
        Returns:
            Interpolated data
        """
        import torch
        import torch.nn.functional as F
        
        # Convert to torch tensor
        data_torch = torch.from_numpy(data).float()
        
        # Reshape for interpolation: [time*channels, 1, height, width]
        n_time, n_channels, h, w = data_torch.shape
        data_flat = data_torch.reshape(n_time * n_channels, 1, h, w)
        
        # Interpolate
        data_interp = F.interpolate(
            data_flat, 
            size=target_resolution,
            mode=self.interpolation_mode,
            align_corners=False if self.interpolation_mode in ['bilinear', 'bicubic'] else None
        )
        
        # Reshape back: [time, channels, new_height, new_width]
        new_h, new_w = target_resolution
        data_interp = data_interp.reshape(n_time, n_channels, new_h, new_w)
        
        return data_interp.numpy()


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