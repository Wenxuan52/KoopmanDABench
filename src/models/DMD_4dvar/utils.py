import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import os
import json
from typing import Tuple, Optional, Dict, Any, List, Union
import glob

import sys
current_file_path = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(current_file_path, '..', '..', '..'))
sys.path.insert(0, project_root)

from src.utils.Datasets import (
    BaseDataset,
    DatasetKol,
    DatasetCylinder,
    DatasetCHAP,
)


class SparseObservationGenerator:
    """Generate sparse observation operators for data assimilation"""
    
    def __init__(self, 
                 obs_mode: str = 'fixed',
                 obs_ratio: float = 0.1,
                 obs_pattern: Optional[str] = None,
                 random_seed: int = 42):
        """
        Args:
            obs_mode: 'fixed' for time-independent H, 'random' for time-dependent H
            obs_ratio: Ratio of observed points (0 < ratio <= 1)
            obs_pattern: Pattern for fixed observations ('grid', 'random', 'boundaries')
            random_seed: Random seed for reproducibility
        """
        self.obs_mode = obs_mode
        self.obs_ratio = obs_ratio
        self.obs_pattern = obs_pattern or 'random'
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        
    def generate_observation_operator(self, 
                                    data_shape: Tuple[int, ...], 
                                    time_steps: Optional[int] = None) -> Union[np.ndarray, Dict[int, np.ndarray]]:
        """
        Generate observation operator(s) H
        
        Args:
            data_shape: Shape of a single data frame (e.g., (height, width) for 2D data)
            time_steps: Number of time steps (only used for 'random' mode)
            
        Returns:
            For 'fixed' mode: Single H matrix [n_obs, n_features]
            For 'random' mode: Dictionary {t: H_t} mapping time to observation operators
        """
        n_features = np.prod(data_shape)
        n_obs = max(1, int(n_features * self.obs_ratio))
        
        if self.obs_mode == 'fixed':
            return self._generate_fixed_operator(data_shape, n_features, n_obs)
        elif self.obs_mode == 'random':
            if time_steps is None:
                raise ValueError("time_steps must be provided for random observation mode")
            return self._generate_random_operators(data_shape, n_features, n_obs, time_steps)
        else:
            raise ValueError(f"Unknown observation mode: {self.obs_mode}")
    
    def _generate_fixed_operator(self, data_shape: Tuple[int, ...], 
                               n_features: int, n_obs: int) -> np.ndarray:
        """Generate time-independent observation operator"""
        H = np.zeros((n_obs, n_features))
        
        if self.obs_pattern == 'grid' and len(data_shape) == 2:
            # Regular grid pattern for 2D data
            height, width = data_shape
            obs_per_dim = int(np.sqrt(n_obs))
            h_indices = np.linspace(0, height-1, obs_per_dim, dtype=int)
            w_indices = np.linspace(0, width-1, obs_per_dim, dtype=int)
            
            obs_idx = 0
            for h in h_indices:
                for w in w_indices:
                    if obs_idx < n_obs:
                        flat_idx = h * width + w
                        H[obs_idx, flat_idx] = 1.0
                        obs_idx += 1
                        
        elif self.obs_pattern == 'boundaries' and len(data_shape) == 2:
            # Observe boundaries for 2D data
            height, width = data_shape
            obs_idx = 0
            
            # Top and bottom rows
            for w in range(width):
                if obs_idx < n_obs:
                    H[obs_idx, w] = 1.0  # Top
                    obs_idx += 1
                if obs_idx < n_obs:
                    H[obs_idx, (height-1)*width + w] = 1.0  # Bottom
                    obs_idx += 1
            
            # Left and right columns (excluding corners)
            for h in range(1, height-1):
                if obs_idx < n_obs:
                    H[obs_idx, h*width] = 1.0  # Left
                    obs_idx += 1
                if obs_idx < n_obs:
                    H[obs_idx, h*width + width-1] = 1.0  # Right
                    obs_idx += 1
                    
        else:
            # Random observation points
            obs_indices = self.rng.choice(n_features, size=n_obs, replace=False)
            for i, idx in enumerate(obs_indices):
                H[i, idx] = 1.0
        
        return H
    
    def _generate_random_operators(self, data_shape: Tuple[int, ...], 
                                 n_features: int, n_obs: int, 
                                 time_steps: int) -> Dict[int, np.ndarray]:
        """Generate time-dependent observation operators"""
        H_dict = {}
        
        for t in range(time_steps):
            # Each time step has different random observations
            H_t = np.zeros((n_obs, n_features))
            obs_indices = self.rng.choice(n_features, size=n_obs, replace=False)
            for i, idx in enumerate(obs_indices):
                H_t[i, idx] = 1.0
            H_dict[t] = H_t
            
        return H_dict


class DMD4DVARDataset(Dataset):
    """Dataset wrapper for DMD+4DVAR that provides sparse observations"""
    
    def __init__(self, 
                 base_dataset: BaseDataset,
                 obs_generator: SparseObservationGenerator,
                 sequence_length: int = 10,
                 obs_frequency: int = 1):
        """
        Args:
            base_dataset: Base dataset (DatasetKol, DatasetCylinder, etc.)
            obs_generator: Sparse observation generator
            sequence_length: Length of sequences to extract
            obs_frequency: Frequency of observations (1 = every step, 2 = every other step, etc.)
        """
        self.base_dataset = base_dataset
        self.obs_generator = obs_generator
        self.sequence_length = sequence_length
        self.obs_frequency = obs_frequency
        
        # Get data shape
        if hasattr(base_dataset, 'data'):
            self.data = base_dataset.data
            self.n_samples, self.n_time_steps = self.data.shape[0], self.data.shape[1]
            self.data_shape = self.data.shape[2:]  # Shape of single frame
        else:
            raise ValueError("Base dataset must have 'data' attribute")
        
        # Generate observation operators
        obs_times = list(range(0, sequence_length, obs_frequency))
        self.H_operators = obs_generator.generate_observation_operator(
            self.data_shape, 
            time_steps=len(obs_times) if obs_generator.obs_mode == 'random' else None
        )
        
        # Store observation times relative to sequence start
        self.obs_times = obs_times
        
    def __len__(self):
        # Number of valid sequences we can extract
        return self.n_samples * (self.n_time_steps - self.sequence_length)
    
    def __getitem__(self, idx):
        # Convert linear index to sample and time indices
        sample_idx = idx // (self.n_time_steps - self.sequence_length)
        time_idx = idx % (self.n_time_steps - self.sequence_length)
        
        # Extract sequence
        sequence = self.data[sample_idx, time_idx:time_idx + self.sequence_length]
        
        # Flatten spatial dimensions
        sequence_flat = sequence.reshape(self.sequence_length, -1)
        
        # Generate observations
        observations = {}
        if self.obs_generator.obs_mode == 'fixed':
            # Use same H for all observation times
            H = self.H_operators
            for t in self.obs_times:
                observations[t] = H @ sequence_flat[t]
        else:
            # Use different H_t for each observation time
            for i, t in enumerate(self.obs_times):
                H_t = self.H_operators[i]
                observations[t] = H_t @ sequence_flat[t]
        
        # Return data
        return {
            'sequence': sequence,
            'sequence_flat': sequence_flat,
            'initial_condition': sequence_flat[0],
            'observations': observations,
            'sample_idx': sample_idx,
            'time_idx': time_idx
        }
    
    def get_observation_operators(self) -> Union[np.ndarray, Dict[int, np.ndarray]]:
        """Get observation operators for use in assimilation"""
        if self.obs_generator.obs_mode == 'fixed':
            # Return same operator for all observation times
            return {t: self.H_operators for t in self.obs_times}
        else:
            # Return time-specific operators
            return {t: self.H_operators[i] for i, t in enumerate(self.obs_times)}


class DMD4DVARDataLoader:
    """DataLoader for DMD+4DVAR models with sparse observations"""
    
    def __init__(self, 
                 dataset_name: str, 
                 data_path: str,
                 obs_mode: str = 'fixed',
                 obs_ratio: float = 0.1,
                 obs_pattern: Optional[str] = None,
                 normalize: bool = True,
                 train_ratio: float = 0.8,
                 random_seed: int = 42,
                 sequence_length: int = 10,
                 obs_frequency: int = 1,
                 **dataset_kwargs):
        """
        Args:
            dataset_name: Name of dataset ("kolmogorov", "cylinder", "chap")
            data_path: Root path to data folder
            obs_mode: 'fixed' or 'random' sparse observation mode
            obs_ratio: Ratio of observed points
            obs_pattern: Pattern for fixed observations ('grid', 'random', 'boundaries')
            normalize: Whether to normalize data
            train_ratio: Ratio for train/val split
            random_seed: Random seed
            sequence_length: Length of sequences to extract
            obs_frequency: Frequency of observations
            **dataset_kwargs: Additional arguments for specific datasets
        """
        self.dataset_name = dataset_name.lower()
        self.data_path = data_path
        self.obs_mode = obs_mode
        self.obs_ratio = obs_ratio
        self.obs_pattern = obs_pattern
        self.normalize = normalize
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        self.sequence_length = sequence_length
        self.obs_frequency = obs_frequency
        self.dataset_kwargs = dataset_kwargs
        
        # Create base datasets
        self.train_base, self.val_base = self._create_base_datasets()
        
        # Create observation generators
        self.train_obs_gen = SparseObservationGenerator(
            obs_mode=obs_mode,
            obs_ratio=obs_ratio,
            obs_pattern=obs_pattern,
            random_seed=random_seed
        )
        
        # Use different seed for validation to get different observation patterns
        self.val_obs_gen = SparseObservationGenerator(
            obs_mode=obs_mode,
            obs_ratio=obs_ratio,
            obs_pattern=obs_pattern,
            random_seed=random_seed + 1000
        )
        
        # Create DMD+4DVAR datasets
        self.train_dataset = DMD4DVARDataset(
            self.train_base,
            self.train_obs_gen,
            sequence_length=sequence_length,
            obs_frequency=obs_frequency
        )
        
        self.val_dataset = DMD4DVARDataset(
            self.val_base,
            self.val_obs_gen,
            sequence_length=sequence_length,
            obs_frequency=obs_frequency
        )
    
    def _create_base_datasets(self) -> Tuple[BaseDataset, BaseDataset]:
        """Create train and validation base datasets"""
        dataset_map = {
            "kolmogorov": (DatasetKol, "kolmogorov"),
            "cylinder": (DatasetCylinder, "Cylinder"),
            "chap": (DatasetCHAP, "CHAP")
        }
        
        if self.dataset_name not in dataset_map:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        DatasetClass, folder_name = dataset_map[self.dataset_name]
        full_path = os.path.join(self.data_path, folder_name)
        
        # Create train dataset
        train_dataset = DatasetClass(
            data_path=full_path,
            normalize=self.normalize,
            train=True,
            train_ratio=self.train_ratio,
            random_seed=self.random_seed,
            **self.dataset_kwargs
        )
        
        # Create validation dataset
        val_dataset = DatasetClass(
            data_path=full_path,
            normalize=self.normalize,
            train=False,
            train_ratio=self.train_ratio,
            random_seed=self.random_seed,
            **self.dataset_kwargs
        )
        
        # Share normalization parameters from train to val
        if self.normalize and hasattr(train_dataset, 'mean'):
            val_dataset.set_normalization_params(train_dataset.mean, train_dataset.std)
        
        return train_dataset, val_dataset
    
    def get_dataloaders(self, batch_size: int = 32, 
                       num_workers: int = 4,
                       shuffle_train: bool = True) -> Tuple[DataLoader, DataLoader]:
        """Create PyTorch DataLoaders"""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def get_data_stats(self) -> Dict[str, Any]:
        """Get statistics about the dataset"""
        stats = {
            "dataset_name": self.dataset_name,
            "train_size": len(self.train_dataset),
            "val_size": len(self.val_dataset),
            "train_base_shape": self.train_base.data.shape if hasattr(self.train_base, 'data') else None,
            "val_base_shape": self.val_base.data.shape if hasattr(self.val_base, 'data') else None,
            "normalized": self.normalize,
            "observation_mode": self.obs_mode,
            "observation_ratio": self.obs_ratio,
            "observation_pattern": self.obs_pattern if self.obs_mode == 'fixed' else None,
            "sequence_length": self.sequence_length,
            "observation_frequency": self.obs_frequency,
            "observation_times": self.train_dataset.obs_times
        }
        
        # Add observation operator info
        if self.obs_mode == 'fixed':
            H = self.train_dataset.H_operators
            stats["observation_operator_shape"] = H.shape
            stats["num_observations_per_time"] = H.shape[0]
        else:
            stats["num_observations_per_time"] = list(self.train_dataset.H_operators.values())[0].shape[0]
            
        return stats
    
    def get_normalization_params(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get normalization parameters from training dataset"""
        if hasattr(self.train_base, 'mean') and hasattr(self.train_base, 'std'):
            return self.train_base.mean, self.train_base.std
        return None, None


# Utility functions for creating synthetic observations
def add_observation_noise(observations: Dict[int, np.ndarray], 
                         noise_level: float = 0.01,
                         noise_type: str = 'gaussian') -> Dict[int, np.ndarray]:
    """Add noise to observations"""
    noisy_obs = {}
    
    for t, obs in observations.items():
        if noise_type == 'gaussian':
            noise = np.random.normal(0, noise_level, obs.shape)
        elif noise_type == 'uniform':
            noise = np.random.uniform(-noise_level, noise_level, obs.shape)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        noisy_obs[t] = obs + noise
    
    return noisy_obs


def visualize_observation_pattern(H: np.ndarray, data_shape: Tuple[int, ...], 
                                 save_path: Optional[str] = None):
    """Visualize observation pattern for 2D data"""
    if len(data_shape) != 2:
        print("Visualization only supported for 2D data")
        return
    
    import matplotlib.pyplot as plt
    
    # Create observation mask
    mask = np.zeros(np.prod(data_shape))
    for i in range(H.shape[0]):
        obs_idx = np.where(H[i, :] > 0)[0]
        if len(obs_idx) > 0:
            mask[obs_idx[0]] = 1
    
    # Reshape to 2D
    mask_2d = mask.reshape(data_shape)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(mask_2d, cmap='binary', interpolation='nearest')
    plt.colorbar(label='Observed (1) / Unobserved (0)')
    plt.title(f'Observation Pattern ({np.sum(mask)} points, {np.sum(mask)/mask.size*100:.1f}%)')
    plt.xlabel('Width')
    plt.ylabel('Height')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


# Example usage
if __name__ == "__main__":
    # Test with Kolmogorov dataset
    print("Testing DMD+4DVAR DataLoader with Kolmogorov dataset...")
    
    # Fixed observation mode
    loader_fixed = DMD4DVARDataLoader(
        dataset_name="cylinder",
        data_path="./data",
        obs_mode='fixed',
        obs_ratio=0.2,
        obs_pattern='grid',
        sequence_length=20,
        obs_frequency=2
    )
    
    print("\nFixed observation mode stats:")
    print(loader_fixed.get_data_stats())
    
    # Random observation mode
    loader_random = DMD4DVARDataLoader(
        dataset_name="kolmogorov",
        data_path="./data",
        obs_mode='random',
        obs_ratio=0.1,
        sequence_length=20,
        obs_frequency=3
    )
    
    print("\nRandom observation mode stats:")
    print(loader_random.get_data_stats())
    
    # Get dataloaders
    train_loader, val_loader = loader_fixed.get_dataloaders(batch_size=4)
    
    # Test iteration
    print("\nTesting data iteration...")
    for i, batch in enumerate(train_loader):
        if i == 0:
            print(f"Batch keys: {batch.keys()}")
            print(f"Sequence shape: {batch['sequence'].shape}")
            print(f"Initial condition shape: {batch['initial_condition'].shape}")
            print(f"Observation times: {list(batch['observations'].keys())}")
            
            # Check observation shapes
            for t, obs in batch['observations'].items():
                print(f"Observations at time {t}: shape {obs.shape}")
            break
    
    # Visualize observation pattern (if fixed mode)
    if loader_fixed.obs_mode == 'fixed':
        print("\nVisualizing observation pattern...")
        H = loader_fixed.train_dataset.H_operators
        data_shape = loader_fixed.train_dataset.data_shape
        if len(data_shape) == 2:
            visualize_observation_pattern(H, data_shape, save_path='temp.png')