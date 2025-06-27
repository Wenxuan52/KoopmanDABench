import torch
from torch.utils.data import DataLoader
import numpy as np
from typing import Optional, Tuple, Dict, Any
import os

import sys

current_file_path = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(current_file_path, '..', '..', '..'))
sys.path.insert(0, project_root)

from src.utils.Datasets import (
    DatasetKol,
    DatasetCylinder,
    DatasetCHAP
)


class CAEMLPDataLoader:
    """Base DataLoader class for CAE-MLP models"""
    
    def __init__(self, 
                 dataset_name: str,
                 data_path: str,
                 batch_size: int = 32,
                 train_ratio: float = 0.8,
                 num_workers: int = 4,
                 shuffle_train: bool = True,
                 shuffle_val: bool = False,
                 drop_last: bool = True,
                 pin_memory: bool = True,
                 normalize: bool = True,
                 random_seed: int = 42,
                 **dataset_kwargs):
        """
        Initialize CAE-MLP DataLoader
        
        Args:
            dataset_name: Name of dataset ('kolmogorov', 'cylinder', 'chap')
            data_path: Root path to data directory
            batch_size: Batch size for training
            train_ratio: Ratio of training data
            num_workers: Number of workers for parallel data loading
            shuffle_train: Whether to shuffle training data
            shuffle_val: Whether to shuffle validation data
            drop_last: Whether to drop last incomplete batch
            pin_memory: Whether to pin memory for faster GPU transfer
            normalize: Whether to normalize data
            random_seed: Random seed for reproducibility
            **dataset_kwargs: Additional arguments for specific datasets
        """
        self.dataset_name = dataset_name.lower()
        self.data_path = data_path
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.normalize = normalize
        self.random_seed = random_seed
        self.dataset_kwargs = dataset_kwargs
        
        # Initialize datasets
        self._init_datasets()
        
        # Create data loaders
        self._create_loaders()
    
    def _init_datasets(self):
        """Initialize train and validation datasets based on dataset name"""
        # Get dataset class and path
        if self.dataset_name == 'kolmogorov':
            dataset_class = DatasetKol
            dataset_path = os.path.join(self.data_path, "kolmogorov")
        elif self.dataset_name == 'cylinder':
            dataset_class = DatasetCylinder
            dataset_path = os.path.join(self.data_path, "Cylinder")
        elif self.dataset_name == 'chap':
            dataset_class = DatasetCHAP
            dataset_path = os.path.join(self.data_path, "CHAP")
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        # Create training dataset
        self.train_dataset = dataset_class(
            data_path=dataset_path,
            normalize=self.normalize,
            train=True,
            train_ratio=self.train_ratio,
            random_seed=self.random_seed,
            **self.dataset_kwargs
        )
        
        # Create validation dataset
        self.val_dataset = dataset_class(
            data_path=dataset_path,
            normalize=self.normalize,
            train=False,
            train_ratio=self.train_ratio,
            random_seed=self.random_seed,
            **self.dataset_kwargs
        )
        
        # Share normalization parameters from train to val
        if self.normalize and hasattr(self.train_dataset, 'mean') and hasattr(self.train_dataset, 'std'):
            self.val_dataset.set_normalization_params(
                self.train_dataset.mean,
                self.train_dataset.std
            )
    
    def _create_loaders(self):
        """Create PyTorch DataLoaders for train and validation sets"""
        # Training loader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )
        
        # Validation loader
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_val,
            num_workers=self.num_workers,
            drop_last=False,  # Don't drop last batch for validation
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )
    
    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Get train and validation loaders"""
        return self.train_loader, self.val_loader
    
    def get_data_shape(self) -> Dict[str, Any]:
        """Get information about data shapes"""
        # Get a sample batch
        x_t, x_next = next(iter(self.train_loader))
        
        info = {
            "batch_shape": x_t.shape,
            "input_dim": x_t.shape[-1],
            "dataset_size_train": len(self.train_dataset),
            "dataset_size_val": len(self.val_dataset),
            "num_batches_train": len(self.train_loader),
            "num_batches_val": len(self.val_loader)
        }
        
        # Add dataset-specific information
        if self.dataset_name == 'cylinder':
            # Cylinder data has 2 channels (u, v)
            info["num_channels"] = 2
            if hasattr(self.train_dataset, 'target_resolution') and self.train_dataset.target_resolution:
                info["spatial_resolution"] = self.train_dataset.target_resolution
        elif self.dataset_name == 'kolmogorov':
            # Kolmogorov is single channel
            info["num_channels"] = 1
            info["spatial_resolution"] = (64, 64)
        elif self.dataset_name == 'chap':
            # CHAP is single channel per chemical
            info["num_channels"] = 1
            info["chemical"] = self.dataset_kwargs.get('chemical', 'Cl')
        
        return info
    
    def get_normalization_params(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get normalization parameters (mean, std) from training dataset"""
        if hasattr(self.train_dataset, 'mean') and hasattr(self.train_dataset, 'std'):
            return self.train_dataset.mean, self.train_dataset.std
        return None, None


def create_caemlp_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """
    Convenience function to create CAE-MLP dataloaders from a configuration dictionary
    
    Args:
        config: Configuration dictionary containing:
            - dataset_name: Name of the dataset
            - data_path: Path to data directory
            - batch_size: Batch size
            - train_ratio: Train/val split ratio
            - num_workers: Number of data loading workers
            - normalize: Whether to normalize data
            - And other CAEMLPDataLoader parameters
    
    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        data_info: Dictionary with data shape information
    """
    # Extract dataset-specific kwargs
    dataset_kwargs = {}
    
    if config['dataset_name'].lower() == 'cylinder':
        # Cylinder-specific parameters
        if 'target_resolution' in config:
            dataset_kwargs['target_resolution'] = config['target_resolution']
        if 'interpolation_mode' in config:
            dataset_kwargs['interpolation_mode'] = config['interpolation_mode']
    
    elif config['dataset_name'].lower() == 'chap':
        # CHAP-specific parameters
        if 'chemical' in config:
            dataset_kwargs['chemical'] = config['chemical']
    
    # Create dataloader
    dataloader = CAEMLPDataLoader(
        dataset_name=config['dataset_name'],
        data_path=config['data_path'],
        batch_size=config.get('batch_size', 32),
        train_ratio=config.get('train_ratio', 0.8),
        num_workers=config.get('num_workers', 4),
        shuffle_train=config.get('shuffle_train', True),
        shuffle_val=config.get('shuffle_val', False),
        drop_last=config.get('drop_last', True),
        pin_memory=config.get('pin_memory', True),
        normalize=config.get('normalize', True),
        random_seed=config.get('random_seed', 42),
        **dataset_kwargs
    )
    
    # Get loaders and info
    train_loader, val_loader = dataloader.get_loaders()
    data_info = dataloader.get_data_shape()
    
    # Add normalization parameters to info
    mean, std = dataloader.get_normalization_params()
    data_info['normalization'] = {
        'enabled': config.get('normalize', True),
        'mean': mean,
        'std': std
    }
    
    return train_loader, val_loader, data_info


class CAEMLPLoss:
    """Loss functions for CAE-MLP models"""
    
    def __init__(self, recon_weight: float = 1.0, pred_weight: float = 1.0,
                 latent_weight: float = 0.1, linear_weight: float = 0.1):
        """
        Initialize loss function
        
        Args:
            recon_weight: Weight for reconstruction loss
            pred_weight: Weight for prediction loss
            latent_weight: Weight for latent space regularization
            linear_weight: Weight for linear constraint (weak linear model only)
        """
        self.recon_weight = recon_weight
        self.pred_weight = pred_weight
        self.latent_weight = latent_weight
        self.linear_weight = linear_weight
        
        # Loss functions
        self.mse_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
    
    def reconstruction_loss(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        """Calculate reconstruction loss"""
        return self.mse_loss(x_recon, x)
    
    def prediction_loss(self, x_next: torch.Tensor, x_next_pred: torch.Tensor) -> torch.Tensor:
        """Calculate prediction loss"""
        return self.mse_loss(x_next_pred, x_next)
    
    def latent_regularization(self, z: torch.Tensor) -> torch.Tensor:
        """L2 regularization on latent codes to prevent explosion"""
        return torch.mean(z.pow(2))
    
    def linear_constraint_loss(self, z_next_nonlinear: torch.Tensor, 
                             z_next_linear: torch.Tensor) -> torch.Tensor:
        """Weak linear constraint loss - encourages nonlinear prediction to be close to linear"""
        return self.mse_loss(z_next_nonlinear, z_next_linear.detach())
    
    def compute_cae_linear_loss(self, x_t: torch.Tensor, x_next: torch.Tensor,
                               x_t_recon: torch.Tensor, x_next_pred: torch.Tensor,
                               z_t: torch.Tensor, z_next: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute loss for CAE + Linear MLP model
        
        Args:
            x_t: Input at time t
            x_next: Ground truth at time t+1
            x_t_recon: Reconstructed x_t
            x_next_pred: Predicted x_{t+1}
            z_t: Latent at time t
            z_next: Predicted latent at time t+1
        
        Returns:
            Dictionary with individual losses and total loss
        """
        # Individual losses
        recon_loss = self.reconstruction_loss(x_t, x_t_recon)
        pred_loss = self.prediction_loss(x_next, x_next_pred)
        latent_reg = self.latent_regularization(z_t) + self.latent_regularization(z_next)
        
        # Total loss
        total_loss = (self.recon_weight * recon_loss + 
                     self.pred_weight * pred_loss + 
                     self.latent_weight * latent_reg)
        
        return {
            'total': total_loss,
            'reconstruction': recon_loss,
            'prediction': pred_loss,
            'latent_reg': latent_reg
        }
    
    def compute_cae_weaklinear_loss(self, x_t: torch.Tensor, x_next: torch.Tensor,
                                   x_t_recon: torch.Tensor, x_next_pred: torch.Tensor,
                                   z_t: torch.Tensor, z_next: torch.Tensor,
                                   z_next_linear: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute loss for CAE + Weak Linear MLP model
        
        Args:
            x_t: Input at time t
            x_next: Ground truth at time t+1
            x_t_recon: Reconstructed x_t
            x_next_pred: Predicted x_{t+1}
            z_t: Latent at time t
            z_next: Predicted latent at time t+1 (nonlinear)
            z_next_linear: Linear approximation of next latent
        
        Returns:
            Dictionary with individual losses and total loss
        """
        # Individual losses
        recon_loss = self.reconstruction_loss(x_t, x_t_recon)
        pred_loss = self.prediction_loss(x_next, x_next_pred)
        latent_reg = self.latent_regularization(z_t) + self.latent_regularization(z_next)
        linear_loss = self.linear_constraint_loss(z_next, z_next_linear)
        
        # Total loss
        total_loss = (self.recon_weight * recon_loss + 
                     self.pred_weight * pred_loss + 
                     self.latent_weight * latent_reg +
                     self.linear_weight * linear_loss)
        
        return {
            'total': total_loss,
            'reconstruction': recon_loss,
            'prediction': pred_loss,
            'latent_reg': latent_reg,
            'linear_constraint': linear_loss
        }


def reshape_data_for_cae(x: torch.Tensor, dataset_name: str, 
                        target_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """
    Reshape data from DataLoader format to CAE input format
    
    Args:
        x: Input tensor from dataloader [batch, features] or [batch, channels, features]
        dataset_name: Name of the dataset
        target_shape: Target spatial shape (H, W). If None, use default for dataset
    
    Returns:
        Reshaped tensor [batch, channels, height, width]
    """
    if dataset_name.lower() == 'kolmogorov':
        # Kolmogorov: [batch, 4096] -> [batch, 1, 64, 64]
        if target_shape is None:
            target_shape = (64, 64)
        
        batch_size = x.shape[0]
        x = x.reshape(batch_size, 1, target_shape[0], target_shape[1])
        
    elif dataset_name.lower() == 'cylinder':
        # Cylinder: [batch, 2, features] -> [batch, 2, H, W]
        if x.dim() == 3:  # Already has channel dimension
            batch_size, n_channels, features = x.shape
            if target_shape is None:
                # Try to infer shape (assuming square for simplicity)
                spatial_size = int(np.sqrt(features))
                target_shape = (spatial_size, spatial_size)
            x = x.reshape(batch_size, n_channels, target_shape[0], target_shape[1])
        else:
            raise ValueError(f"Unexpected tensor shape for Cylinder dataset: {x.shape}")
            
    elif dataset_name.lower() == 'chap':
        # CHAP: [batch, features] -> [batch, 1, H, W]
        # Need to know original spatial dimensions
        if target_shape is None:
            raise ValueError("target_shape must be provided for CHAP dataset")
        
        batch_size = x.shape[0]
        x = x.reshape(batch_size, 1, target_shape[0], target_shape[1])
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return x


# Example usage for testing
if __name__ == "__main__":
    # Test with different datasets
    configs = [
        {
            'dataset_name': 'kolmogorov',
            'data_path': 'data',
            'batch_size': 16,
            'num_workers': 2
        },
        {
            'dataset_name': 'cylinder',
            'data_path': 'data',
            'batch_size': 8,
            'num_workers': 2
            # 'target_resolution': (32, 128),
            # 'interpolation_mode': 'bilinear'
        }
        # {
        #     'dataset_name': 'chap',
        #     'data_path': 'data',
        #     'batch_size': 4,
        #     'num_workers': 2,
        #     'chemical': 'Cl'
        # }
    ]
    
    for config in configs:
        print(f"\n{'='*50}")
        print(f"Testing {config['dataset_name']} dataset")
        print(f"{'='*50}")
        
        try:
            train_loader, val_loader, data_info = create_caemlp_dataloaders(config)
            
            print(f"Dataset info:")
            for key, value in data_info.items():
                if key != 'normalization':
                    print(f"  {key}: {value}")
            
            # Test loading one batch
            x_t, x_next = next(iter(train_loader))
            print(f"\nBatch test:")
            print(f"  x_t shape: {x_t.shape}")
            print(f"  x_next shape: {x_next.shape}")
            print(f"  x_t range: [{x_t.min():.4f}, {x_t.max():.4f}]")
            
        except Exception as e:
            print(f"Error loading {config['dataset_name']}: {e}")
    

    # Add loss testing
    print(f"\n{'='*50}")
    print("Testing loss functions")
    print(f"{'='*50}")
    
    # Create dummy tensors
    batch_size = 4
    x_t = torch.randn(batch_size, 1, 64, 64)
    x_next = torch.randn(batch_size, 1, 64, 64)
    x_t_recon = torch.randn(batch_size, 1, 64, 64)
    x_next_pred = torch.randn(batch_size, 1, 64, 64)
    z_t = torch.randn(batch_size, 32)
    z_next = torch.randn(batch_size, 32)
    z_next_linear = torch.randn(batch_size, 32)
    
    # Test loss computation
    loss_fn = CAEMLPLoss()
    
    # Test CAE Linear loss
    losses_linear = loss_fn.compute_cae_linear_loss(
        x_t, x_next, x_t_recon, x_next_pred, z_t, z_next
    )
    print("\nCAE Linear MLP losses:")
    for name, loss in losses_linear.items():
        print(f"  {name}: {loss.item():.4f}")
    
    # Test CAE Weak Linear loss
    losses_weak = loss_fn.compute_cae_weaklinear_loss(
        x_t, x_next, x_t_recon, x_next_pred, z_t, z_next, z_next_linear
    )
    print("\nCAE Weak Linear MLP losses:")
    for name, loss in losses_weak.items():
        print(f"  {name}: {loss.item():.4f}")