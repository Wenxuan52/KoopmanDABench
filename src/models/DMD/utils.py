import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import json
from typing import Tuple, Optional, Dict, Any, List
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


class DMDDataLoader:
    """Dataset manager for DMD models"""
    
    def __init__(self, dataset_name: str, data_path: str, normalize: bool = True, 
                 train_ratio: float = 0.8, random_seed: int = 42, **dataset_kwargs):
        """
        Args:
            dataset_name: Name of dataset ("kolmogorov", "cylinder", "chap")
            data_path: Root path to data folder
            normalize: Whether to normalize data
            train_ratio: Ratio for train/val split
            random_seed: Random seed
            **dataset_kwargs: Additional arguments for specific datasets
        """
        self.dataset_name = dataset_name.lower()
        self.data_path = data_path
        self.normalize = normalize
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        self.dataset_kwargs = dataset_kwargs
        
        # Create datasets
        self.train_dataset, self.val_dataset = self._create_datasets()
    
    def _create_datasets(self) -> Tuple[BaseDataset, BaseDataset]:
        """Create train and validation datasets"""
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
    
    def get_data_stats(self) -> Dict[str, Any]:
        """Get statistics about the dataset"""
        return {
            "dataset_name": self.dataset_name,
            "train_size": len(self.train_dataset),
            "val_size": len(self.val_dataset),
            "train_data_shape": self.train_dataset.data.shape if hasattr(self.train_dataset, 'data') else None,
            "val_data_shape": self.val_dataset.data.shape if hasattr(self.val_dataset, 'data') else None,
            "normalized": self.normalize
        }


# Example usage in main.py:
if __name__ == "__main__":
    # Test Kolmogorov dataset
    print("Testing Kolmogorov dataset...")
    kol_loader = DMDDataLoader(
        dataset_name="kolmogorov",
        data_path="./data"
    )
    print(kol_loader.get_data_stats())
    print(f"Train data shape: {kol_loader.train_dataset.data.shape}")
    print(f"Val data shape: {kol_loader.val_dataset.data.shape}")
    
    # # Test Cylinder dataset
    # print("\nTesting Cylinder dataset...")
    # cyl_loader = DMDDataLoader(
    #     dataset_name="cylinder",
    #     data_path="./data",
    #     target_resolution=(100, 140),  # Optional: interpolate to this resolution
    #     interpolation_mode='bilinear'
    # )
    # print(cyl_loader.get_data_stats())
    
    # # Test CHAP dataset
    # print("\nTesting CHAP dataset...")
    # chap_loader = DMDDataLoader(
    #     dataset_name="chap",
    #     data_path="./data",
    #     chemical="Cl"  # Can be "Cl", "NH4", "NO3", "SO4"
    # )
    # print(chap_loader.get_data_stats())