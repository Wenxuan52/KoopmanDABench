import torch

import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Union
import random

import os
import sys

current_file_path = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(current_file_path, '..', '..', '..'))
sys.path.insert(0, project_root)

from src.utils.Datasets import DatasetKol, DatasetCylinder


class SingleFrameMultiStepDataset(Dataset):
    
    def __init__(self, data: np.ndarray, prediction_steps: int = 5, is_train: bool = True):
        self.data = data
        self.prediction_steps = prediction_steps
        self.is_train = is_train
        
        if data.ndim < 2:
            raise ValueError(f"数据至少需要2维 [samples, time], 当前: {data.shape}")
        
        self.n_samples = data.shape[0]
        self.time_steps = data.shape[1]
        
        self.valid_time_steps = self.time_steps - prediction_steps
        
        if self.valid_time_steps <= 0:
            raise ValueError(
                f"时间步数 ({self.time_steps}) 必须大于预测步数 ({prediction_steps})"
            )
        
        self.total_samples = self.n_samples * self.valid_time_steps
        
        print(f"SingleFrameMultiStepDataset:")
        print(f"  原始数据形状: {data.shape}")
        print(f"  预测步数: {prediction_steps}")
        print(f"  每个样本的有效时间步: {self.valid_time_steps}")
        print(f"  总训练样本数: {self.total_samples}")
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        sample_idx = idx // self.valid_time_steps
        time_idx = idx % self.valid_time_steps
        
        input_frame = self.data[sample_idx, time_idx]
        
        target_frames = self.data[sample_idx, 
                                 time_idx + 1:time_idx + 1 + self.prediction_steps]
        
        input_tensor = torch.from_numpy(input_frame).float()
        target_tensor = torch.from_numpy(target_frames).float()
        
        return input_tensor, target_tensor


class SingleFrameDataLoader:
    
    def __init__(self, 
                 dataset: Union['DatasetKol', 'DatasetCylinder'], 
                 prediction_steps: int = 5,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 pin_memory: bool = True):
        self.base_dataset = dataset
        self.prediction_steps = prediction_steps
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        self.train_dataset = SingleFrameMultiStepDataset(
            dataset.train_data, 
            prediction_steps, 
            is_train=True
        )
        
        self.val_dataset = SingleFrameMultiStepDataset(
            dataset.val_data, 
            prediction_steps, 
            is_train=False
        )
        
        # 创建 PyTorch DataLoader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True  # 确保批次大小一致
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,  # 验证时不打乱
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
    
    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        return self.train_loader, self.val_loader
    
    def get_data_info(self):
        print(f"\n=== SingleFrameDataLoader 信息 ===")
        print(f"预测步数: {self.prediction_steps}")
        print(f"批次大小: {self.batch_size}")
        print(f"训练样本数: {len(self.train_dataset)}")
        print(f"验证样本数: {len(self.val_dataset)}")
        print(f"训练批次数: {len(self.train_loader)}")
        print(f"验证批次数: {len(self.val_loader)}")
        
        sample_input, sample_target = next(iter(self.train_loader))
        print(f"输入批次形状: {sample_input.shape}")
        print(f"目标批次形状: {sample_target.shape}")
        
        if sample_input.ndim == 4:  # [batch, channels, H, W]
            print(f"输入: [batch_size={sample_input.shape[0]}, "
                  f"channels={sample_input.shape[1]}, "
                  f"H={sample_input.shape[2]}, W={sample_input.shape[3]}]")
            print(f"目标: [batch_size={sample_target.shape[0]}, "
                  f"prediction_steps={sample_target.shape[1]}, "
                  f"channels={sample_target.shape[2]}, "
                  f"H={sample_target.shape[3]}, W={sample_target.shape[4]}]")
        elif sample_input.ndim == 3:  # [batch, H, W]
            print(f"输入: [batch_size={sample_input.shape[0]}, "
                  f"H={sample_input.shape[1]}, W={sample_input.shape[2]}]")
            print(f"目标: [batch_size={sample_target.shape[0]}, "
                  f"prediction_steps={sample_target.shape[1]}, "
                  f"H={sample_target.shape[2]}, W={sample_target.shape[3]}]")
            
def create_single_frame_loaders(data_path: str, 
                               dataset_type: str = 'kolmogorov',
                               prediction_steps: int = 5,
                               batch_size: int = 32,
                               normalize: bool = True,
                               train_ratio: float = 0.8,
                               shuffle: bool = True,
                               target_resolution: Optional[Tuple[int, int]] = None,
                               **kwargs) -> Tuple[DataLoader, DataLoader]:
    
    
    if dataset_type.lower() == 'kolmogorov':
        base_dataset = DatasetKol(
            data_path=data_path,
            normalize=normalize,
            train_ratio=train_ratio,
            **kwargs
        )
    elif dataset_type.lower() == 'cylinder':
        base_dataset = DatasetCylinder(
            data_path=data_path,
            normalize=normalize,
            train_ratio=train_ratio,
            target_resolution=target_resolution,
            **kwargs
        )
    else:
        raise ValueError(f"未知的数据集类型: {dataset_type}")
    
    dataloader = SingleFrameDataLoader(
        dataset=base_dataset,
        prediction_steps=prediction_steps,
        batch_size=batch_size,
        shuffle=shuffle
    )
    
    dataloader.get_data_info()
    
    return dataloader.get_loaders()


if __name__ == '__main__':
    train_loader, val_loader = create_single_frame_loaders(
        data_path = './data/Cylinder',
        dataset_type = 'cylinder',
        prediction_steps = 5,
        batch_size = 32,
        normalize=True,
        train_ratio=0.8,
        )
    
    for i, (input_batch, target_batch) in enumerate(train_loader):
        print(f"批次 {i}:")
        print(f"  输入形状: {input_batch.shape}")    # [16, 64, 64] 对于Kolmogorov
        print(f"  目标形状: {target_batch.shape}")   # [16, 10, 64, 64]
        
        if i >= 2:  # 只看前几个批次
            break