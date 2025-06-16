import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import h5py
import os
from typing import Dict, Optional, Tuple, List


class BaseDataset(Dataset):
    def __init__(self, data_path: str, split: str = 'train', input_steps: int = 10, output_steps: int = 10, stride: int = 1, normalize: bool = True, use_sparse: bool = False):
        self.data_path = data_path
        self.split = split
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.stride = stride
        self.normalize = normalize
        self.use_sparse = use_sparse


class DMDDatasetH5(BaseDataset):
    def __init__(self,
                data_path: str,
                split: str = 'train',
                input_steps: int = 10, output_steps: int = 10, stride: int = 1, normalize: bool = True, use_sparse: bool = False):
        self.data_path = data_path
        self.split = split
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.stride = stride
        self.normalize = normalize
        self.use_sparse = use_sparse
        