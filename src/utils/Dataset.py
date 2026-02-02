from torch.utils.data import Dataset, DataLoader
import os
import sys
import numpy as np
import torch
import h5py
torch.set_default_dtype(torch.float32)

class CylinderDynamicsDataset(Dataset):
    """Base dataset class for DMD models"""
    
    def __init__(self,
                data_path: str,
                seq_length: int = 12,
                mean=None,
                std=None,
                ):

        self.data_path = data_path

        self.seq_length = seq_length
        
        self.data = np.load(self.data_path).astype(np.float32)
        print(f"Loaded Cylinder data with shape: {self.data.shape}")
        
        self.n_samples, self.frames, self.channel, self.H, self.W = self.data.shape
        
        if mean is None:
            mean_value = np.mean(self.data, axis=(0, 1, 3, 4))  # shape: [channel]
            self.mean = torch.from_numpy(mean_value.astype(np.float32))
        else:
            self.mean = mean

        if std is None:
            std_value = np.std(self.data, axis=(0, 1, 3, 4))  # shape: [channel]
            std_value[std_value < 1e-6] = 1.0  # avoid divide-by-zero
            self.std = torch.from_numpy(std_value.astype(np.float32))
        else:
            self.std = std
        
        self.num_per_sample = self.frames - self.seq_length

        self.total_sample = self.num_per_sample * self.n_samples

        self.create_data_set(self.data)
    
    def create_data_set(self, data):
        pool = []
        for i in range(self.n_samples):
            for j in range(self.num_per_sample):
                pool.append(data[i, j:j+self.seq_length+1])
        print('dataset total samples:', len(pool))
        self.pool = pool
    
    def __len__(self):
        return self.total_sample

    def __getitem__(self, idx):
        data = self.pool[idx]
        pre_seq = torch.tensor(data[:self.seq_length], dtype=torch.float32)
        post_seq = torch.tensor(data[1:self.seq_length+1], dtype=torch.float32)
        return self.normalize(pre_seq), self.normalize(post_seq)
    
    def normalize(self, x):
        mean = self.mean.reshape(1, -1, 1, 1)
        std = self.std.reshape(1, -1, 1, 1)
        return (x - mean) / std
    
    def denormalizer(self):
        def denormalize(x: torch.Tensor) -> torch.Tensor:
            mean = self.mean.reshape(1, -1, 1, 1)
            std = self.std.reshape(1, -1, 1, 1)
            try:
                return x * std + mean
            except:
                std_np = std.numpy()
                mean_np = mean.numpy()
                return x * std_np + mean_np
        return denormalize


class KolDynamicsDataset(Dataset):
    """Base dataset class for DMD models"""
    
    def __init__(self,
                data_path: str,
                seq_length: int = 12,
                mean=None,
                std=None,
                ):

        self.data_path = data_path

        self.seq_length = seq_length
        
        self.data = np.load(self.data_path).astype(np.float32)
        self.data = np.expand_dims(self.data, axis=2)
        print(f"Loaded Kolmogorov data with shape: {self.data.shape}")
        
        self.n_samples, self.frames, self.channel, self.H, self.W = self.data.shape
        
        if mean is None:
            mean_value = np.mean(self.data, axis=(0, 1, 3, 4))  # shape: [channel]
            self.mean = torch.from_numpy(mean_value.astype(np.float32))
        else:
            self.mean = mean

        if std is None:
            std_value = np.std(self.data, axis=(0, 1, 3, 4))  # shape: [channel]
            std_value[std_value < 1e-6] = 1.0  # avoid divide-by-zero
            self.std = torch.from_numpy(std_value.astype(np.float32))
        else:
            self.std = std
        
        self.num_per_sample = self.frames - self.seq_length

        self.total_sample = self.num_per_sample * self.n_samples

        self.create_data_set(self.data)
    
    def create_data_set(self, data):
        pool = []
        for i in range(self.n_samples):
            for j in range(self.num_per_sample):
                pool.append(data[i, j:j+self.seq_length+1])
        print('dataset total samples:', len(pool))
        self.pool = pool
    
    def __len__(self):
        return self.total_sample

    def __getitem__(self, idx):
        data = self.pool[idx]
        pre_seq = torch.tensor(data[:self.seq_length], dtype=torch.float32)
        post_seq = torch.tensor(data[1:self.seq_length+1], dtype=torch.float32)
        return self.normalize(pre_seq), self.normalize(post_seq)
    
    def normalize(self, x):
        mean = self.mean.reshape(1, -1, 1, 1)
        std = self.std.reshape(1, -1, 1, 1)
        return (x - mean) / std
    
    def denormalizer(self):
        def denormalize(x: torch.Tensor) -> torch.Tensor:
            mean = self.mean.reshape(1, -1, 1, 1)
            std = self.std.reshape(1, -1, 1, 1)
            try:
                return x * std + mean
            except:
                std_np = std.numpy()
                mean_np = mean.numpy()
                return x * std_np + mean_np
        return denormalize


class DamDynamicsDataset(Dataset):
    """Base dataset class for DMD models"""
    
    def __init__(self,
                data_path: str,
                seq_length: int = 12,
                mean=None,
                std=None,
                ):

        self.data_path = data_path

        self.seq_length = seq_length
        
        self.data = np.load(self.data_path).astype(np.float32)
        print(f"Loaded Cylinder data with shape: {self.data.shape}")
        
        self.n_samples, self.frames, self.channel, self.H, self.W = self.data.shape
        
        if mean is None:
            mean_value = np.mean(self.data, axis=(0, 1, 3, 4))  # shape: [channel]
            self.mean = torch.from_numpy(mean_value.astype(np.float32))
        else:
            self.mean = mean

        if std is None:
            std_value = np.std(self.data, axis=(0, 1, 3, 4))  # shape: [channel]
            std_value[std_value < 1e-6] = 1.0  # avoid divide-by-zero
            self.std = torch.from_numpy(std_value.astype(np.float32))
        else:
            self.std = std
        
        self.num_per_sample = self.frames - self.seq_length

        self.total_sample = self.num_per_sample * self.n_samples

        self.create_data_set(self.data)
    
    def create_data_set(self, data):
        pool = []
        for i in range(self.n_samples):
            for j in range(self.num_per_sample):
                pool.append(data[i, j:j+self.seq_length+1])
        print('dataset total samples:', len(pool))
        self.pool = pool
    
    def __len__(self):
        return self.total_sample

    def __getitem__(self, idx):
        data = self.pool[idx]
        pre_seq = torch.tensor(data[:self.seq_length], dtype=torch.float32)
        post_seq = torch.tensor(data[1:self.seq_length+1], dtype=torch.float32)
        return self.normalize(pre_seq), self.normalize(post_seq)
    
    def normalize(self, x):
        mean = self.mean.reshape(1, -1, 1, 1)
        std = self.std.reshape(1, -1, 1, 1)
        return (x - mean) / std
    
    def denormalizer(self):
        def denormalize(x: torch.Tensor) -> torch.Tensor:
            mean = self.mean.reshape(1, -1, 1, 1)
            std = self.std.reshape(1, -1, 1, 1)
            try:
                return x * std + mean
            except:
                std_np = std.numpy()
                mean_np = mean.numpy()
                return x * std_np + mean_np
        return denormalize


class ERA5Dataset(Dataset):
    def __init__(self, 
                 data_path: str,
                 seq_length: int = 12,
                 min_path: str = None,
                 max_path: str = None):
        self.data_path = data_path
        self.seq_length = seq_length

        with h5py.File(self.data_path, 'r') as f:
            self.data = f['data'][:]  # shape: [N, H, W, C]
        print(f"Loaded ERA5 data with shape: {self.data.shape}")
        self.n_frames, self.H, self.W, self.C = self.data.shape

        if min_path is not None and max_path is not None:
            self.min = torch.from_numpy(np.load(min_path).astype(np.float32))  # shape: [C]
            self.max = torch.from_numpy(np.load(max_path).astype(np.float32))
        else:
            raise ValueError("min_path and max_path must be provided")

        self.num_per_sample = self.n_frames - seq_length
        self.create_data_set(self.data)

    def create_data_set(self, data):
        self.pool = []
        for i in range(self.num_per_sample):
            self.pool.append(data[i:i+self.seq_length+1])
        self.total_sample = len(self.pool)
        print('Dataset total samples:', len(self.pool))

    def __len__(self):
        return len(self.pool)

    def __getitem__(self, idx):
        data = self.pool[idx]
        data = torch.from_numpy(data).permute(0, 3, 1, 2).float()

        pre_seq = data[:self.seq_length]
        post_seq = data[1:self.seq_length+1]

        return self.normalize(pre_seq), self.normalize(post_seq)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        min_v = self.min.reshape(1, -1, 1, 1)
        max_v = self.max.reshape(1, -1, 1, 1)
        return (x - min_v) / (max_v - min_v + 1e-6)

    def denormalizer(self):
        def denormalize(x: torch.Tensor) -> torch.Tensor:
            min_v = self.min.reshape(1, -1, 1, 1)
            max_v = self.max.reshape(1, -1, 1, 1)
            return x * (max_v - min_v + 1e-6) + min_v
        return denormalize


class ERA5HighDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 seq_length: int = 12,
                 min_path: str = None,
                 max_path: str = None,
                 dataset_key: str = "data"):
        self.data_path = data_path
        self.seq_length = seq_length

        with h5py.File(self.data_path, 'r') as f:
            self.data = f[dataset_key][:]  # shape: [N, 240, 121] or [N, 240, 121, C]
        print(f"Loaded ERA5High data with shape: {self.data.shape}")

        # 统一成 [N, H, W, C]
        if self.data.ndim == 3:
            # [N, H, W] -> [N, H, W, 1]
            self.data = self.data[..., None]
        elif self.data.ndim != 4:
            raise ValueError(f"Unsupported data shape: {self.data.shape}. "
                             f"Expected [N,H,W] or [N,H,W,C].")

        self.n_frames, self.H, self.W, self.C = self.data.shape

        if min_path is not None and max_path is not None:
            self.min = torch.from_numpy(np.load(min_path).astype(np.float32))  # shape: [C]
            self.max = torch.from_numpy(np.load(max_path).astype(np.float32))
        else:
            raise ValueError("min_path and max_path must be provided")

        # （可选但强烈建议）检查 min/max 的维度对不对
        if self.min.numel() != self.C or self.max.numel() != self.C:
            raise ValueError(f"min/max should have shape [C]={self.C}, "
                             f"but got min:{tuple(self.min.shape)} max:{tuple(self.max.shape)}")

        self.num_per_sample = self.n_frames - seq_length
        self.create_data_set(self.data)

    def create_data_set(self, data):
        self.pool = []
        for i in range(self.num_per_sample):
            self.pool.append(data[i:i + self.seq_length + 1])
        self.total_sample = len(self.pool)
        print('Dataset total samples:', len(self.pool))

    def __len__(self):
        return len(self.pool)

    def __getitem__(self, idx):
        data = self.pool[idx]  # [T+1, H, W, C]
        data = torch.from_numpy(data).permute(0, 3, 1, 2).float()  # -> [T+1, C, H, W]

        pre_seq = data[:self.seq_length]
        post_seq = data[1:self.seq_length + 1]

        return self.normalize(pre_seq), self.normalize(post_seq)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        min_v = self.min.reshape(1, -1, 1, 1)
        max_v = self.max.reshape(1, -1, 1, 1)
        return (x - min_v) / (max_v - min_v + 1e-6)

    def denormalizer(self):
        def denormalize(x: torch.Tensor) -> torch.Tensor:
            min_v = self.min.reshape(1, -1, 1, 1)
            max_v = self.max.reshape(1, -1, 1, 1)
            return x * (max_v - min_v + 1e-6) + min_v
        return denormalize


if __name__ == '__main__':

    print("Testing Dataset classes...")

    # CDD = CylinderDynamicsDataset(data_path="data/cylinder/cylinder_train_data.npy",
    #             seq_length = 12,
    #             mean=None,
    #             std=None)
    
    # print(CDD.mean)
    # print(CDD.std)

    # inx = 10
    # print(CDD[inx][0].dtype)
    # print(CDD[inx][1].dtype)

    # print(CDD[inx][0].shape)
    # print(CDD[inx][1].shape)

    # print(CDD[inx][0].min())
    # print(CDD[inx][1].max())

    # print(CDD.total_sample)

    # val_CDD = CylinderDynamicsDataset(data_path="data/cylinder/cylinder_val_data.npy",
    #             seq_length = 12,
    #             mean=CDD.mean,
    #             std=CDD.std)
    
    # print(val_CDD.mean)
    # print(val_CDD.std)

    # print(val_CDD[inx][0].shape)
    # print(val_CDD[inx][1].shape)

    # print(val_CDD[inx][0].min())
    # print(val_CDD[inx][1].max())

    # KDD = KolDynamicsDataset(data_path="data/kol/kolmogorov_train_data.npy",
    #             seq_length = 12,
    #             mean=None,
    #             std=None)
    
    # print(KDD.mean)
    # print(KDD.std)

    # inx = 10
    # print(KDD[inx][0].dtype)
    # print(KDD[inx][1].dtype)

    # print(KDD[inx][0].shape)
    # print(KDD[inx][1].shape)

    # print(KDD[inx][0].min())
    # print(KDD[inx][1].max())

    # print(KDD.total_sample)

    # val_KDD = KolDynamicsDataset(data_path="data/kol/kolmogorov_val_data.npy",
    #             seq_length = 12,
    #             mean=KDD.mean,
    #             std=KDD.std)
    
    # print(val_KDD.mean)
    # print(val_KDD.std)

    # print(val_KDD[inx][0].shape)
    # print(val_KDD[inx][1].shape)

    # print(val_KDD[inx][0].min())
    # print(val_KDD[inx][1].max())

    # DDD = DamDynamicsDataset(data_path="data/dam/dam_train_data.npy",
    #             seq_length = 12,
    #             mean=None,
    #             std=None)
    
    # print(DDD.mean)
    # print(DDD.std)

    # inx = 10
    # print(DDD[inx][0].dtype)
    # print(DDD[inx][1].dtype)

    # print(DDD[inx][0].shape)
    # print(DDD[inx][1].shape)

    # print(DDD[inx][0].min())
    # print(DDD[inx][1].max())

    # print(DDD.total_sample)

    # val_DDD = DamDynamicsDataset(data_path="data/dam/dam_val_data.npy",
    #             seq_length = 12,
    #             mean=DDD.mean,
    #             std=DDD.std)
    
    # print(val_DDD.mean)
    # print(val_DDD.std)

    # print(val_DDD[inx][0].shape)
    # print(val_DDD[inx][1].shape)

    # print(val_DDD[inx][0].min())
    # print(val_DDD[inx][1].max())

    train_set = ERA5HighDataset(
        data_path="data/ERA5_high/raw_data/weatherbench_train.h5",
        seq_length=12,
        min_path="data/ERA5_high/raw_data/era5high_240x121_min.npy",
        max_path="data/ERA5_high/raw_data/era5high_240x121_max.npy"
    )

    val_set = ERA5HighDataset(
        data_path="data/ERA5_high/raw_data/weatherbench_test.h5",
        seq_length=12,
        min_path="data/ERA5_high/raw_data/era5high_240x121_min.npy",
        max_path="data/ERA5_high/raw_data/era5high_240x121_max.npy",
    )

    inx = 1

    print(train_set.min.shape)
    print(train_set.max.shape)

    print(train_set[inx][0].shape)

    inx = 10
    x, y = train_set[inx]
    print("Train x shape:", x.shape)  # [12, C, H, W]
    print("Train y shape:", y.shape)  # [12, C, H, W]
    print("x min:", x.min().item(), "x max:", x.max().item())
    print("y min:", y.min().item(), "y max:", y.max().item())

    assert x.min() >= 0.0 and x.max() <= 1.0, "x not in [0, 1]"
    assert y.min() >= 0.0 and y.max() <= 1.0, "y not in [0, 1]"

    denorm = train_set.denormalizer()
    x_denorm = denorm(x)
    print("Denormalized x shape:", x_denorm.shape)
    print("Channel 0 x_denorm min:", x_denorm[:, 0, ...].min().item(), "Channel 0 x_denorm max:", x_denorm[:, 0, ...].max().item())
    print("Channel 1 x_denorm min:", x_denorm[:, 1, ...].min().item(), "Channel 1 x_denorm max:", x_denorm[:, 1, ...].max().item())
    print("Channel 2 x_denorm min:", x_denorm[:, 2, ...].min().item(), "Channel 2 x_denorm max:", x_denorm[:, 2, ...].max().item())
    print("Channel 3 x_denorm min:", x_denorm[:, 3, ...].min().item(), "Channel 3 x_denorm max:", x_denorm[:, 3, ...].max().item())
    print("Channel 4 x_denorm min:", x_denorm[:, 4, ...].min().item(), "Channel 4 x_denorm max:", x_denorm[:, 4, ...].max().item())