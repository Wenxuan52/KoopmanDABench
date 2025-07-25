from torch.utils.data import Dataset, DataLoader
import os
import sys
import numpy as np
import torch
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


if __name__ == '__main__':
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

    KDD = KolDynamicsDataset(data_path="data/kolmogorov/RE40_n4_T10/kolmogorov_train_data.npy",
                seq_length = 12,
                mean=None,
                std=None)
    
    print(KDD.mean)
    print(KDD.std)

    inx = 10
    print(KDD[inx][0].dtype)
    print(KDD[inx][1].dtype)

    print(KDD[inx][0].shape)
    print(KDD[inx][1].shape)

    print(KDD[inx][0].min())
    print(KDD[inx][1].max())

    print(KDD.total_sample)

    val_KDD = KolDynamicsDataset(data_path="data/kolmogorov/RE40_n4_T10/kolmogorov_val_data.npy",
                seq_length = 12,
                mean=KDD.mean,
                std=KDD.std)
    
    print(val_KDD.mean)
    print(val_KDD.std)

    print(val_KDD[inx][0].shape)
    print(val_KDD[inx][1].shape)

    print(val_KDD[inx][0].min())
    print(val_KDD[inx][1].max())
