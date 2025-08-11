import numpy as np
import torch
import os
import sys
import yaml
import random
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
sys.path.append(src_directory)

from src.utils.Dataset import ERA5Dataset
from src.models.DMD.base import TorchDMD

def set_seed(seed: int = 42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Set random seed to {seed}")

def prepare_era5_dmd_data(era5_dataset, device):
    """Prepare ERA5 data for DMD training and move to specified device"""
    
    # Get raw data from ERA5Dataset
    raw_data = era5_dataset.data  # Shape: (N, H, W, C) = (50258, 64, 32, 5)
    
    # Convert to torch tensor and reorder dimensions to (N, C, H, W)
    if isinstance(raw_data, np.ndarray):
        data_tensor = torch.from_numpy(raw_data).float()
    else:
        data_tensor = raw_data.float()
    
    # Permute from (N, H, W, C) to (N, C, H, W)
    data_tensor = data_tensor.permute(0, 3, 1, 2)  # (50258, 5, 64, 32)
    
    # Normalize the data using the dataset's normalize function
    print("[INFO] Normalizing ERA5 data...")
    normalized_data = era5_dataset.normalize(data_tensor)
    
    N, C, H, W = normalized_data.shape
    D = C * H * W  # Flattened spatial-channel dimension
    
    # For ERA5, we create pairs from consecutive time steps
    # Each sample at time t pairs with sample at time t+1
    num_pairs = N - 1
    
    print(f"Input raw data shape: {raw_data.shape}")
    print(f"Tensor data shape after permute: {data_tensor.shape}")
    print(f"Normalized data shape: {normalized_data.shape}")
    print(f"Normalized data range: [{normalized_data.min():.4f}, {normalized_data.max():.4f}]")
    print(f"Total consecutive pairs: {num_pairs}")
    print(f"Each snapshot flattened to: {D}-dim")
    print(f"Data will be moved to device: {device}")
    
    # Create tensors to hold reshaped data directly on the target device
    X_train = torch.zeros((D, num_pairs), dtype=torch.float32, device=device)
    y_train = torch.zeros((D, num_pairs), dtype=torch.float32, device=device)
    
    # Move normalized data to device
    normalized_data = normalized_data.to(device)
    
    # Create pairs from consecutive time steps
    for i in tqdm(range(num_pairs), desc="Preparing DMD pairs"):
        # Current state: (C, H, W) -> (C*H*W,)
        current_state = normalized_data[i].reshape(-1)  # Shape: (D,) = (10240,)
        # Next state: (C, H, W) -> (C*H*W,)
        next_state = normalized_data[i + 1].reshape(-1)  # Shape: (D,) = (10240,)
        
        X_train[:, i] = current_state
        y_train[:, i] = next_state
    
    print(f"[INFO] X_train shape: {X_train.shape}")
    print(f"[INFO] y_train shape: {y_train.shape}")
    print(f"[INFO] X_train range: [{X_train.min():.4f}, {X_train.max():.4f}]")
    print(f"[INFO] y_train range: [{y_train.min():.4f}, {y_train.max():.4f}]")
    print(f"[INFO] Data preparation completed on {device}")
    return X_train, y_train

def get_memory_usage():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        return f"Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB"
    return "CPU mode"

def visualize(data):
    plt.figure(figsize=(6, 3))
    plt.imshow(data, cmap='viridis')
    plt.axis('off')
    plt.savefig('temp.png', bbox_inches='tight', dpi=100)
    plt.close()

def main():
    start_time = time.time()
    
    set_seed(42)
    torch.set_default_dtype(torch.float32)
    
    # Load configuration
    config_path = "../../../../configs/DMD_ERA5.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Device setup
    if config['use_gpu'] and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"[INFO] Using GPU: {torch.cuda.get_device_properties(0).name}")
        print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
    else:
        device = torch.device("cpu")
        print(f"[INFO] Using CPU device")
        if config['use_gpu'] and not torch.cuda.is_available():
            print("[WARNING] GPU requested but not available, falling back to CPU")
    
    print(f"[INFO] Memory usage at start: {get_memory_usage()}")
    print("[INFO] Starting ERA5 DMD Model Training")
    print(f"[INFO] Configuration: {config}")
    
    # ========================================
    # Train Forward Model
    # ========================================
    print("\n" + "="*50)
    print("Fitting FORWARD MODEL")
    print("="*50)
    
    # Load ERA5 dataset
    print("[INFO] Loading ERA5 dataset...")
    era5_train_set = ERA5Dataset(
        data_path="../../../../data/ERA5/ERA5_data/train_seq_state.h5",
        seq_length=config['seq_length'],
        min_path="../../../../data/ERA5/ERA5_data/min_val.npy",
        max_path="../../../../data/ERA5/ERA5_data/max_val.npy"
    )
    
    print(f"[INFO] ERA5 dataset loaded with shape: {era5_train_set.data.shape}")
    print(f"[INFO] Memory usage after data loading: {get_memory_usage()}")

    # Prepare DMD data and move to device
    print("[INFO] Preparing ERA5 DMD data...")
    X_train, y_train = prepare_era5_dmd_data(era5_train_set, device)

    print(f"X_train shape: {X_train.shape} on {X_train.device}")
    print(f"y_train shape: {y_train.shape} on {y_train.device}")
    print(f"[INFO] Memory usage after data preparation: {get_memory_usage()}")

    # Initialize DMD model with device
    print("[INFO] Initializing DMD model...")
    dmd = TorchDMD(svd_rank=config['rank'], device=device)
    
    print(f"[INFO] Memory usage after model initialization: {get_memory_usage()}")

    # Fit the model
    print("[INFO] Fitting DMD model...")

    temp_image = X_train[:, 1000].reshape(5, 64, 32)
    visualize(temp_image[0].cpu().numpy())

    dmd.fit(X_train, y_train)

    print(f"[INFO] Memory usage after fitting: {get_memory_usage()}")

    # Save the model
    save_path = os.path.join(config['save_folder'], "dmd_model.pth")
    os.makedirs(config['save_folder'], exist_ok=True)
    
    print("[INFO] Saving ERA5 DMD model...")
    dmd.save_dmd(save_path)

    print('[INFO] ERA5 DMD fit successfully completed!')
    print(f"[INFO] Final memory usage: {get_memory_usage()}")
    
    end_time = time.time()
    total_runtime = end_time - start_time
    print(f"[INFO] Total Runtime: {total_runtime:.2f} seconds")


if __name__ == "__main__":
    main()