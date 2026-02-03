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

from src.utils.Dataset import ERA5HighDataset
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
    """Prepare ERA5 high-res data for DMD training and move to specified device"""

    raw_data = era5_dataset.data  # Shape: (N, H, W, C) = (54056, 240, 121, 5)

    if isinstance(raw_data, np.ndarray):
        data_tensor = torch.from_numpy(raw_data).float()
    else:
        data_tensor = raw_data.float()

    # Permute from (N, H, W, C) to (N, C, H, W)
    data_tensor = data_tensor.permute(0, 3, 1, 2)  # (N, 5, 240, 121)

    # Normalize the data using the dataset's normalize function
    print("[INFO] Normalizing ERA5 high-res data...")
    normalized_data = era5_dataset.normalize(data_tensor)

    N, C, H, W = normalized_data.shape
    D = C * H * W  # Flattened spatial-channel dimension

    num_pairs = N - 1

    print(f"Input raw data shape: {raw_data.shape}")
    print(f"Tensor data shape after permute: {data_tensor.shape}")
    print(f"Normalized data shape: {normalized_data.shape}")
    print(f"Normalized data range: [{normalized_data.min():.4f}, {normalized_data.max():.4f}]")
    print(f"Total consecutive pairs: {num_pairs}")
    print(f"Each snapshot flattened to: {D}-dim")
    print(f"Data will be moved to device: {device}")

    X_train = torch.zeros((D, num_pairs), dtype=torch.float32, device=device)
    y_train = torch.zeros((D, num_pairs), dtype=torch.float32, device=device)

    normalized_data = normalized_data.to(device)

    for i in tqdm(range(num_pairs), desc="Preparing DMD pairs"):
        current_state = normalized_data[i].reshape(-1)
        next_state = normalized_data[i + 1].reshape(-1)

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
        memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        return f"Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB"
    return "CPU mode"


def visualize(data, save_path="temp.png"):
    plt.figure(figsize=(6, 3))
    plt.imshow(data, cmap='viridis')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close()


def main():
    start_time = time.time()

    set_seed(42)
    torch.set_default_dtype(torch.float32)

    # Load configuration
    config_path = "../../../../configs/DMD_ERA5_HIGH.yaml"
    if not os.path.exists(config_path):
        config_path = "../../../../configs/DMD_ERA5.yaml"

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Device setup
    if config.get('use_gpu', False) and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"[INFO] Using GPU: {torch.cuda.get_device_properties(0).name}")
        print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
    else:
        device = torch.device("cpu")
        print(f"[INFO] Using CPU device")
        if config.get('use_gpu', False) and not torch.cuda.is_available():
            print("[WARNING] GPU requested but not available, falling back to CPU")

    print(f"[INFO] Memory usage at start: {get_memory_usage()}")
    print("[INFO] Starting ERA5 High-Resolution DMD Model Training")
    print(f"[INFO] Configuration: {config}")

    # ========================================
    # Train Forward Model
    # ========================================
    print("\n" + "=" * 50)
    print("Fitting FORWARD MODEL")
    print("=" * 50)

    # Load ERA5 dataset
    print("[INFO] Loading ERA5 high-resolution dataset...")
    train_data_path = config.get(
        "train_data_path",
        "../../../../data/ERA5_high/raw_data/weatherbench_train.h5",
    )
    min_path = config.get(
        "min_path",
        "../../../../data/ERA5_high/raw_data/era5high_240x121_min.npy",
    )
    max_path = config.get(
        "max_path",
        "../../../../data/ERA5_high/raw_data/era5high_240x121_max.npy",
    )

    era5_train_set = ERA5HighDataset(
        data_path=train_data_path,
        seq_length=config['seq_length'],
        min_path=min_path,
        max_path=max_path,
    )

    print(f"[INFO] ERA5 high-res dataset loaded with shape: {era5_train_set.data.shape}")
    print(f"[INFO] Memory usage after data loading: {get_memory_usage()}")

    # Prepare DMD data and move to device
    print("[INFO] Preparing ERA5 high-res DMD data...")
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

    sample_index = min(1000, X_train.shape[1] - 1)
    temp_image = X_train[:, sample_index].reshape(5, 240, 121)
    visualize(temp_image[0].cpu().numpy())

    dmd.fit(X_train, y_train)

    print(f"[INFO] Memory usage after fitting: {get_memory_usage()}")

    # Save the model
    save_folder = config.get("save_folder", "../../../../results/DMD/ERA5_High")
    save_path = os.path.join(save_folder, "dmd_model.pth")
    os.makedirs(save_folder, exist_ok=True)

    print("[INFO] Saving ERA5 high-res DMD model...")
    dmd.save_dmd(save_path)

    print('[INFO] ERA5 high-res DMD fit successfully completed!')
    print(f"[INFO] Final memory usage: {get_memory_usage()}")

    end_time = time.time()
    total_runtime = end_time - start_time
    print(f"[INFO] Total Runtime: {total_runtime:.2f} seconds")


if __name__ == "__main__":
    main()
