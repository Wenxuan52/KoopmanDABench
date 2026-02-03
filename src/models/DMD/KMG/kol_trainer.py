import numpy as np
import torch
import os
import sys
import yaml
import random
import time

current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
sys.path.append(src_directory)

from src.utils.Dataset import KolDynamicsDataset
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


def prepare_dmd_data(data, device):
    """Prepare data for DMD training and move to specified device"""

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()

    N, T, C, H, W = data.shape
    D = C * H * W
    num_pairs = (T - 1) * N

    print(f"Input data shape: {data.shape}")
    print(f"Each sample provides {T-1} pairs.")
    print(f"Total (X, X') pairs: {num_pairs}")
    print(f"Each snapshot flattened to: {D}-dim")
    print(f"Data will be moved to device: {device}")

    X_train = torch.zeros((D, num_pairs), dtype=torch.float32, device=device)
    y_train = torch.zeros((D, num_pairs), dtype=torch.float32, device=device)

    data = data.to(device)

    idx = 0
    for i in range(N):
        sample = data[i]
        x_seq = sample[:-1].reshape(T - 1, D)
        y_seq = sample[1:].reshape(T - 1, D)

        X_train[:, idx:idx + T - 1] = x_seq.T
        y_train[:, idx:idx + T - 1] = y_seq.T

        idx += T - 1

    print(f"[INFO] Data preparation completed on {device}")
    return X_train, y_train


def get_memory_usage():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        return f"Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB"
    return "CPU mode"


def main():
    start_time = time.time()

    set_seed(42)
    torch.set_default_dtype(torch.float32)

    config_path = "../../../../configs/DMD_KOL.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

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
    print("[INFO] Starting Kolmogorov DMD Model Training")
    print(f"[INFO] Configuration: {config}")

    print("\n" + "=" * 50)
    print("Fitting FORWARD MODEL")
    print("=" * 50)

    print("[INFO] Loading dataset...")
    kol_train_dataset = KolDynamicsDataset(
        data_path="../../../../data/kol/kolmogorov_train_data.npy",
        seq_length=config['seq_length'],
        mean=None,
        std=None
    )

    train_data = torch.tensor(kol_train_dataset.data, dtype=torch.float32)
    norm_train_data = kol_train_dataset.normalize(train_data)

    print(f"[INFO] Memory usage after data loading: {get_memory_usage()}")

    print("[INFO] Preparing DMD data...")
    X_train, y_train = prepare_dmd_data(norm_train_data, device)

    print(f"X_train shape: {X_train.shape} on {X_train.device}")
    print(f"y_train shape: {y_train.shape} on {y_train.device}")
    print(f"[INFO] Memory usage after data preparation: {get_memory_usage()}")

    print("[INFO] Initializing DMD model...")
    dmd = TorchDMD(svd_rank=config['rank'], device=device)

    print(f"[INFO] Memory usage after model initialization: {get_memory_usage()}")

    print("[INFO] Fitting DMD model...")
    dmd.fit(X_train, y_train)

    print(f"[INFO] Memory usage after fitting: {get_memory_usage()}")

    save_path = os.path.join(config['save_folder'], "dmd_model.pth")
    os.makedirs(config['save_folder'], exist_ok=True)

    print("[INFO] Saving DMD model...")
    dmd.save_dmd(save_path)

    print('[INFO] DMD fit successfully completed!')
    print(f"[INFO] Final memory usage: {get_memory_usage()}")

    end_time = time.time()
    total_runtime = end_time - start_time
    print(f"[INFO] Total Runtime: {total_runtime:.2f} seconds")


if __name__ == "__main__":
    main()
