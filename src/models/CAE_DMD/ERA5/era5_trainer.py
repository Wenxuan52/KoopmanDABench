import numpy as np
import torch
import os
import sys
import yaml
import random
import time
from tqdm import tqdm

current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
sys.path.append(src_directory)

from src.utils.Dataset import ERA5Dataset
from src.models.DMD.base import TorchDMD
from src.models.CAE_Koopman.ERA5.era5_model_FTF import ERA5_C_FORWARD

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

def extract_latent_dynamics_era5(koopman_model, era5_dataset, device):
    """Extract latent space dynamics from trained CAE+Koopman model for ERA5 data"""
    
    koopman_model.eval()
    
    # Get raw data and prepare consecutive pairs
    raw_data = era5_dataset.data  # Shape: (N, H, W, C) = (50258, 64, 32, 5)
    
    # Convert to torch tensor and reorder dimensions to (N, C, H, W)
    data_tensor = torch.from_numpy(raw_data).float()
    data_tensor = data_tensor.permute(0, 3, 1, 2)  # (50258, 5, 64, 32)
    
    # Normalize the data
    normalized_data = era5_dataset.normalize(data_tensor)
    
    N, C, H, W = normalized_data.shape
    hidden_dim = koopman_model.hidden_dim
    
    # For ERA5, create pairs from consecutive time steps
    num_pairs = N - 1
    
    print(f"ERA5 raw data shape: {raw_data.shape}")
    print(f"Normalized data shape: {normalized_data.shape}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Total consecutive pairs: {num_pairs}")
    
    # Initialize tensors to store latent dynamics
    Z_current = torch.zeros((hidden_dim, num_pairs), dtype=torch.float32, device=device)
    Z_next = torch.zeros((hidden_dim, num_pairs), dtype=torch.float32, device=device)
    
    # Move data to device
    normalized_data = normalized_data.to(device)
    
    # Extract latent dynamics in batches to save memory
    batch_size = 1000  # Process in batches to avoid memory issues
    
    with torch.no_grad():
        for start_idx in tqdm(range(0, num_pairs, batch_size), desc="Extracting latent dynamics"):
            end_idx = min(start_idx + batch_size, num_pairs)
            
            # Get current batch
            current_batch = normalized_data[start_idx:end_idx]  # (batch_size, C, H, W)
            next_batch = normalized_data[start_idx+1:end_idx+1]  # (batch_size, C, H, W)
            
            # Encode current and next states
            z_current_batch = koopman_model.K_S(current_batch)  # (batch_size, hidden_dim)
            z_next_batch = koopman_model.K_S(next_batch)        # (batch_size, hidden_dim)
            
            # Store in matrix format: (hidden_dim, batch_size)
            batch_actual_size = end_idx - start_idx
            Z_current[:, start_idx:end_idx] = z_current_batch.T
            Z_next[:, start_idx:end_idx] = z_next_batch.T
    
    print(f"[INFO] ERA5 latent dynamics extraction completed")
    return Z_current, Z_next

def get_memory_usage():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        return f"Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB"
    return "CPU mode"

def main():
    start_time = time.time()
    
    set_seed(42)
    torch.set_default_dtype(torch.float32)
    
    # Load configuration
    config_path = "../../../../configs/CAE_DMD_ERA5.yaml"
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
    print("[INFO] Starting CAE+DMD ERA5 Model Training")
    print(f"[INFO] Configuration: {config}")
    
    # ========================================
    # Load Pre-trained CAE+Koopman Model
    # ========================================
    print("\n" + "="*50)
    print("Loading Pre-trained CAE+Koopman Model")
    print("="*50)
    
    # Load the trained CAE+Koopman model
    koopman_model = ERA5_C_FORWARD()
    koopman_model.load_state_dict(
        torch.load('../../../../results/CAE_Koopman/ERA5/model_weights_FTF/forward_model.pt', 
                  weights_only=True, map_location=device)
    )
    koopman_model.C_forward = torch.load(
        '../../../../results/CAE_Koopman/ERA5/model_weights_FTF/C_forward.pt', 
        weights_only=True, map_location=device
    )
    koopman_model.to(device)
    
    print(f"[INFO] CAE+Koopman model loaded successfully")
    print(f"[INFO] Hidden dimension: {koopman_model.hidden_dim}")
    print(f"[INFO] Memory usage after model loading: {get_memory_usage()}")
    
    # ========================================
    # Load Dataset and Extract Latent Dynamics
    # ========================================
    print("\n" + "="*50)
    print("Extracting Latent Space Dynamics")
    print("="*50)
    
    # Load ERA5 dataset
    print("[INFO] Loading ERA5 dataset...")
    era5_train_dataset = ERA5Dataset(
        data_path="../../../../data/ERA5/ERA5_data/train_seq_state.h5",
        seq_length=config['seq_length'],
        min_path="../../../../data/ERA5/ERA5_data/min_val.npy",
        max_path="../../../../data/ERA5/ERA5_data/max_val.npy"
    )
    
    print(f"[INFO] ERA5 dataset loaded with shape: {era5_train_dataset.data.shape}")
    print(f"[INFO] Memory usage after data loading: {get_memory_usage()}")

    # Extract latent dynamics using CAE+Koopman encoder
    print("[INFO] Extracting latent space dynamics...")
    Z_current, Z_next = extract_latent_dynamics_era5(koopman_model, era5_train_dataset, device)

    print(f"Z_current shape: {Z_current.shape} on {Z_current.device}")
    print(f"Z_next shape: {Z_next.shape} on {Z_next.device}")
    print(f"Z_current range: [{Z_current.min():.4f}, {Z_current.max():.4f}]")
    print(f"Z_next range: [{Z_next.min():.4f}, {Z_next.max():.4f}]")
    print(f"[INFO] Memory usage after latent extraction: {get_memory_usage()}")

    # ========================================
    # Train DMD Model
    # ========================================
    print("\n" + "="*50)
    print("Fitting DMD on Latent Space")
    print("="*50)

    # Initialize DMD model with device
    print("[INFO] Initializing DMD model...")
    dmd = TorchDMD(svd_rank=config['rank'], device=device)
    
    print(f"[INFO] Memory usage after DMD initialization: {get_memory_usage()}")

    # Fit the DMD model on latent dynamics
    print("[INFO] Fitting DMD model on latent space...")
    dmd.fit(Z_current, Z_next)

    print(f"[INFO] Memory usage after DMD fitting: {get_memory_usage()}")

    # ========================================
    # Save Models
    # ========================================
    print("\n" + "="*50)
    print("Saving CAE+DMD Model")
    print("="*50)

    # Create save directory
    save_folder = config['save_folder']
    os.makedirs(save_folder, exist_ok=True)
    
    # Save DMD model
    dmd_save_path = os.path.join(save_folder, "dmd_model.pth")
    print(f"[INFO] Saving DMD model to {dmd_save_path}...")
    dmd.save_dmd(dmd_save_path)
    
    # Save CAE encoder/decoder paths and dataset info for future use
    model_info = {
        'encoder_decoder_path': '../../../../results/CAE_Koopman/ERA5/model_weights_FTF/forward_model.pt',
        'C_forward_path': '../../../../results/CAE_Koopman/ERA5/model_weights_FTF/C_forward.pt',
        'hidden_dim': koopman_model.hidden_dim,
        'dataset_info': {
            'min_path': "../../../../data/ERA5/ERA5_data/min_val.npy",
            'max_path': "../../../../data/ERA5/ERA5_data/max_val.npy",
            'data_shape': era5_train_dataset.data.shape
        }
    }
    
    info_save_path = os.path.join(save_folder, "model_info.pth")
    torch.save(model_info, info_save_path)
    print(f"[INFO] Model info saved to {info_save_path}")

    print('[INFO] CAE+DMD ERA5 training completed successfully!')
    print(f"[INFO] Final memory usage: {get_memory_usage()}")
    
    end_time = time.time()
    total_runtime = end_time - start_time
    print(f"[INFO] Total Runtime: {total_runtime:.2f} seconds")


if __name__ == "__main__":
    main()