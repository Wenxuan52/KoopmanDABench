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

from src.utils.Dataset import DamDynamicsDataset
from src.models.DMD.base import TorchDMD
from src.models.CAE_Koopman.Dam.dam_model import DAM_C_FORWARD

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

def extract_latent_dynamics(koopman_model, data, device):
    """Extract latent space dynamics from trained CAE+Koopman model"""
    
    koopman_model.eval()
    N, T, C, H, W = data.shape
    hidden_dim = koopman_model.hidden_dim
    
    # Calculate total number of (z_t, z_{t+1}) pairs
    num_pairs = (T - 1) * N
    
    print(f"Input data shape: {data.shape}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Total latent pairs: {num_pairs}")
    
    # Initialize tensors to store latent dynamics
    Z_current = torch.zeros((hidden_dim, num_pairs), dtype=torch.float32, device=device)
    Z_next = torch.zeros((hidden_dim, num_pairs), dtype=torch.float32, device=device)
    
    # Move data to device
    data = data.to(device)
    
    idx = 0
    with torch.no_grad():
        for i in tqdm(range(N), desc="Extracting latent dynamics"):
            sample = data[i]  # shape (T, C, H, W)
            
            # Encode current and next states
            z_current_seq = koopman_model.K_S(sample[:-1])  # shape (T-1, hidden_dim)
            z_next_seq = koopman_model.K_S(sample[1:])      # shape (T-1, hidden_dim)
            
            # Store in matrix format: (hidden_dim, T-1)
            Z_current[:, idx:idx + T - 1] = z_current_seq.T
            Z_next[:, idx:idx + T - 1] = z_next_seq.T
            
            idx += T - 1
    
    print(f"[INFO] Latent dynamics extraction completed")
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
    config_path = "../../../../configs/CAE_DMD_DAM.yaml"
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
    print("[INFO] Starting CAE+DMD Cylinder Flow Model Training")
    print(f"[INFO] Configuration: {config}")
    
    # ========================================
    # Load Pre-trained CAE+Koopman Model
    # ========================================
    print("\n" + "="*50)
    print("Loading Pre-trained CAE+Koopman Model")
    print("="*50)
    
    # Load the trained CAE+Koopman model
    koopman_model = DAM_C_FORWARD()
    koopman_model.load_state_dict(
        torch.load('../../../../results/CAE_Koopman/Dam/model_weights/forward_model.pt', 
                  weights_only=True, map_location=device)
    )
    koopman_model.C_forward = torch.load(
        '../../../../results/CAE_Koopman/Dam/model_weights/C_forward.pt', 
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
    
    # Load dynamics dataset
    print("[INFO] Loading dataset...")
    dam_train_dataset = DamDynamicsDataset(data_path="../../../../data/dam/dam_train_data.npy",
                seq_length = config['seq_length'],
                mean=None,
                std=None)

    train_data = torch.tensor(dam_train_dataset.data, dtype=torch.float32)
    norm_train_data = dam_train_dataset.normalize(train_data)

    print(f"[INFO] Memory usage after data loading: {get_memory_usage()}")

    # Extract latent dynamics using CAE+Koopman encoder
    print("[INFO] Extracting latent space dynamics...")
    Z_current, Z_next = extract_latent_dynamics(koopman_model, norm_train_data, device)

    print(f"Z_current shape: {Z_current.shape} on {Z_current.device}")
    print(f"Z_next shape: {Z_next.shape} on {Z_next.device}")
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
    
    # Save CAE encoder/decoder paths for future use
    model_info = {
        'encoder_decoder_path': '../../../../results/CAE_Koopman/Dam/model_weights/forward_model.pt',
        'hidden_dim': koopman_model.hidden_dim,
        'dataset_stats': {
            'mean': dam_train_dataset.mean,
            'std': dam_train_dataset.std
        }
    }
    
    info_save_path = os.path.join(save_folder, "model_info.pth")
    torch.save(model_info, info_save_path)
    print(f"[INFO] Model info saved to {info_save_path}")

    print('[INFO] CAE+DMD training completed successfully!')
    print(f"[INFO] Final memory usage: {get_memory_usage()}")
    
    end_time = time.time()
    total_runtime = end_time - start_time
    print(f"[INFO] Total Runtime: {total_runtime:.2f} seconds")


if __name__ == "__main__":
    main()