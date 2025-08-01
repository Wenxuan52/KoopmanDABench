import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import random
import yaml
import pickle
from typing import Optional

import os
import sys
current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
sys.path.append(src_directory)

from src.utils.Dataset import ERA5Dataset

from src.models.CAE_DMD.trainer import set_seed, train_jointly_forward_model, save_training_log


def main():
    from era5_model import ERA5_C_FORWARD

    set_seed(42)
    torch.set_default_dtype(torch.float32)
    
    # Device setup
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using {device} device")
    if torch.cuda.is_available():
        print(f"[INFO] {torch.cuda.get_device_properties(0)}")
    
    # Load configuration
    config_path = "../../../../configs/CAE_DMD_ERA5.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    print("[INFO] Starting Cylinder Flow Model Training")
    print(f"[INFO] Configuration: {config}")
    
    # ========================================
    # Train Forward Model
    # ========================================
    print("\n" + "="*50)
    print("TRAINING FORWARD MODEL")
    print("="*50)

    weighted_M = np.load('../../../../data/ERA5/ERA5_data/weight_matrix.npy')

    era5_train_set = ERA5Dataset(
        data_path="../../../../data/ERA5/ERA5_data/train_seq_state.h5",
        seq_length=config['seq_length'],
        min_path="../../../../data/ERA5/ERA5_data/min_val.npy",
        max_path="../../../../data/ERA5/ERA5_data/max_val.npy"
    )

    era5_val_set = ERA5Dataset(
        data_path="../../../../data/ERA5/ERA5_data/val_seq_state.h5",
        seq_length=config['seq_length'],
        min_path="../../../../data/ERA5/ERA5_data/min_val.npy",
        max_path="../../../../data/ERA5/ERA5_data/max_val.npy"
    )
    
    # Create forward model
    forward_model = ERA5_C_FORWARD()
    
    print("\n" + "="*50)
    print("JOINT TRAINING")
    print("="*50)
    
    train_loss, val_loss = train_jointly_forward_model(
        forward_model=forward_model,
        train_dataset=era5_train_set,
        val_dataset=era5_val_set,
        model_save_folder=config['save_folder'],
        learning_rate=config['learning_rate'],
        lamb=config['lamb'],
        batch_size=config['batch_size'],
        num_epochs=config['num_epochs'],
        decay_step=config['decay_step'],
        decay_rate=config['decay_rate'],
        device=device,
        patience=config['patience'],
        weight_matrix=weighted_M
    )
    
    save_training_log(train_loss, val_loss, f"{config['save_folder']}/losses", 0)

if __name__ == "__main__":
    main()