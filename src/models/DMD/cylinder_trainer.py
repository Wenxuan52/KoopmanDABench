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

from trainer import set_seed, train_jointly_forward_model, save_training_log

import os
import sys
current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", ".."))
sys.path.append(src_directory)

from src.utils.Dataset import CylinderDynamicsDataset


def main():
    from cylinder_model import CYLINDER_C_FORWARD

    set_seed(42)
    torch.set_default_dtype(torch.float32)
    
    # Device setup
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using {device} device")
    if torch.cuda.is_available():
        print(f"[INFO] {torch.cuda.get_device_properties(0)}")
    
    # Training configuration
    config = {
        'save_folder': "cyl_model_weights",
        'seq_length': 12,
        'num_epochs': 300,
        'decay_step': 20,
        'batch_size': 24,
        'learning_rate': 1e-3,
        'decay_rate': 0.8,
        'lamb': 1.0,
        'patience': 30
    }
    
    print("[INFO] Starting Cylinder Flow Model Training")
    print(f"[INFO] Configuration: {config}")
    
    # ========================================
    # Train Forward Model
    # ========================================
    print("\n" + "="*50)
    print("TRAINING FORWARD MODEL")
    print("="*50)
    
    # Load dynamics dataset
    cyl_train_dataset = CylinderDynamicsDataset(data_path="../../../data/cylinder/cylinder_train_data.npy",
                seq_length = config['seq_length'],
                mean=None,
                std=None)
    
    cyl_val_dataset = CylinderDynamicsDataset(data_path="../../../data/cylinder/cylinder_val_data.npy",
                seq_length = config['seq_length'],
                mean=cyl_train_dataset.mean,
                std=cyl_train_dataset.std)
    
    # Create forward model
    forward_model = CYLINDER_C_FORWARD()
    
    print("\n" + "="*50)
    print("JOINT TRAINING")
    print("="*50)
    
    train_loss, val_loss = train_jointly_forward_model(
        forward_model=forward_model,
        train_dataset=cyl_train_dataset,
        val_dataset=cyl_val_dataset,
        model_save_folder=config['save_folder'],
        learning_rate=config['learning_rate'],
        lamb=config['lamb'],
        batch_size=config['batch_size'],
        num_epochs=config['num_epochs'],
        decay_step=config['decay_step'],
        decay_rate=config['decay_rate'],
        device=device,
        patience=config['patience']
    )
    
    save_training_log(train_loss, val_loss, f"{config['save_folder']}/losses", 0)

if __name__ == "__main__":
    main()