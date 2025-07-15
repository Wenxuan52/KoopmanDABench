import numpy as np
import torch
import random
import json
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import yaml
from typing import Optional

from trainer import set_seed, train_jointly_forward_model, save_training_log

import os
import sys
current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", ".."))
sys.path.append(src_directory)

from src.utils.Dataset import KolDynamicsDataset


def main():
    from kol_model import KOL_C_FORWARD

    set_seed(42)
    torch.set_default_dtype(torch.float32)
    
    # Device setup
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using {device} device")
    if torch.cuda.is_available():
        print(f"[INFO] {torch.cuda.get_device_properties(0)}")
    
    # Training configuration
    config = {
        'save_folder': "kol_model_weights",
        'seq_length': 10,
        'num_epochs': 100,
        'decay_step': 20,
        'weight_decay': 1e-4,
        'batch_size': 8,
        'learning_rate': 1e-3,
        'decay_rate': 0.8,
        'lamb': 1.0,
    }
    
    print("[INFO] Starting Kolmogorov Flow Model Training")
    print(f"[INFO] Configuration: {config}")
    
    # ========================================
    # Train Forward Model
    # ========================================
    print("\n" + "="*50)
    print("TRAINING FORWARD MODEL")
    print("="*50)
    
    # Load dynamics dataset
    kol_train_dataset = KolDynamicsDataset(data_path="../../../data/kolmogorov/RE450_n4/kolmogorov_train_data.npy",
                seq_length = config['seq_length'],
                mean=None,
                std=None)
    
    kol_val_dataset = KolDynamicsDataset(data_path="../../../data/kolmogorov/RE450_n4/kolmogorov_val_data.npy",
                seq_length = config['seq_length'],
                mean=kol_train_dataset.mean,
                std=kol_train_dataset.std)
    
    # Create forward model
    forward_model = KOL_C_FORWARD()

    print("\n" + "="*50)
    print("JOINT TRAINING")
    print("="*50)
    
    train_loss, val_loss = train_jointly_forward_model(
        forward_model=forward_model,
        train_dataset=kol_train_dataset,
        val_dataset=kol_val_dataset,
        model_save_folder=config['save_folder'],
        learning_rate=config['learning_rate'],
        lamb=config['lamb'],
        batch_size=config['batch_size'],
        num_epochs=config['num_epochs'],
        decay_step=config['decay_step'],
        weight_decay=config['weight_decay'],
        decay_rate=config['decay_rate'],
        device=device
    )
    
    save_training_log(train_loss, val_loss, f"{config['save_folder']}/losses", 0)

if __name__ == "__main__":
    main()