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

import os
import sys
current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
sys.path.append(src_directory)

from src.utils.Dataset import KolDynamicsDataset

from src.models.WAE.trainer import set_seed, train_ms_latent_linear_model, save_training_log


def main():
    from kol_model import KOL_C_FORWARD

    set_seed(42)
    torch.set_default_dtype(torch.float32)
    
    # Device setup
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using {device} device")
    if torch.cuda.is_available():
        print(f"[INFO] {torch.cuda.get_device_properties(0)}")
    
    # Load configuration
    config_path = "../../../../configs/CAE_Weaklinear_KOL.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    print("[INFO] Starting Kolmogorov Flow Model Training")
    print(f"[INFO] Configuration: {config}")

    if config['load_model']:
        print(f"[INFO] Loading checkpoint from {config['ckpt_path']}")
        checkpoint = torch.load(config['ckpt_path'], map_location=device)

        forward_model.load_state_dict(checkpoint["model_state"])
    
    # ========================================
    # Train Forward Model
    # ========================================
    print("\n" + "="*50)
    print("TRAINING FORWARD MODEL")
    print("="*50)
    
    # Load dynamics dataset
    kol_train_dataset = KolDynamicsDataset(data_path="../../../../data/kol/kolmogorov_train_data.npy",
                seq_length = config['seq_length'],
                mean=None,
                std=None)
    
    kol_val_dataset = KolDynamicsDataset(data_path="../../../../data/kol/kolmogorov_val_data.npy",
                seq_length = config['seq_length'],
                mean=kol_train_dataset.mean,
                std=kol_train_dataset.std)
    
    # Create forward model
    forward_model = KOL_C_FORWARD()

    print("\n" + "="*50)
    print("JOINT TRAINING")
    print("="*50)
    
    train_loss, val_loss = train_ms_latent_linear_model(
        forward_model=forward_model,
        train_dataset=kol_train_dataset,
        val_dataset=kol_val_dataset,
        model_save_folder=config['save_folder'],
        learning_rate=config['learning_rate'],
        lamb=config['lamb'],
        lamb_multi=config['lamb_ms'],
        lamb_latent=config['lamb_latent'],
        batch_size=config['batch_size'],
        num_epochs=config['num_epochs'],
        decay_step=config['decay_step'],
        weight_decay=config['weight_decay'],
        decay_rate=config['decay_rate'],
        device=device,
        patience=config['patience'],
        multi_step=config['multi_step']
    )
    
    save_training_log(train_loss, val_loss, f"{config['save_folder']}/losses", 0)

if __name__ == "__main__":
    main()