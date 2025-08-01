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

from src.utils.Dataset import DamDynamicsDataset

from src.models.CAE_DMD.trainer import set_seed, train_jointly_forward_model, save_training_log


def main():
    from dam_model import DAM_C_FORWARD

    set_seed(42)
    torch.set_default_dtype(torch.float32)
    
    # Device setup
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using {device} device")
    if torch.cuda.is_available():
        print(f"[INFO] {torch.cuda.get_device_properties(0)}")
    
    # Load configuration
    config_path = "../../../../configs/CAE_DMD_DAM.yaml"
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
    
    # Load dynamics dataset
    dam_train_dataset = DamDynamicsDataset(data_path="../../../../data/dam/dam_train_data.npy",
                seq_length = config['seq_length'],
                mean=None,
                std=None)
    
    dam_val_dataset = DamDynamicsDataset(data_path="../../../../data/dam/dam_val_data.npy",
                seq_length = config['seq_length'],
                mean=dam_train_dataset.mean,
                std=dam_train_dataset.std)
    
    # Create forward model
    forward_model = DAM_C_FORWARD()
    
    print("\n" + "="*50)
    print("JOINT TRAINING")
    print("="*50)
    
    train_loss, val_loss = train_jointly_forward_model(
        forward_model=forward_model,
        train_dataset=dam_train_dataset,
        val_dataset=dam_val_dataset,
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