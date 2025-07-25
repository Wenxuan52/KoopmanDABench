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

from src.models.CAE_Linear.trainer import (set_seed, train_jointly_forward_model, train_cae_pretraining, 
                    train_linear_predictor, save_training_log, load_best_model)


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
    config_path = "../../../configs/CAE_Linear_KOL.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    print("[INFO] Starting Kolmogorov Flow Model Training")
    print(f"[INFO] Configuration: {config}")
    print(f"[INFO] Training mode: {config['train_mode']}")
    
    # Load dynamics dataset
    kol_train_dataset = KolDynamicsDataset(data_path="../../../../data/kolmogorov/RE450_n4/kolmogorov_train_data.npy",
                seq_length = config['seq_length'],
                mean=None,
                std=None)
    
    kol_val_dataset = KolDynamicsDataset(data_path="../../../../data/kolmogorov/RE450_n4/kolmogorov_val_data.npy",
                seq_length = config['seq_length'],
                mean=kol_train_dataset.mean,
                std=kol_train_dataset.std)
    
    # Create forward model
    forward_model = KOL_C_FORWARD()

    if config['train_mode'] == 'jointly':
        print("\n" + "="*50)
        print("JOINT TRAINING")
        print("="*50)
        
        train_loss, val_loss = train_jointly_forward_model(
            forward_model=forward_model,
            train_dataset=kol_train_dataset,
            val_dataset=kol_val_dataset,
            model_save_folder=config['jointly_save_folder'],
            learning_rate=config['learning_rate'],
            lamb=config['lamb'],
            batch_size=config['batch_size'],
            num_epochs=config['S1_epochs'],
            decay_step=config['decay_step'],
            decay_rate=config['decay_rate'],
            device=device
        )
        
        save_training_log(train_loss, val_loss, 'jointly', 
                         f"{config['jointly_save_folder']}/losses", 0)

if __name__ == "__main__":
    main()