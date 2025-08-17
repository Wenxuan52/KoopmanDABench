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

from trainer import (
    set_seed, 
    train_cae_dmd_multistage,
    train_stage1_reconstruction,
    train_stage2_dmd_fitting, 
    train_stage3_joint_optimization,
    save_training_log
)

import os
import sys
current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
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
    
    # Load configuration
    config_path = "../../../../configs/CAE_DMD_CYL.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    print("[INFO] Starting Cylinder Flow CAE+DMD Model Training")
    print(f"[INFO] Configuration: {config}")
    
    # Load dynamics dataset
    cyl_train_dataset = CylinderDynamicsDataset(
        data_path="../../../../data/cylinder/cylinder_train_data.npy",
        seq_length=config['seq_length'],
        mean=None,
        std=None
    )
    
    cyl_val_dataset = CylinderDynamicsDataset(
        data_path="../../../../data/cylinder/cylinder_val_data.npy",
        seq_length=config['seq_length'],
        mean=cyl_train_dataset.mean,
        std=cyl_train_dataset.std
    )
    
    # Create DMD forward model
    dmd_model = CYLINDER_C_FORWARD()
    
    print("\n" + "="*60)
    print("MULTI-STAGE CAE+DMD TRAINING")
    print("="*60)
    
    # Prepare stage configurations
    stage1_config = {
        'learning_rate': config['stage1']['learning_rate'],
        'weight_decay': config['stage1'].get('weight_decay', 0),
        'batch_size': config['stage1']['batch_size'],
        'num_epochs': config['stage1']['num_epochs'],
        'decay_step': config['stage1'].get('decay_step', 20),
        'decay_rate': config['stage1'].get('decay_rate', 0.8),
        'gradclip': config['stage1'].get('gradclip', 1.0),
        'patience': config['stage1'].get('patience', 30)
    }
    
    stage2_config = {
        'batch_size': config['stage2']['batch_size']
    }
    
    stage3_config = {
        'learning_rate': config['stage3']['learning_rate'],
        'lamb': config['stage3']['lamb'],
        'weight_decay': config['stage3'].get('weight_decay', 0),
        'batch_size': config['stage3']['batch_size'],
        'num_epochs': config['stage3']['num_epochs'],
        'decay_step': config['stage3'].get('decay_step', 15),
        'decay_rate': config['stage3'].get('decay_rate', 0.8),
        'gradclip': config['stage3'].get('gradclip', 1.0),
        'patience': config['stage3'].get('patience', 20)
    }
    
    # Run multi-stage training
    training_results = train_cae_dmd_multistage(
        dmd_model=dmd_model,
        train_dataset=cyl_train_dataset,
        val_dataset=cyl_val_dataset,
        model_save_folder=config['save_folder'],
        stage1_config=stage1_config,
        stage2_config=stage2_config,
        stage3_config=stage3_config,
        device=device
    )
    
    print(f"\n[INFO] Training completed successfully!")
    print(f"[INFO] Results saved to: {config['save_folder']}")


if __name__ == "__main__":
    main()