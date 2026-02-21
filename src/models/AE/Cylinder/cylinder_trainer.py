import numpy as np
import torch
import yaml
import os
import sys

current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", ".."))
sys.path.append(src_directory)

import os
import sys
current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
sys.path.append(src_directory)

from src.utils.Dataset import CylinderDynamicsDataset

from src.models.AE.trainer import (
    set_seed,
    train_ms_forward_model,
    save_training_log,
)


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
    config_path = "../../../../configs/CAE_MLP_CYL.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    print("[INFO] Starting Cylinder Flow Model Training")
    print(f"[INFO] Configuration: {config}")
    print(f"[INFO] Training mode: {config['train_mode']}")
    
    # Load datasets
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
    
    # Create forward model
    forward_model = CYLINDER_C_FORWARD()
    
    # Training based on mode
    if config['train_mode'] == 'jointly':
        print("\n" + "="*50)
        print("LATENT + MULTI-STEP TRAINING")
        print("="*50)
        
        train_loss, val_loss = train_ms_forward_model(
            forward_model=forward_model,
            train_dataset=cyl_train_dataset,
            val_dataset=cyl_val_dataset,
            model_save_folder=config['jointly_save_folder'],
            learning_rate=config['learning_rate'],
            lamb=config['lamb'],
            lamb_ms=config['lamb_ms'],
            batch_size=config['batch_size'],
            num_epochs=config['S1_epochs'],
            decay_step=config['decay_step'],
            decay_rate=config['decay_rate'],
            device=device,
            multi_step=config['multi_step']
        )
        
        save_training_log(train_loss, val_loss, 'jointly', 
                         f"{config['jointly_save_folder']}/losses", 0)
    
    else:
        raise ValueError(f"Unknown training mode: {config['train_mode']}. "
                        "Supported modes: 'jointly', 'separately', 'multiply'")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    print("="*50)


if __name__ == "__main__":
    main()
