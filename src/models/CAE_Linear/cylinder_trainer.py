import numpy as np
import torch
import yaml
import os
import sys

current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", ".."))
sys.path.append(src_directory)

from src.utils.Dataset import CylinderDynamicsDataset
from src.utils.utils import dict2namespace, count_parameters

from trainer import (set_seed, train_jointly_forward_model, train_cae_pretraining, 
                    train_linear_predictor, save_training_log, load_best_model)


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
    config_path = "../../../configs/CAE_Linear.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    print("[INFO] Starting Cylinder Flow Model Training")
    print(f"[INFO] Configuration: {config}")
    print(f"[INFO] Training mode: {config['train_mode']}")
    
    # Load datasets
    cyl_train_dataset = CylinderDynamicsDataset(
        data_path="../../../data/cylinder/cylinder_train_data.npy",
        seq_length=config['seq_length'],
        mean=None,
        std=None
    )
    
    cyl_val_dataset = CylinderDynamicsDataset(
        data_path="../../../data/cylinder/cylinder_val_data.npy",
        seq_length=config['seq_length'],
        mean=cyl_train_dataset.mean,
        std=cyl_train_dataset.std
    )
    
    # Create forward model
    forward_model = CYLINDER_C_FORWARD()
    
    # Training based on mode
    if config['train_mode'] == 'jointly':
        print("\n" + "="*50)
        print("JOINT TRAINING")
        print("="*50)
        
        train_loss, val_loss = train_jointly_forward_model(
            forward_model=forward_model,
            train_dataset=cyl_train_dataset,
            val_dataset=cyl_val_dataset,
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
    
    elif config['train_mode'] == 'separately':
        print("\n" + "="*50)
        print("TWO-STAGE TRAINING")
        print("="*50)
        
        # Stage 1: CAE Pre-training
        print("\nStage 1: CAE Pre-training")
        train_loss_1, val_loss_1 = train_cae_pretraining(
            forward_model=forward_model,
            train_dataset=cyl_train_dataset,
            val_dataset=cyl_val_dataset,
            model_save_folder=config['cae_save_folder'],
            learning_rate=config['learning_rate'],
            batch_size=config['batch_size'],
            num_epochs=config['S1_epochs'],
            decay_step=config['decay_step'],
            decay_rate=config['decay_rate'],
            device=device
        )
        
        save_training_log(train_loss_1, val_loss_1, 'cae_pretraining',
                         f"{config['cae_save_folder']}/losses", 1)
        
        # Stage 2: Load best CAE model and train linear predictor
        print("\nStage 2: Linear Predictor Training")
        forward_model = load_best_model(forward_model, config['cae_save_folder'], device)
        
        train_loss_2, val_loss_2 = train_linear_predictor(
            forward_model=forward_model,
            train_dataset=cyl_train_dataset,
            val_dataset=cyl_val_dataset,
            model_save_folder=config['linear_save_folder'],
            learning_rate=config['learning_rate'],
            lamb=config['lamb'],
            batch_size=config['batch_size'],
            num_epochs=config['S2_epochs'],
            decay_step=config['decay_step'],
            decay_rate=config['decay_rate'],
            device=device
        )
        
        save_training_log(train_loss_2, val_loss_2, 'linear_predictor',
                         f"{config['linear_save_folder']}/losses", 2)
    
    elif config['train_mode'] == 'multiply':
        print("\n" + "="*50)
        print("THREE-STAGE MULTIPLY TRAINING")
        print("="*50)
        
        # Stage 1: CAE Pre-training
        print("\nStage 1: CAE Pre-training")
        train_loss_1, val_loss_1 = train_cae_pretraining(
            forward_model=forward_model,
            train_dataset=cyl_train_dataset,
            val_dataset=cyl_val_dataset,
            model_save_folder=config['cae_save_folder'],
            learning_rate=config['learning_rate'],
            batch_size=config['batch_size'],
            num_epochs=config['S1_epochs'],
            decay_step=config['decay_step'],
            decay_rate=config['decay_rate'],
            device=device
        )
        
        save_training_log(train_loss_1, val_loss_1, 'cae_pretraining',
                         f"{config['cae_save_folder']}/losses", 1)
        
        # Stage 2: Load best CAE model and train linear predictor
        print("\nStage 2: Linear Predictor Training")
        forward_model = load_best_model(forward_model, config['cae_save_folder'], device)
        
        train_loss_2, val_loss_2 = train_linear_predictor(
            forward_model=forward_model,
            train_dataset=cyl_train_dataset,
            val_dataset=cyl_val_dataset,
            model_save_folder=config['linear_save_folder'],
            learning_rate=config['learning_rate'],
            lamb=config['lamb'],
            batch_size=config['batch_size'],
            num_epochs=config['S2_epochs'],
            decay_step=config['decay_step'],
            decay_rate=config['decay_rate'],
            device=device
        )
        
        save_training_log(train_loss_2, val_loss_2, 'linear_predictor',
                         f"{config['linear_save_folder']}/losses", 2)
        
        # Stage 3: Load best linear model and joint fine-tuning
        print("\nStage 3: Joint Fine-tuning")
        forward_model = load_best_model(forward_model, config['linear_save_folder'], device)
        
        train_loss_3, val_loss_3 = train_jointly_forward_model(
            forward_model=forward_model,
            train_dataset=cyl_train_dataset,
            val_dataset=cyl_val_dataset,
            model_save_folder=config['jointly_save_folder'],
            learning_rate=config['learning_rate'] * 0.1,  # Lower learning rate for fine-tuning
            lamb=config['lamb'],
            batch_size=config['batch_size'],
            num_epochs=config['S3_epochs'],
            decay_step=config['decay_step'],
            decay_rate=config['decay_rate'],
            device=device
        )
        
        save_training_log(train_loss_3, val_loss_3, 'joint_finetuning',
                         f"{config['jointly_save_folder']}/losses", 3)
    
    else:
        raise ValueError(f"Unknown training mode: {config['train_mode']}. "
                        "Supported modes: 'jointly', 'separately', 'multiply'")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    print("="*50)


if __name__ == "__main__":
    main()