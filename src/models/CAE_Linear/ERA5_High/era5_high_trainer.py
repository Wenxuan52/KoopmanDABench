import numpy as np
import torch
import yaml

import os
import sys

current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
sys.path.append(src_directory)

from src.utils.Dataset import ERA5HighDataset
from src.models.CAE_Linear.trainer import set_seed, train_ms_forward_model, save_training_log


def main():
    from era5_high_model import ERA5_C_FORWARD

    set_seed(42)
    torch.set_default_dtype(torch.float32)

    # Device setup
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using {device} device")
    if torch.cuda.is_available():
        print(f"[INFO] {torch.cuda.get_device_properties(0)}")

    # Load configuration
    config_path = "../../../../configs/CAE_Linear_ERA5_HIGH.yaml"
    if not os.path.exists(config_path):
        config_path = "../../../../configs/CAE_Linear_ERA5.yaml"

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    print("[INFO] Starting ERA5 High-Resolution Model Training")
    print(f"[INFO] Configuration: {config}")

    # ========================================
    # Train Forward Model
    # ========================================
    print("\n" + "=" * 50)
    print("TRAINING FORWARD MODEL")
    print("=" * 50)

    train_data_path = config.get(
        "train_data_path",
        "../../../../data/ERA5_high/raw_data/weatherbench_train.h5",
    )
    val_data_path = config.get(
        "val_data_path",
        "../../../../data/ERA5_high/raw_data/weatherbench_test.h5",
    )
    min_path = config.get(
        "min_path",
        "../../../../data/ERA5_high/raw_data/era5high_240x121_min.npy",
    )
    max_path = config.get(
        "max_path",
        "../../../../data/ERA5_high/raw_data/era5high_240x121_max.npy",
    )
    save_folder = config.get(
        "save_folder",
        "../../../../results/CAE_Linear/ERA5_High/3loss_model",
    )

    era5_train_set = ERA5HighDataset(
        data_path=train_data_path,
        seq_length=config['seq_length'],
        min_path=min_path,
        max_path=max_path,
    )

    era5_val_set = ERA5HighDataset(
        data_path=val_data_path,
        seq_length=config['seq_length'],
        min_path=min_path,
        max_path=max_path,
    )

    # Create forward model
    forward_model = ERA5_C_FORWARD()

    print("\n" + "=" * 50)
    print("JOINT TRAINING")
    print("=" * 50)

    weight_matrix_path = config.get("weight_matrix_path")
    weight_matrix = None
    if weight_matrix_path is not None and os.path.exists(weight_matrix_path):
        weight_matrix = np.load(weight_matrix_path)

    train_loss, val_loss = train_ms_forward_model(
        forward_model=forward_model,
        train_dataset=era5_train_set,
        val_dataset=era5_val_set,
        model_save_folder=save_folder,
        learning_rate=config['learning_rate'],
        lamb=config['lamb'],
        lamb_ms=config['lamb_ms'],
        batch_size=config['batch_size'],
        num_epochs=config['num_epochs'],
        decay_step=config['decay_step'],
        decay_rate=config['decay_rate'],
        device=device,
        patience=config['patience'],
        weight_matrix=weight_matrix,
        multi_step=config['multi_step'],
    )

    save_training_log(train_loss, val_loss, f"{save_folder}/losses", 0)


if __name__ == "__main__":
    main()
