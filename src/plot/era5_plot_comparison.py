import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", ".."))
sys.path.append(src_directory)

from src.utils.Dataset import ERA5Dataset

def plot_era5_comparisons(raw_data, dmd_data, cae_dmd_data, cae_koopman_data, cae_linear_data, 
                         cae_weaklinear_data, cae_mlp_data, channel=0, time_indices=[1, 20, 40, 60, 80, 99], 
                         save_dir="figures"):
    """
    Plot comparison and error plots for ERA5 data
    
    Args:
        raw_data: Ground truth data, shape (T, C, H, W)
        *_data: Prediction data from different methods, shape (T, C, H, W)
        channel: Channel to visualize (0-4)
            0: Geopotential
            1: Temperature  
            2: Humidity
            3: Wind speed (u-component)
            4: Wind speed (v-component)
        time_indices: Time steps to visualize
        save_dir: Directory to save figures
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Channel names for labeling
    channel_names = ['Geopotential', 'Temperature', 'Humidity', 'Wind_u', 'Wind_v']
    channel_name = channel_names[channel]
    
    # Convert to numpy if needed and extract the specified channel
    if isinstance(raw_data, torch.Tensor):
        raw = raw_data[:, channel].numpy()
    else:
        raw = raw_data[:, channel]
    
    # Extract channel from all prediction data
    dmd = dmd_data[:, channel]
    cae_dmd = cae_dmd_data[:, channel] 
    cae_koopman = cae_koopman_data[:, channel]
    cae_linear = cae_linear_data[:, channel]
    cae_weaklinear = cae_weaklinear_data[:, channel]
    cae_mlp = cae_mlp_data[:, channel]
    
    titles = ['Groundtruth', 'DMD', 'DMD ROM', 'Koopman ROM', 'Linear ROM', 'Weaklinear ROM', 'MLP ROM']
    datas = [raw, dmd, cae_dmd, cae_koopman, cae_linear, cae_weaklinear, cae_mlp]
    
    # First figure: Predictions comparison
    fig1, axes1 = plt.subplots(nrows=7, ncols=len(time_indices), figsize=(25, 17))
    all_pred_data = np.concatenate([d for d in datas], axis=0)
    vmin_pred, vmax_pred = all_pred_data.min(), all_pred_data.max()
    
    for row in range(7):
        for col, t in enumerate(time_indices):
            img = datas[row][t]
            ax = axes1[row, col]
            im = ax.imshow(img.T, cmap='viridis', vmin=vmin_pred, vmax=vmax_pred)
            ax.axis('off')
            if row == 0:
                ax.set_title(f"t={t+1000}", fontsize=20, fontweight='bold')
            if col == 0:
                ax.text(-0.1, 0.5, f'{titles[row]}', fontsize=18, fontweight='bold', 
                       transform=ax.transAxes, rotation=90, va='center', ha='right')
            fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    fig1.suptitle(f"ERA5 {channel_name} Field Comparison", fontsize=30, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig1.savefig(os.path.join(save_dir, f"era5_{channel_name}_comparison.png"))
    plt.close(fig1)
    
    # Second figure: Error comparison
    fig2, axes2 = plt.subplots(nrows=7, ncols=len(time_indices), figsize=(25, 17))
    errors = []
    
    # Calculate errors for all predictions
    for t in time_indices:
        raw_img = raw[t]
        for pred in [dmd, cae_dmd, cae_koopman, cae_linear, cae_weaklinear, cae_mlp]:
            err = np.abs(pred[t] - raw_img)
            errors.append(err)
    errors_all = np.stack(errors)
    vmin_err, vmax_err = errors_all.min(), errors_all.max()
    
    for col, t in enumerate(time_indices):
        # First row: Groundtruth
        img_raw = raw[t]
        ax = axes2[0, col]
        im = ax.imshow(img_raw.T, cmap='viridis', vmin=vmin_pred, vmax=vmax_pred)
        ax.axis('off')
        if col == 0:
            ax.text(-0.1, 0.5, 'Groundtruth', fontsize=18, fontweight='bold', 
                   transform=ax.transAxes, rotation=90, va='center', ha='right')
        ax.set_title(f"t={t+1000}", fontsize=18)
        fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Remaining rows: Error plots
        for i, pred in enumerate([dmd, cae_dmd, cae_koopman, cae_linear, cae_weaklinear, cae_mlp], start=1):
            err = np.abs(pred[t] - img_raw)
            ax = axes2[i, col]
            im_err = ax.imshow(err.T, cmap='magma', vmin=vmin_err, vmax=vmax_err)
            ax.axis('off')
            if col == 0:
                ax.text(-0.1, 0.5, f"{titles[i]}", fontsize=18, fontweight='bold', 
                       transform=ax.transAxes, rotation=90, va='center', ha='right')
            fig2.colorbar(im_err, ax=ax, fraction=0.046, pad=0.04)
    
    fig2.suptitle(f"ERA5 {channel_name} Field Error Comparison", fontsize=30, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig2.savefig(os.path.join(save_dir, f"era5_{channel_name}_error.png"))
    plt.close(fig2)

if __name__ == '__main__':
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    fig_save_path = '../../results/Comparison/figures/'
    start_T = 1000
    prediction_step = 100
    
    # Load dataset
    print("[INFO] Loading ERA5 datasets...")
    era5_test_dataset = ERA5Dataset(
        data_path="../../data/ERA5/ERA5_data/test_seq_state.h5",
        seq_length=12,
        min_path="../../data/ERA5/ERA5_data/min_val.npy",
        max_path="../../data/ERA5/ERA5_data/max_val.npy"
    )
    
    denorm = era5_test_dataset.denormalizer()
    
    # Prepare groundtruth data
    raw_test_data = era5_test_dataset.data  # shape: [N, H, W, C]
    groundtruth = raw_test_data[start_T:start_T + prediction_step, ...]  # (100, 64, 32, 5)
    groundtruth = torch.tensor(groundtruth, dtype=torch.float32).permute(0, 3, 1, 2)  # (100, 5, 64, 32)
    
    print(f"[INFO] Groundtruth shape: {groundtruth.shape}")
    
    # Load prediction results
    dmd_data = np.load('../../results/DMD/figures/era5_dmd_rollout.npy')
    cae_dmd_data = np.load('../../results/CAE_DMD/figures/era5_rollout.npy')
    cae_koopman_data = np.load('../../results/CAE_Koopman/figures/era5_rollout.npy')
    cae_linear_data = np.load('../../results/CAE_Linear/figures/era5_rollout.npy')
    cae_weaklinear_data = np.load('../../results/CAE_Weaklinear/figures/era5_rollout.npy')
    cae_mlp_data = np.load('../../results/CAE_MLP/figures/era5_rollout.npy')
    
    print(f"[INFO] Prediction data shapes:")
    print(f"  DMD: {dmd_data.shape}")
    print(f"  CAE_DMD: {cae_dmd_data.shape}")
    
    # Generate plots for all channels
    channel_names = ['Geopotential', 'Temperature', 'Humidity', 'Wind_u', 'Wind_v']
    
    for channel in range(5):
        print(f"[INFO] Generating plots for {channel_names[channel]} (channel {channel})")
        plot_era5_comparisons(
            groundtruth, dmd_data, cae_dmd_data, cae_koopman_data, 
            cae_linear_data, cae_weaklinear_data, cae_mlp_data,
            channel=channel,
            time_indices=[1, 20, 40, 60, 80, 99],
            save_dir=fig_save_path
        )
    
    print(f"[INFO] All plots saved to {fig_save_path}")