import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", ".."))
sys.path.append(src_directory)

from src.utils.Dataset import ERA5Dataset

if __name__ == '__main__':
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    fig_save_path = '../../results/Comparison/figures/'
    os.makedirs(fig_save_path, exist_ok=True)
    start_T = 1000
    prediction_step = 100
    time_indices = [1, 20, 40, 60, 80, 99]
    
    # Channel names and information
    channel_names = ['Geopotential', 'Temperature', 'Humidity', 'Wind_u', 'Wind_v']
    
    # Load dataset
    print("[INFO] Loading ERA5 datasets...")
    era5_test_dataset = ERA5Dataset(
        data_path="../../data/ERA5/ERA5_data/test_seq_state.h5",
        seq_length=12,
        min_path="../../data/ERA5/ERA5_data/min_val.npy",
        max_path="../../data/ERA5/ERA5_data/max_val.npy"
    )
    
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
    
    titles = ['Groundtruth', 'DMD', 'DMD ROM', 'Koopman ROM', 'Linear ROM', 'Weaklinear ROM', 'MLP ROM']
    
    # Process each channel separately
    for channel in range(5):
        print(f"[INFO] Processing {channel_names[channel]} (channel {channel})")
        
        # Extract channel data
        raw_data = groundtruth[:, channel].numpy()
        datas = [
            raw_data,
            dmd_data[:, channel],
            cae_dmd_data[:, channel],
            cae_koopman_data[:, channel],
            cae_linear_data[:, channel],
            cae_weaklinear_data[:, channel],
            cae_mlp_data[:, channel]
        ]
        
        # Calculate color ranges
        all_pred_data = np.concatenate([d for d in datas], axis=0)
        vmin_pred, vmax_pred = all_pred_data.min(), all_pred_data.max()
        
        # Calculate all errors for error color range
        errors_all = []
        for t in time_indices:
            raw_img = raw_data[t]
            for pred in datas[1:]:  # Skip groundtruth
                err = np.abs(pred[t] - raw_img)
                errors_all.append(err)
        errors_all = np.stack(errors_all)
        vmin_err, vmax_err = errors_all.min(), errors_all.max()
        
        # Create integrated plot: 13 rows (1 groundtruth + 6 prediction + 6 error), 6 columns (time steps)
        fig, axes = plt.subplots(nrows=13, ncols=len(time_indices), figsize=(26, 30))
        
        for col, t in enumerate(time_indices):
            row_idx = 0
            
            # First row: Groundtruth
            img = raw_data[t]
            ax = axes[row_idx, col]
            im = ax.imshow(img.T, cmap='viridis', vmin=vmin_pred, vmax=vmax_pred)
            ax.axis('off')
            if col == 0:
                ax.text(-0.1, 0.5, 'Groundtruth', fontsize=18, fontweight='bold', 
                       transform=ax.transAxes, rotation=90, va='center', ha='right')
            ax.set_title(f"t={t+1000}", fontsize=20, fontweight='bold')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            row_idx += 1
            
            # Each method followed by its error
            for i in range(1, 7):  # 6 prediction methods
                # Prediction row
                pred_img = datas[i][t]
                ax_pred = axes[row_idx, col]
                im_pred = ax_pred.imshow(pred_img.T, cmap='viridis', vmin=vmin_pred, vmax=vmax_pred)
                ax_pred.axis('off')
                if col == 0:
                    ax_pred.text(-0.1, 0.5, f'{titles[i]}', fontsize=18, fontweight='bold', 
                               transform=ax_pred.transAxes, rotation=90, va='center', ha='right')
                fig.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)
                row_idx += 1
                
                # Error row (immediately after prediction)
                err_img = np.abs(pred_img - raw_data[t])
                ax_err = axes[row_idx, col]
                im_err = ax_err.imshow(err_img.T, cmap='magma', vmin=vmin_err, vmax=vmax_err)
                ax_err.axis('off')
                if col == 0:
                    ax_err.text(-0.1, 0.5, f'Error', fontsize=18, fontweight='bold', 
                              transform=ax_err.transAxes, rotation=90, va='center', ha='right')
                fig.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.04)
                row_idx += 1
        
        fig.suptitle(f"ERA5 {channel_names[channel]} Field: Predictions and Errors", fontsize=30, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        # Save figure
        save_filename = f"era5_{channel_names[channel]}_comparison_integrated.png"
        fig.savefig(os.path.join(fig_save_path, save_filename), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"[INFO] Integrated comparison plot for {channel_names[channel]} saved to: {os.path.join(fig_save_path, save_filename)}")
    
    print(f"[INFO] All integrated comparison plots saved to {fig_save_path}")