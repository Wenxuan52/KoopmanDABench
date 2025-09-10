import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", ".."))
sys.path.append(src_directory)

from src.utils.Dataset import ERA5Dataset

import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", ".."))
sys.path.append(src_directory)

from src.utils.Dataset import ERA5Dataset

def plot_compact_comparison(raw_data, dmd_data, cae_dmd_data, cae_koopman_data, 
                         cae_linear_data, cae_weaklinear_data, cae_mlp_data, 
                         channel=0, time_indices=[1, 50], save_dir="figures"):
    """
    Plot compact comparison for ERA5 data with specific channel
    
    Args:
        channel: Channel to visualize (0-4)
            0: Geopotential
            1: Temperature  
            2: Humidity
            3: Wind speed (u-component)
            4: Wind speed (v-component)
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

    titles = ['Ground Truth', 'DMD', 'DMD ROM', 'Koopman ROM', 'Linear ROM', 'Weaklinear ROM', 'MLP ROM']
    datas = [raw, dmd, cae_dmd, cae_koopman, cae_linear, cae_weaklinear, cae_mlp]

    # Create figure with 4 rows and 8 columns (7 for plots + 1 for colorbar)
    fig, axes = plt.subplots(nrows=4, ncols=8, figsize=(20, 6))

    # Calculate global min/max for flow fields
    all_pred_data = np.concatenate([d for d in datas], axis=0) 
    vmin_pred, vmax_pred = all_pred_data.min(), all_pred_data.max()

    all_errors = []
    for t_idx in time_indices:
        raw_img = raw[t_idx, 0]
        for pred in [dmd, cae_dmd, cae_koopman, cae_linear, cae_weaklinear, cae_mlp]:
            err = np.abs(pred[t_idx, 0] - raw_img)
            all_errors.append(err)
    all_errors = np.stack(all_errors)
    vmin_err, vmax_err = all_errors.min(), all_errors.max()

    # Base datetime: 2018-05-05 00:00 (start_T = 1000)
    base_datetime = datetime(2018, 5, 5, 0, 0)

    # Plot for each time step
    for time_row, t_idx in enumerate(time_indices):
        # Flow field row (rows 0 and 2)
        flow_row = time_row * 2
        for col in range(7):
            ax = axes[flow_row, col]
            img = datas[col][t_idx]
            im_flow = ax.imshow(img.T, cmap='viridis', vmin=vmin_pred, vmax=vmax_pred)
            ax.axis('off')
            
            # Add titles only for first row
            if flow_row == 0:
                ax.set_title(titles[col], fontsize=18, fontweight='bold')
            
            # Add time step label on the left (each time step is 4 hours)
            if col == 0:
                current_time = base_datetime + timedelta(hours=4*t_idx)
                time_str = current_time.strftime('%m-%d\n%H:%M')
                ax.text(-0.15, 0.5, time_str, fontsize=20, fontweight='bold', 
                        transform=ax.transAxes, rotation=90, va='center', ha='right')
        
        # Add colorbar for flow field row
        cbar_ax_flow = axes[flow_row, 7]
        cbar_flow = fig.colorbar(im_flow, ax=cbar_ax_flow, fraction=1.0)
        cbar_flow.ax.tick_params(labelsize=10)
        cbar_ax_flow.axis('off')
        
        # Error row (rows 1 and 3)
        error_row = time_row * 2 + 1
        raw_img = raw[t_idx]
        
        # Calculate errors for this specific time step
        # errors_this_time = []
        # for pred in [dmd, cae_dmd, cae_koopman, cae_linear, cae_weaklinear, cae_mlp]:
        #     err = np.abs(pred[t_idx] - raw_img)
        #     errors_this_time.append(err)
        # errors_this_time = np.stack(errors_this_time)
        # vmin_err_this, vmax_err_this = errors_this_time.min(), errors_this_time.max()
        
        im_err = None  # Initialize to store the last error image for colorbar
        for col in range(7):
            ax = axes[error_row, col]
            
            if col == 0:
                # First column: display "Error" text aligned with time indices
                ax.axis('off')
                ax.text(-0.15, 0.5, 'Error', fontsize=20, fontweight='bold', 
                        transform=ax.transAxes, rotation=90, va='center', ha='center')
            else:
                # Calculate and plot error
                pred_img = datas[col][t_idx]
                err = np.abs(pred_img - raw_img)
                im_err = ax.imshow(err.T, cmap='magma', vmin=vmin_err, vmax=vmax_err)
                ax.axis('off')
        
        # Add colorbar for error row using the specific time step's range
        if im_err is not None:
            cbar_ax_err = axes[error_row, 7]
            cbar_err = fig.colorbar(im_err, ax=cbar_ax_err, fraction=1.0)
            cbar_err.ax.tick_params(labelsize=10)
            cbar_ax_err.axis('off')

    # Add overall title
    fig.suptitle(f"ERA5 {channel_name} Field: Predictions and Errors Comparison", fontsize=24, fontweight='bold')

    # Adjust layout to be more compact
    plt.tight_layout()
    plt.subplots_adjust(top=0.84, hspace=0.1, wspace=0.02)

    # Save with compression
    fig.savefig(os.path.join(save_dir, f"era5_{channel_name}_compact_comparison.png"), 
                dpi=100, bbox_inches='tight', pad_inches=0.1, 
                facecolor='white', edgecolor='none')

    # # Also save as PDF for better quality if needed
    # fig.savefig(os.path.join(save_dir, f"era5_{channel_name}_compact_comparison.pdf"), 
    #             bbox_inches='tight', pad_inches=0.1)

    plt.close(fig)
    print(f"[INFO] Compact comparison plot for {channel_name} saved to {save_dir}")

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

    # Load all prediction data
    dmd_data = np.load('../../results/DMD/figures/era5_dmd_rollout.npy')
    cae_dmd_data = np.load('../../results/CAE_DMD/figures/era5_rollout.npy')
    cae_koopman_data = np.load('../../results/CAE_Koopman/figures/era5_rollout.npy')
    cae_linear_data = np.load('../../results/CAE_Linear/figures/era5_rollout.npy')
    cae_weaklinear_data = np.load('../../results/CAE_Weaklinear/figures/era5_rollout.npy')
    cae_mlp_data = np.load('../../results/CAE_MLP/figures/era5_rollout.npy')

    print(f"[INFO] Data shapes - DMD: {dmd_data.shape}, CAE_DMD: {cae_dmd_data.shape}")
    
    # Generate plots for all channels
    channel_names = ['Geopotential', 'Temperature', 'Humidity', 'Wind_u', 'Wind_v']
    
    for channel in range(5):
        print(f"[INFO] Generating compact plot for {channel_names[channel]} (channel {channel})")
        # Create compact comparison plot
        plot_compact_comparison(groundtruth, dmd_data, cae_dmd_data, cae_koopman_data, 
                              cae_linear_data, cae_weaklinear_data, cae_mlp_data, 
                              channel=channel, time_indices=[1, 80], save_dir=fig_save_path)
    
    print(f"[INFO] All compact plots saved to {fig_save_path}")