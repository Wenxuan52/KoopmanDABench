import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", ".."))
sys.path.append(src_directory)

from src.utils.Dataset import CylinderDynamicsDataset

def plot_compact_comparison(raw_data, dmd_data_uv, cae_dmd_data_uv, cae_koopman_data_uv, 
                         cae_linear_data_uv, cae_weaklinear_data_uv, cae_mlp_data_uv, 
                         time_indices=[10, 100], save_dir="figures"):
    os.makedirs(save_dir, exist_ok=True)

    # Convert all inputs to numpy arrays
    raw = raw_data.numpy()
    dmd = cae_linear_data_uv
    cae_dmd = cae_weaklinear_data_uv
    cae_koopman = cae_dmd_data_uv
    cae_linear = cae_mlp_data_uv
    cae_weaklinear = cae_koopman_data_uv
    
    # Updated titles according to requirements
    titles = ['Ground Truth', 'PFNN', 'KRR', 'KAE', 'VAE', 'Ours']
    datas = [raw, dmd, cae_dmd, cae_koopman, cae_linear, cae_weaklinear]

    # Create figure with 4 rows and 7 columns (6 for plots + 1 for colorbar)
    fig, axes = plt.subplots(nrows=4, ncols=7, figsize=(18, 10))

    # Calculate global min/max for flow fields
    all_pred_data = np.concatenate([d[:, 0] for d in datas], axis=0) 
    vmin_pred, vmax_pred = all_pred_data.min(), all_pred_data.max()

    all_errors = []
    for t_idx in time_indices:
        raw_img = raw[t_idx, 0]
        for pred in [dmd, cae_dmd, cae_koopman, cae_linear, cae_weaklinear]:
            err = np.abs(pred[t_idx, 0] - raw_img)
            all_errors.append(err)
    all_errors = np.stack(all_errors)
    vmin_err, vmax_err = all_errors.min(), all_errors.max()

    # Time labels corresponding to time_indices [10, 100]
    time_labels = ["t = 0.71s", "t = 0.75s"]

    # Plot for each time step
    for time_row, t_idx in enumerate(time_indices):
        # Flow field row (rows 0 and 2)
        flow_row = time_row * 2
        for col in range(6):  # Only 6 models now
            ax = axes[flow_row, col]
            img = datas[col][t_idx, 0]
            im_flow = ax.imshow(img, cmap='viridis', vmin=vmin_pred, vmax=vmax_pred)
            ax.axis('off')
            
            # Add titles only for first row
            if flow_row == 0:
                ax.set_title(titles[col], fontsize=18, fontweight='bold')
            
            # Add time step label on the left
            if col == 0:
                ax.text(-0.3, 0.5, time_labels[time_row], fontsize=20, fontweight='bold', 
                        transform=ax.transAxes, rotation=90, va='center', ha='center')
        
        # Add colorbar for flow field row
        cbar_ax_flow = axes[flow_row, 6]
        cbar_flow = fig.colorbar(im_flow, ax=cbar_ax_flow, fraction=1.0)
        cbar_flow.ax.tick_params(labelsize=10)
        cbar_ax_flow.axis('off')
        
        # Error row (rows 1 and 3)
        error_row = time_row * 2 + 1
        raw_img = raw[t_idx, 0]
        
        im_err = None  # Initialize to store the last error image for colorbar
        for col in range(6):  # Only 6 models now
            ax = axes[error_row, col]
            
            if col == 0:
                # First column: display "Error" text aligned with time indices
                ax.axis('off')
                ax.text(-0.15, 0.5, 'Error', fontsize=20, fontweight='bold', 
                        transform=ax.transAxes, rotation=90, va='center', ha='center')
            else:
                # Calculate and plot error
                pred_img = datas[col][t_idx, 0]
                err = np.abs(pred_img - raw_img)
                im_err = ax.imshow(err, cmap='magma', vmin=vmin_err, vmax=vmax_err)
                ax.axis('off')
        
        # Add colorbar for error row
        if im_err is not None:
            cbar_ax_err = axes[error_row, 6]
            cbar_err = fig.colorbar(im_err, ax=cbar_ax_err, fraction=1.0)
            cbar_err.ax.tick_params(labelsize=10)
            cbar_ax_err.axis('off')

    # Add overall title
    fig.suptitle("Kármán vortex street: Continuous Prediction from t=0.7s", fontsize=24, fontweight='bold')

    # Adjust layout to be more compact
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, hspace=0.1, wspace=0.02)

    # Save as PDF
    fig.savefig(os.path.join(save_dir, "cyl_compact_comparison.pdf"), 
                bbox_inches='tight', pad_inches=0.1, 
                facecolor='white', edgecolor='none')

    plt.close(fig)
    print(f"[INFO] Compact comparison plot saved to {save_dir}")


if __name__ == '__main__':
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    fig_save_path = '../../results/Comparison/figures/'
    start_T = 700
    prediction_step = 300
    val_idx = 3
    
    # Load dataset
    print("[INFO] Loading datasets...")
    cyl_train_dataset = CylinderDynamicsDataset(
        data_path="../../data/cylinder/cylinder_train_data.npy",
        seq_length=12,
        mean=None,
        std=None
    )
    
    cyl_val_dataset = CylinderDynamicsDataset(
        data_path="../../data/cylinder/cylinder_val_data.npy",
        seq_length=12,
        mean=cyl_train_dataset.mean,
        std=cyl_train_dataset.std
    )
    
    denorm = cyl_val_dataset.denormalizer()

    # Prepare groundtruth data
    groundtruth = cyl_val_dataset.data[val_idx, start_T:start_T + prediction_step, ...]
    groundtruth = torch.tensor(groundtruth, dtype=torch.float32)
    
    # Convert to velocity magnitude
    raw_data_uv = (groundtruth[:, 0, :, :] ** 2 + groundtruth[:, 1, :, :] ** 2) ** 0.5
    raw_data_uv = raw_data_uv.unsqueeze(1)
    print(f"[INFO] Groundtruth shape: {raw_data_uv.shape}")

    # Load all prediction data
    dmd_data_uv = np.load('../../results/DMD/figures/cyl_rollout.npy')
    cae_dmd_data_uv = np.load('../../results/CAE_DMD/figures/cyl_rollout.npy')
    cae_koopman_data_uv = np.load('../../results/CAE_Koopman/figures/cyl_rollout.npy')
    cae_linear_data_uv = np.load('../../results/CAE_Linear/figures/cyl_rollout.npy')
    cae_weaklinear_data_uv = np.load('../../results/CAE_Weaklinear/figures/cyl_rollout.npy')
    cae_mlp_data_uv = np.load('../../results/CAE_MLP/figures/cyl_rollout.npy')

    print(f"[INFO] Data shapes - DMD: {dmd_data_uv.shape}, CAE_DMD: {cae_dmd_data_uv.shape}")
    
    # Create compact comparison plot with updated time indices
    plot_compact_comparison(raw_data_uv, dmd_data_uv, cae_dmd_data_uv, cae_koopman_data_uv, 
                          cae_linear_data_uv, cae_weaklinear_data_uv, cae_mlp_data_uv, 
                          time_indices=[10, 20], save_dir=fig_save_path)