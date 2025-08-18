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

if __name__ == '__main__':
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    fig_save_path = '../../results/Comparison/figures/'
    os.makedirs(fig_save_path, exist_ok=True)
    start_T = 700
    prediction_step = 300
    val_idx = 3
    time_indices = [1, 50, 100, 150, 200, 250, 299]
    
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

    # Prepare groundtruth data
    groundtruth = cyl_val_dataset.data[val_idx, start_T:start_T + prediction_step, ...]
    groundtruth = torch.tensor(groundtruth, dtype=torch.float32)
    
    # Convert to velocity magnitude
    raw_data_uv = (groundtruth[:, 0, :, :] ** 2 + groundtruth[:, 1, :, :] ** 2) ** 0.5
    raw_data_uv = raw_data_uv.unsqueeze(1).numpy()

    # Load prediction results
    dmd_data_uv = np.load('../../results/DMD/figures/cyl_rollout.npy')
    cae_dmd_data_uv = np.load('../../results/CAE_DMD/figures/cyl_rollout.npy')
    cae_koopman_data_uv = np.load('../../results/CAE_Koopman/figures/cyl_rollout.npy')
    cae_linear_data_uv = np.load('../../results/CAE_Linear/figures/cyl_rollout.npy')
    cae_weaklinear_data_uv = np.load('../../results/CAE_Weaklinear/figures/cyl_rollout.npy')
    cae_mlp_data_uv = np.load('../../results/CAE_MLP/figures/cyl_rollout.npy')

    titles = ['Groundtruth', 'DMD', 'DMD ROM', 'Koopman ROM', 'Linear ROM', 'Weaklinear ROM', 'MLP ROM']
    datas = [raw_data_uv, dmd_data_uv, cae_dmd_data_uv, cae_koopman_data_uv, cae_linear_data_uv, cae_weaklinear_data_uv, cae_mlp_data_uv]

    # Calculate color ranges
    all_pred_data = np.concatenate([d[:, 0] for d in datas], axis=0)
    vmin_pred, vmax_pred = all_pred_data.min(), all_pred_data.max()

    # Calculate all errors for error color range
    errors_all = []
    for t in time_indices:
        raw_img = raw_data_uv[t, 0]
        for pred in datas[1:]:  # Skip groundtruth
            err = np.abs(pred[t, 0] - raw_img)
            errors_all.append(err)
    errors_all = np.stack(errors_all)
    vmin_err, vmax_err = errors_all.min(), errors_all.max()

    # Create integrated plot: 13 rows (1 groundtruth + 6 prediction + 6 error), 7 columns (time steps)
    fig, axes = plt.subplots(nrows=13, ncols=len(time_indices), figsize=(26, 45))

    for col, t in enumerate(time_indices):
        row_idx = 0
        
        # First row: Groundtruth
        img = raw_data_uv[t, 0]
        ax = axes[row_idx, col]
        im = ax.imshow(img, cmap='viridis', vmin=vmin_pred, vmax=vmax_pred)
        ax.axis('off')
        if col == 0:
            ax.text(-0.1, 0.5, 'Groundtruth', fontsize=18, fontweight='bold', 
                   transform=ax.transAxes, rotation=90, va='center', ha='right')
        ax.set_title(f"t={t+700}", fontsize=20, fontweight='bold')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        row_idx += 1

        # Each method followed by its error
        for i in range(1, 7):  # 6 prediction methods
            # Prediction row
            pred_img = datas[i][t, 0]
            ax_pred = axes[row_idx, col]
            im_pred = ax_pred.imshow(pred_img, cmap='viridis', vmin=vmin_pred, vmax=vmax_pred)
            ax_pred.axis('off')
            if col == 0:
                ax_pred.text(-0.1, 0.5, f'{titles[i]}', fontsize=18, fontweight='bold', 
                           transform=ax_pred.transAxes, rotation=90, va='center', ha='right')
            fig.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)
            row_idx += 1

            # Error row (immediately after prediction)
            err_img = np.abs(pred_img - raw_data_uv[t, 0])
            ax_err = axes[row_idx, col]
            im_err = ax_err.imshow(err_img, cmap='magma', vmin=vmin_err, vmax=vmax_err)
            ax_err.axis('off')
            if col == 0:
                ax_err.text(-0.1, 0.5, f'{titles[i]} Error', fontsize=18, fontweight='bold', 
                          transform=ax_err.transAxes, rotation=90, va='center', ha='right')
            fig.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.04)
            row_idx += 1

    fig.suptitle("CFD Bench Cylinder Flow: Predictions and Errors", fontsize=30, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    fig.savefig(os.path.join(fig_save_path, "cyl_comparison_integrated.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"[INFO] Integrated comparison plot saved to: {os.path.join(fig_save_path, 'cyl_comparison_integrated.png')}")