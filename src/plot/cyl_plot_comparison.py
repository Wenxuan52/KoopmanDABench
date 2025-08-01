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

def plot_comparisons(raw_data, dmd_data_uv, cae_dmd_data_uv, cae_linear_data_uv, cae_weaklinear_data_uv, time_indices=[1, 4, 7, 10], save_dir="figures"):
    os.makedirs(save_dir, exist_ok=True)

    # Convert all inputs to numpy arrays
    raw = raw_data.numpy()
    dmd = dmd_data_uv
    cae_dmd = cae_dmd_data_uv
    cae_linear = cae_linear_data_uv
    cae_weaklinear = cae_weaklinear_data_uv

    titles = ['Grundtruth', 'DMD', 'CAE_DMD', 'CAE_Linear', 'CAE_Weaklinear']
    datas = [raw, dmd, cae_dmd, cae_linear, cae_weaklinear]

    # First figure: Predictions comparison
    fig1, axes1 = plt.subplots(nrows=5, ncols=len(time_indices), figsize=(20, 18))
    all_pred_data = np.concatenate([d[:, 0] for d in datas], axis=0) 
    vmin_pred, vmax_pred = all_pred_data.min(), all_pred_data.max()

    for row in range(5):
        for col, t in enumerate(time_indices):
            img = datas[row][t, 0]
            ax = axes1[row, col]
            im = ax.imshow(img, cmap='viridis', vmin=vmin_pred, vmax=vmax_pred)
            ax.axis('off')
            ax.set_title(f"{titles[row]} t={t}", fontsize=18)
            fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig1.suptitle("Prediction Comparison", fontsize=24)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig1.savefig(os.path.join(save_dir, "cyl_comparison.png"))
    plt.close(fig1)

    # Second figure: Error comparison
    fig2, axes2 = plt.subplots(nrows=5, ncols=len(time_indices), figsize=(20, 18))
    errors = []

    # Calculate errors for all predictions
    for t in time_indices:
        raw_img = raw[t, 0]
        for pred in [dmd, cae_dmd, cae_linear, cae_weaklinear]:
            err = np.abs(pred[t, 0] - raw_img)
            errors.append(err)
    errors_all = np.stack(errors)
    vmin_err, vmax_err = errors_all.min(), errors_all.max()

    for col, t in enumerate(time_indices):
        # First row: Groundtruth
        img_raw = raw[t, 0]
        ax = axes2[0, col]
        im = ax.imshow(img_raw, cmap='viridis', vmin=vmin_pred, vmax=vmax_pred)
        ax.axis('off')
        ax.set_title(f"Groundtruth t={t}", fontsize=18)
        fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if col == 0:
            ax.set_ylabel('Groundtruth')

        # Remaining rows: Error plots
        for i, pred in enumerate([dmd, cae_dmd, cae_linear, cae_weaklinear], start=1):
            err = np.abs(pred[t, 0] - img_raw)
            ax = axes2[i, col]
            im_err = ax.imshow(err, cmap='magma', vmin=vmin_err, vmax=vmax_err)
            ax.axis('off')
            ax.set_title(f"Error {titles[i]} t={t}", fontsize=18)
            fig2.colorbar(im_err, ax=ax, fraction=0.046, pad=0.04)
            if col == 0:
                ax.set_ylabel(f"Error {titles[i]}")

    fig2.suptitle("Error Comparison", fontsize=24)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig2.savefig(os.path.join(save_dir, "cyl_error.png"))
    plt.close(fig2)

def prepare_dmd_data(data, device):
    """Convert (T, C, H, W) to flattened format for DMD"""
    T, C, H, W = data.shape
    data_flat = data.reshape(T, C * H * W).T.to(device)  # Shape: [D, T]
    return data_flat

def reconstruct_from_dmd(data_flat, original_shape):
    """Convert flattened DMD output back to (T, C, H, W)"""
    D, T = data_flat.shape
    T_orig, C, H, W = original_shape
    return data_flat.T.reshape(T, C, H, W)  # Shape: [T, C, H, W]

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

    dmd_data_uv = np.load('../../results/DMD/figures/cyl_rollout.npy')

    cae_dmd_data_uv = np.load('../../results/CAE_DMD/figures/cyl_rollout.npy')

    cae_linear_data_uv = np.load('../../results/CAE_Linear/figures/cyl_rollout.npy')

    cae_weaklinear_data_uv = np.load('../../results/CAE_Weaklinear/figures/cyl_rollout.npy')

    # print(dmd_data_uv.shape)
    # print(cae_dmd_data_uv.shape)
    # print(cae_linear_data_uv.shape)
    # print(cae_weaklinear_data_uv.shape)

    plot_comparisons(raw_data_uv, dmd_data_uv, cae_dmd_data_uv, cae_linear_data_uv, cae_weaklinear_data_uv, 
                    time_indices=[1, 50, 100, 200, 299], save_dir=fig_save_path)

