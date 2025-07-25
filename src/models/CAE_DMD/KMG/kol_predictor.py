import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import yaml
from typing import Optional
import matplotlib.pyplot as plt

import os
import sys
current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", ".."))
sys.path.append(src_directory)

from src.utils.Dataset import KolDynamicsDataset


def plot_comparisons(raw_data, reconstruct, onestep, rollout, time_indices=[1, 4, 7, 10], save_dir="figures"):
    os.makedirs(save_dir, exist_ok=True)

    raw = raw_data.numpy()
    recon = reconstruct.detach().numpy()
    one = onestep.detach().numpy()
    roll = rollout.detach().numpy()

    titles = ['Grundtruth', 'Reconstruct', 'One-step', 'Rollout']
    datas = [raw, recon, one, roll]

    fig1, axes1 = plt.subplots(nrows=4, ncols=len(time_indices), figsize=(20, 14))
    all_pred_data = np.concatenate([d[:, 0] for d in datas], axis=0) 
    vmin_pred, vmax_pred = all_pred_data.min(), all_pred_data.max()

    for row in range(4):
        for col, t in enumerate(time_indices):
            img = datas[row][t, 0]
            ax = axes1[row, col]
            im = ax.imshow(img, cmap='RdBu_r', vmin=vmin_pred, vmax=vmax_pred)
            ax.axis('off')
            ax.set_title(f"{titles[row]} t={t}", fontsize=18)
            fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig1.suptitle("Prediction Comparison", fontsize=24)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig1.savefig(os.path.join(save_dir, "kol_comparison.png"))
    plt.close(fig1)

    fig2, axes2 = plt.subplots(nrows=4, ncols=len(time_indices), figsize=(20, 14))
    errors = []

    for t in time_indices:
        raw_img = raw[t, 0]
        for pred in [recon, one, roll]:
            err = np.abs(pred[t, 0] - raw_img)
            errors.append(err)
    errors_all = np.stack(errors)
    vmin_err, vmax_err = errors_all.min(), errors_all.max()

    for col, t in enumerate(time_indices):
        img_raw = raw[t, 0]
        ax = axes2[0, col]
        im = ax.imshow(img_raw, cmap='RdBu_r', vmin=vmin_pred, vmax=vmax_pred)
        ax.axis('off')
        ax.set_title(f"Groundtruth t={t}", fontsize=18)
        fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if col == 0:
            ax.set_ylabel('Groundtruth')

        for i, pred in enumerate([recon, one, roll], start=1):
            err = np.abs(pred[t, 0] - img_raw)
            ax = axes2[i, col]
            im_err = ax.imshow(err, cmap='magma', vmin=vmin_err, vmax=vmax_err) # 
            ax.set_title(f"Error {titles[i]} t={t}", fontsize=18)
            fig2.colorbar(im_err, ax=ax, fraction=0.046, pad=0.04)
            if col == 0:
                ax.set_ylabel(f"Error {titles[i]}")

    fig2.suptitle("Error Comparison", fontsize=24)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig2.savefig(os.path.join(save_dir, "kol_error.png"))
    plt.close(fig2)


def compute_metrics(groundtruth, reconstruction, onestep, rollout):
    def calculate_metrics(pred, gt):
        pred_np = pred.detach().cpu().numpy()
        gt_np = gt.detach().cpu().numpy()
        
        # MSE
        mse = F.mse_loss(pred, gt).item()
        
        # MAE
        mae = F.l1_loss(pred, gt).item()
        
        # Relative L2
        l2_error = torch.norm(pred - gt)
        l2_gt = torch.norm(gt)
        relative_l2 = (l2_error / l2_gt).item()
        
        # SSIM
        ssim_values = []
        for t in range(pred_np.shape[0]):
            ssim_val = ssim(gt_np[t, 0], pred_np[t, 0], data_range=gt_np[t, 0].max() - gt_np[t, 0].min())
            ssim_values.append(ssim_val)
        avg_ssim = np.mean(ssim_values)
        
        return {
            'MSE': mse,
            'MAE': mae,
            'Relative_L2': relative_l2,
            'SSIM': avg_ssim
        }
    
    results = {}
    results['reconstruction'] = calculate_metrics(reconstruction, groundtruth)
    results['onestep'] = calculate_metrics(onestep, groundtruth)
    results['rollout'] = calculate_metrics(rollout, groundtruth)
    
    return results


def compute_temporal_metrics(groundtruth, reconstruction, onestep, rollout):
    def calculate_temporal_metrics(pred, gt):
        T = pred.shape[0]
        pred_np = pred.detach().cpu().numpy()
        gt_np = gt.detach().cpu().numpy()
        
        mse_list = []
        mae_list = []
        relative_l2_list = []
        ssim_list = []
        
        for t in range(T):
            pred_t = pred[t:t+1]  # [1, 1, 64, 64]
            gt_t = gt[t:t+1]      # [1, 1, 64, 64]
            
            # MSE
            mse = F.mse_loss(pred_t, gt_t).item()
            mse_list.append(mse)
            
            # MAE
            mae = F.l1_loss(pred_t, gt_t).item()
            mae_list.append(mae)
            
            # Relative L2
            l2_error = torch.norm(pred_t - gt_t)
            l2_gt = torch.norm(gt_t)
            relative_l2 = (l2_error / l2_gt).item()
            relative_l2_list.append(relative_l2)
            
            # SSIM
            ssim_val = ssim(gt_np[t, 0], pred_np[t, 0], data_range=gt_np[t, 0].max() - gt_np[t, 0].min())
            ssim_list.append(ssim_val)
        
        return {
            'MSE': mse_list,
            'MAE': mae_list,
            'Relative_L2': relative_l2_list,
            'SSIM': ssim_list
        }
    
    results = {}
    results['reconstruction'] = calculate_temporal_metrics(reconstruction, groundtruth)
    results['onestep'] = calculate_temporal_metrics(onestep, groundtruth)
    results['rollout'] = calculate_temporal_metrics(rollout, groundtruth)
    
    return results


if __name__ == '__main__':
    # from kol_model import KOL_C_FORWARD
    from kol_model_new import KOL_C_FORWARD

    start_T = 100
    
    prediction_step = 10
    
    foward_step = 10

    val_idx = 9

    kol_train_dataset = KolDynamicsDataset(data_path="../../../data/kolmogorov/RE40_T20/kolmogorov_train_data.npy",
                seq_length = foward_step,
                mean=None,
                std=None)
    
    kol_val_dataset = KolDynamicsDataset(data_path="../../../data/kolmogorov/RE40_T20/kolmogorov_val_data.npy",
                seq_length = foward_step,
                mean=kol_train_dataset.mean,
                std=kol_train_dataset.std)
    
    denorm = kol_val_dataset.denormalizer()

    groundtruth = kol_val_dataset.data[val_idx, start_T:start_T + prediction_step, ...]
    groundtruth = torch.tensor(groundtruth, dtype=torch.float32)
    print(groundtruth.shape)
    print(groundtruth.min())
    print(groundtruth.max())

    forward_model = KOL_C_FORWARD()
    forward_model.load_state_dict(torch.load('kol_model_weights_new/forward_model.pt', weights_only=True, map_location='cpu'))
    forward_model.C_forward = torch.load('kol_model_weights_new/C_forward.pt', weights_only=True, map_location='cpu')
    forward_model.eval()

    U, S, Vh = torch.linalg.svd(forward_model.C_forward)
    print(S.max())

    print(forward_model)

    normalize_groundtruth = kol_val_dataset.normalize(groundtruth)

    print(normalize_groundtruth.shape)
    print(normalize_groundtruth.min())
    print(normalize_groundtruth.max())

    state = normalize_groundtruth
    
    with torch.no_grad():
        z = forward_model.K_S(state)
        reconstruct = forward_model.K_S_preimage(z)

    de_reconstruct = denorm(reconstruct)
    print(de_reconstruct.shape)
    print(de_reconstruct.min())
    print(de_reconstruct.max())

    with torch.no_grad():
        z_current = forward_model.K_S(state)
        z_next = torch.mm(z_current, forward_model.C_forward)
        onestep = forward_model.K_S_preimage(z_next)

    onestep = torch.cat([normalize_groundtruth[0:1, ...], onestep[:-1, ...]])
    de_onestep = denorm(onestep)
    print(de_onestep.shape)
    print(de_onestep.min())
    print(de_onestep.max())

    predictions = []
    current_state = state[0, ...].unsqueeze(0)
    n_steps = foward_step
    
    with torch.no_grad():
        for step in range(n_steps):
            z_current = forward_model.K_S(current_state)
            z_next = forward_model.latent_forward(z_current)
            next_state = forward_model.K_S_preimage(z_next)
            
            predictions.append(next_state)
            current_state = next_state
            
    rollout = torch.cat(predictions, dim=0)
    rollout = torch.cat([normalize_groundtruth[0:1, ...], rollout[:-1, ...]])
    de_rollout = denorm(rollout)
    print(de_rollout.shape)
    print(de_rollout.min())
    print(de_rollout.max())

    plot_comparisons(groundtruth, de_reconstruct, de_onestep, de_rollout, time_indices=[1, 4, 7, 9])
