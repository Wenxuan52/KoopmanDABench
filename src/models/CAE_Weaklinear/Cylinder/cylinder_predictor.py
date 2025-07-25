import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

import os
import sys
current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
sys.path.append(src_directory)

from src.utils.Dataset import CylinderDynamicsDataset


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
            im = ax.imshow(img, cmap='viridis', vmin=vmin_pred, vmax=vmax_pred)
            ax.axis('off')
            ax.set_title(f"{titles[row]} t={t}", fontsize=18)
            fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig1.suptitle("Prediction Comparison", fontsize=24)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig1.savefig(os.path.join(save_dir, "cyl_comparison.png"))
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
        im = ax.imshow(img_raw, cmap='viridis', vmin=vmin_pred, vmax=vmax_pred)
        ax.axis('off')
        ax.set_title(f"Groundtruth t={t}", fontsize=18)
        fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if col == 0:
            ax.set_ylabel('Groundtruth')

        for i, pred in enumerate([recon, one, roll], start=1):
            err = np.abs(pred[t, 0] - img_raw)
            ax = axes2[i, col]
            im_err = ax.imshow(err, cmap='magma', vmin=vmin_err, vmax=vmax_err) #
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
    from cylinder_model import CYLINDER_C_FORWARD

    fig_save_path = '../../../results/CAE_Weaklinear/figures/'
    
    start_T = 700
    
    prediction_step = 300
    
    foward_step = 12

    val_idx = 3

    cyl_train_dataset = CylinderDynamicsDataset(data_path="../../../data/cylinder/cylinder_train_data.npy",
                seq_length = foward_step,
                mean=None,
                std=None)
    
    cyl_val_dataset = CylinderDynamicsDataset(data_path="../../../data/cylinder/cylinder_val_data.npy",
                seq_length = foward_step,
                mean=cyl_train_dataset.mean,
                std=cyl_train_dataset.std)

    denorm = cyl_val_dataset.denormalizer()

    groundtruth = cyl_val_dataset.data[val_idx, start_T:start_T + prediction_step, ...]
    groundtruth = torch.tensor(groundtruth, dtype=torch.float32)
    print(groundtruth.shape)
    print(groundtruth.min())
    print(groundtruth.max())
    
    raw_data_uv = (groundtruth[:, 0, :, :] ** 2 + groundtruth[:, 1, :, :] ** 2) ** 0.5
    raw_data_uv = raw_data_uv.unsqueeze(1)
    print(raw_data_uv.shape)
    
    forward_model = CYLINDER_C_FORWARD()
    forward_model.load_state_dict(torch.load('../../../results/CAE_Weaklinear/Cylinder/jointly_model_weights/forward_model.pt', weights_only=True, map_location='cpu'))
    forward_model.eval()

    print(forward_model)


    normalize_groundtruth = cyl_val_dataset.normalize(groundtruth)

    print(normalize_groundtruth.shape)
    print(normalize_groundtruth.min())
    print(normalize_groundtruth.max())

    state = normalize_groundtruth
    
    with torch.no_grad():
        z = forward_model.K_S(state)
        reconstruct = forward_model.K_S_preimage(z)

    de_reconstruct = denorm(reconstruct)
    de_reconstruct_uv = (de_reconstruct[:, 0, :, :] ** 2 + de_reconstruct[:, 1, :, :] ** 2) ** 0.5
    de_reconstruct_uv = de_reconstruct_uv.unsqueeze(1)
    print(de_reconstruct_uv.shape)
    print(de_reconstruct_uv.min())
    print(de_reconstruct_uv.max())

    with torch.no_grad():
        z_current = forward_model.K_S(state)
        z_next = forward_model.latent_forward(z_current)
        onestep = forward_model.K_S_preimage(z_next)

    onestep = torch.cat([normalize_groundtruth[0:1, ...], onestep[:-1, ...]])
    de_onestep = denorm(onestep)
    de_onestep_uv = (de_onestep[:, 0, :, :] ** 2 + de_onestep[:, 1, :, :] ** 2) ** 0.5
    de_onestep_uv = de_onestep_uv.unsqueeze(1)
    print(de_onestep_uv.shape)
    print(de_onestep_uv.min())
    print(de_onestep_uv.max())

    predictions = []
    current_state = state[0, ...].unsqueeze(0)
    print(current_state.shape)
    n_steps = prediction_step
    
    with torch.no_grad():
        for step in range(n_steps):
            z_current = forward_model.K_S(current_state)
            z_next = forward_model.latent_forward(z_current)
            next_state = forward_model.K_S_preimage(z_next)
            
            predictions.append(next_state)
            current_state = next_state
            
    rollout = torch.cat(predictions, dim=0)
    rollout = torch.cat([normalize_groundtruth[0:1, ...], rollout[:-1, ...]])

    print(rollout.shape)

    de_rollout = denorm(rollout)
    de_rollout_uv = (de_rollout[:, 0, :, :] ** 2 + de_rollout[:, 1, :, :] ** 2) ** 0.5
    de_rollout_uv = de_rollout_uv.unsqueeze(1)
    print(de_rollout_uv.shape)
    print(de_rollout_uv.min())
    print(de_rollout_uv.max())

    # plot_comparisons(raw_data_uv, de_reconstruct_uv, de_onestep_uv, de_rollout_uv,
    #                 time_indices=[1, 200, 500, 800, 999], save_dir=fig_save_path)

    plot_comparisons(raw_data_uv, de_reconstruct_uv, de_onestep_uv, de_rollout_uv,
                    time_indices=[1, 50, 100, 200, 299], save_dir=fig_save_path)

    # Compute Metric

    overall_metrics = compute_metrics(raw_data_uv, de_reconstruct_uv, de_onestep_uv, de_rollout_uv)
    
    print("Overall metric:")
    for method, metrics in overall_metrics.items():
        print(f"{method}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.6f}")
        print()
    
    temporal_metrics = compute_temporal_metrics(raw_data_uv, de_reconstruct_uv, de_onestep_uv, de_rollout_uv)
    
    print("Per Time Frame Metric (first 5 frames):")
    for method, metrics in temporal_metrics.items():
        print(f"{method}:")
        for metric_name, values in metrics.items():
            print(f"  {metric_name}: {values[:5]}")
        print()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics_names = ['MSE', 'MAE', 'Relative_L2', 'SSIM']
    
    for i, metric_name in enumerate(metrics_names):
        ax = axes[i//2, i%2]
        
        for method in ['reconstruction', 'onestep', 'rollout']:
            values = temporal_metrics[method][metric_name]
            ax.plot(values, label=method)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} over Time')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_path, "cyl_perframe_metric.png"))