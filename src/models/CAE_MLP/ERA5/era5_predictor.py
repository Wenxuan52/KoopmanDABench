import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import h5py

import psutil
import time
import pickle

import os
import sys
current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
sys.path.append(src_directory)

from src.utils.Dataset import ERA5Dataset

def get_memory_usage():
    """Get current memory usage for CPU and GPU"""
    cpu_memory = psutil.virtual_memory().used / 1024**3  # GB
    gpu_memory = 0
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
    return cpu_memory, gpu_memory

def save_inference_stats(stats_dict, save_path):
    """Save inference monitoring statistics"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(stats_dict, f)
    print(f"[INFO] Inference statistics saved to: {save_path}")


def plot_era5_comparisons(raw_data, rollout, time_indices=[1, 4, 7, 10], save_dir="figures"):
    """
    Plot ERA5 predictions comparison
    raw_data, rollout: shape (time_steps, 5, 64, 32) - 5 channels
    """
    os.makedirs(save_dir, exist_ok=True)

    raw = raw_data.numpy()
    roll = rollout.detach().numpy()
    
    channel_names = ['Geopotential', 'Temperature', 'Humidity', 'Wind_u', 'Wind_v']
    
    # First figure: Prediction comparison (5 channels x 2 rows each = 10 rows, but displayed horizontally)
    # Each channel has 2 rows: Ground truth and Rollout prediction
    fig1, axes1 = plt.subplots(nrows=10, ncols=len(time_indices), figsize=(6*len(time_indices), 20))
    
    for ch_idx in range(5):  # 5 channels
        # Calculate vmin, vmax for this channel across all time steps
        channel_data = np.concatenate([raw[:, ch_idx], roll[:, ch_idx]], axis=0)
        vmin_ch, vmax_ch = channel_data.min(), channel_data.max()
        
        for col, t in enumerate(time_indices):
            # Ground truth row (even rows: 0, 2, 4, 6, 8)
            gt_row = ch_idx * 2
            img_gt = raw[t, ch_idx].T  # Transpose for horizontal display
            ax_gt = axes1[gt_row, col]
            im_gt = ax_gt.imshow(img_gt, cmap='viridis', vmin=vmin_ch, vmax=vmax_ch, origin='lower', aspect='auto')
            ax_gt.axis('off')
            ax_gt.set_title(f"GT {channel_names[ch_idx]} t={t}", fontsize=11)
            fig1.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)
            
            # Rollout prediction row (odd rows: 1, 3, 5, 7, 9)
            pred_row = ch_idx * 2 + 1
            img_pred = roll[t, ch_idx].T  # Transpose for horizontal display
            ax_pred = axes1[pred_row, col]
            im_pred = ax_pred.imshow(img_pred, cmap='viridis', vmin=vmin_ch, vmax=vmax_ch, origin='lower', aspect='auto')
            ax_pred.axis('off')
            ax_pred.set_title(f"Pred {channel_names[ch_idx]} t={t}", fontsize=11)
            fig1.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)
            
            # Add channel labels on the left
            if col == 0:
                ax_gt.set_ylabel(f'GT\n{channel_names[ch_idx]}', rotation=0, labelpad=50, ha='right', va='center', fontsize=10)
                ax_pred.set_ylabel(f'Pred\n{channel_names[ch_idx]}', rotation=0, labelpad=50, ha='right', va='center', fontsize=10)

    fig1.suptitle("ERA5 Rollout Prediction Comparison", fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.96, left=0.08)
    fig1.savefig(os.path.join(save_dir, "era5_rollout_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # Second figure: Error comparison (10 rows x time_steps columns)
    fig2, axes2 = plt.subplots(nrows=10, ncols=len(time_indices), figsize=(6*len(time_indices), 20))

    for ch_idx in range(5):  # 5 channels
        # Calculate vmin, vmax for this channel for ground truth
        channel_data = np.concatenate([raw[:, ch_idx], roll[:, ch_idx]], axis=0)
        vmin_ch, vmax_ch = channel_data.min(), channel_data.max()
        
        # Calculate error range for this channel independently
        channel_errors = []
        for t in time_indices:
            err = np.abs(roll[t, ch_idx] - raw[t, ch_idx])
            channel_errors.append(err)
        channel_errors = np.stack(channel_errors)
        vmin_err_ch, vmax_err_ch = channel_errors.min(), channel_errors.max()
        
        for col, t in enumerate(time_indices):
            # Ground truth row (even rows: 0, 2, 4, 6, 8)
            gt_row = ch_idx * 2
            img_gt = raw[t, ch_idx].T  # Transpose for horizontal display
            
            ax_gt = axes2[gt_row, col]
            im_gt = ax_gt.imshow(img_gt, cmap='viridis', vmin=vmin_ch, vmax=vmax_ch, origin='lower', aspect='auto')
            ax_gt.axis('off')
            ax_gt.set_title(f"GT {channel_names[ch_idx]} t={t}", fontsize=11)
            fig2.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)
            
            # Error row (odd rows: 1, 3, 5, 7, 9) - each channel has independent colorbar
            err_row = ch_idx * 2 + 1
            error = np.abs(roll[t, ch_idx] - raw[t, ch_idx]).T  # Transpose for horizontal display
            ax_err = axes2[err_row, col]
            im_err = ax_err.imshow(error, cmap='magma', vmin=vmin_err_ch, vmax=vmax_err_ch, origin='lower', aspect='auto')
            ax_err.axis('off')
            ax_err.set_title(f"Error {channel_names[ch_idx]} t={t}", fontsize=11)
            fig2.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.04)
            
            # Add channel labels on the left
            if col == 0:
                ax_gt.set_ylabel(f'GT\n{channel_names[ch_idx]}', rotation=0, labelpad=50, ha='right', va='center', fontsize=10)
                ax_err.set_ylabel(f'Error\n{channel_names[ch_idx]}', rotation=0, labelpad=50, ha='right', va='center', fontsize=10)

    fig2.suptitle("ERA5 Rollout Error Comparison", fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.96, left=0.08)
    fig2.savefig(os.path.join(save_dir, "era5_rollout_error.png"), dpi=150, bbox_inches='tight')
    plt.close(fig2)


def compute_era5_metrics(groundtruth, rollout):
    """Compute metrics for ERA5 data - both overall and per-channel"""
    channel_names = ['Geopotential', 'Temperature', 'Humidity', 'Wind_u', 'Wind_v']
    
    def calculate_metrics(pred, gt):
        pred_np = pred.detach().cpu().numpy()
        gt_np = gt.detach().cpu().numpy()
        
        # Overall metrics
        mse = F.mse_loss(pred, gt).item()
        mae = F.l1_loss(pred, gt).item()
        
        l2_error = torch.norm(pred - gt)
        l2_gt = torch.norm(gt)
        relative_l2 = (l2_error / l2_gt).item()

        rmse = torch.sqrt(F.mse_loss(pred, gt))
        gt_mean = torch.mean(gt)
        rrmse = (rmse / gt_mean).item()
        
        # Overall SSIM - compute for each channel separately then average
        ssim_values = []
        for t in range(pred_np.shape[0]):
            for ch in range(pred_np.shape[1]):  # 5 channels
                if gt_np[t, ch].max() - gt_np[t, ch].min() > 0:  # avoid division by zero
                    ssim_val = ssim(gt_np[t, ch], pred_np[t, ch], 
                                  data_range=gt_np[t, ch].max() - gt_np[t, ch].min())
                    ssim_values.append(ssim_val)
        avg_ssim = np.mean(ssim_values) if ssim_values else 0.0
        
        overall_metrics = {
            'MSE': mse,
            'MAE': mae,
            'Relative_L2': relative_l2,
            'RRMSE': rrmse,
            'SSIM': avg_ssim
        }
        
        # Per-channel metrics
        per_channel_metrics = {}
        for ch_idx, ch_name in enumerate(channel_names):
            # Extract channel data: shape [T, H, W]
            pred_ch = pred[:, ch_idx, :, :]  # [T, H, W]
            gt_ch = gt[:, ch_idx, :, :]      # [T, H, W]
            
            # Channel-specific metrics
            mse_ch = F.mse_loss(pred_ch, gt_ch).item()
            mae_ch = F.l1_loss(pred_ch, gt_ch).item()
            
            l2_error_ch = torch.norm(pred_ch - gt_ch)
            l2_gt_ch = torch.norm(gt_ch)
            relative_l2_ch = (l2_error_ch / l2_gt_ch).item()
            
            rmse_ch = torch.sqrt(F.mse_loss(pred_ch, gt_ch))
            gt_mean_ch = torch.mean(gt_ch)
            rrmse_ch = (rmse_ch / gt_mean_ch).item()
            
            # Channel-specific SSIM
            ssim_values_ch = []
            for t in range(pred_np.shape[0]):
                if gt_np[t, ch_idx].max() - gt_np[t, ch_idx].min() > 0:
                    ssim_val = ssim(gt_np[t, ch_idx], pred_np[t, ch_idx], 
                                  data_range=gt_np[t, ch_idx].max() - gt_np[t, ch_idx].min())
                    ssim_values_ch.append(ssim_val)
            avg_ssim_ch = np.mean(ssim_values_ch) if ssim_values_ch else 0.0
            
            per_channel_metrics[ch_name] = {
                'MSE': mse_ch,
                'MAE': mae_ch,
                'Relative_L2': relative_l2_ch,
                'RRMSE': rrmse_ch,
                'SSIM': avg_ssim_ch
            }
        
        return overall_metrics, per_channel_metrics
    
    overall_metrics, per_channel_metrics = calculate_metrics(rollout, groundtruth)
    
    results = {
        'rollout_overall': overall_metrics,
        'rollout_per_channel': per_channel_metrics
    }
    
    return results


def compute_era5_temporal_metrics(groundtruth, rollout):
    """Compute temporal metrics for ERA5 data - both overall and per-channel"""
    channel_names = ['Geopotential', 'Temperature', 'Humidity', 'Wind_u', 'Wind_v']
    
    def calculate_temporal_metrics(pred, gt):
        T = pred.shape[0]
        pred_np = pred.detach().cpu().numpy()
        gt_np = gt.detach().cpu().numpy()
        
        # Overall temporal metrics
        mse_list = []
        mae_list = []
        relative_l2_list = []
        rrmse_list = []
        ssim_list = []
        
        # Per-channel temporal metrics
        per_channel_temporal = {ch_name: {
            'MSE': [], 'MAE': [], 'Relative_L2': [], 'RRMSE': [], 'SSIM': []
        } for ch_name in channel_names}
        
        for t in range(T):
            pred_t = pred[t:t+1]  # [1, 5, 64, 32]
            gt_t = gt[t:t+1]      # [1, 5, 64, 32]
            
            # Overall metrics for this timestep
            mse = F.mse_loss(pred_t, gt_t).item()
            mse_list.append(mse)
            
            mae = F.l1_loss(pred_t, gt_t).item()
            mae_list.append(mae)
            
            l2_error = torch.norm(pred_t - gt_t)
            l2_gt = torch.norm(gt_t)
            relative_l2 = (l2_error / l2_gt).item()
            relative_l2_list.append(relative_l2)
            
            rmse = torch.sqrt(F.mse_loss(pred_t, gt_t))
            gt_mean = torch.mean(gt_t)
            rrmse = (rmse / gt_mean).item()
            rrmse_list.append(rrmse)
            
            # Overall SSIM for this timestep
            ssim_vals_t = []
            for ch in range(pred_np.shape[1]):  # 5 channels
                if gt_np[t, ch].max() - gt_np[t, ch].min() > 0:
                    ssim_val = ssim(gt_np[t, ch], pred_np[t, ch], 
                                  data_range=gt_np[t, ch].max() - gt_np[t, ch].min())
                    ssim_vals_t.append(ssim_val)
            avg_ssim_t = np.mean(ssim_vals_t) if ssim_vals_t else 0.0
            ssim_list.append(avg_ssim_t)
            
            # Per-channel metrics for this timestep
            for ch_idx, ch_name in enumerate(channel_names):
                pred_ch_t = pred_t[:, ch_idx:ch_idx+1, :, :]  # [1, 1, 64, 32]
                gt_ch_t = gt_t[:, ch_idx:ch_idx+1, :, :]      # [1, 1, 64, 32]
                
                mse_ch = F.mse_loss(pred_ch_t, gt_ch_t).item()
                per_channel_temporal[ch_name]['MSE'].append(mse_ch)
                
                mae_ch = F.l1_loss(pred_ch_t, gt_ch_t).item()
                per_channel_temporal[ch_name]['MAE'].append(mae_ch)
                
                l2_error_ch = torch.norm(pred_ch_t - gt_ch_t)
                l2_gt_ch = torch.norm(gt_ch_t)
                relative_l2_ch = (l2_error_ch / l2_gt_ch).item()
                per_channel_temporal[ch_name]['Relative_L2'].append(relative_l2_ch)
                
                rmse_ch = torch.sqrt(F.mse_loss(pred_ch_t, gt_ch_t))
                gt_mean_ch = torch.mean(gt_ch_t)
                rrmse_ch = (rmse_ch / gt_mean_ch).item()
                per_channel_temporal[ch_name]['RRMSE'].append(rrmse_ch)
                
                # Channel-specific SSIM for this timestep
                if gt_np[t, ch_idx].max() - gt_np[t, ch_idx].min() > 0:
                    ssim_ch = ssim(gt_np[t, ch_idx], pred_np[t, ch_idx], 
                                 data_range=gt_np[t, ch_idx].max() - gt_np[t, ch_idx].min())
                    per_channel_temporal[ch_name]['SSIM'].append(ssim_ch)
                else:
                    per_channel_temporal[ch_name]['SSIM'].append(0.0)
        
        overall_temporal = {
            'MSE': mse_list,
            'MAE': mae_list,
            'Relative_L2': relative_l2_list,
            'RRMSE': rrmse_list,
            'SSIM': ssim_list
        }
        
        return overall_temporal, per_channel_temporal
    
    overall_temporal, per_channel_temporal = calculate_temporal_metrics(rollout, groundtruth)
    
    results = {
        'rollout_overall': overall_temporal,
        'rollout_per_channel': per_channel_temporal
    }
    
    return results


if __name__ == '__main__':
    from era5_model_FTF import ERA5_C_FORWARD

    fig_save_path = '../../../../results/CAE_MLP/figures/'
    
    start_T = 1000
    prediction_step = 100
    forward_step = 12

    # Load ERA5 datasets
    print("Loading ERA5 datasets...")
    
    era5_test_dataset = ERA5Dataset(
        data_path="../../../../data/ERA5/ERA5_data/test_seq_state.h5",
        seq_length=forward_step,
        min_path="../../../../data/ERA5/ERA5_data/min_val.npy",
        max_path="../../../../data/ERA5/ERA5_data/max_val.npy"
    )

    # Get denormalizer function
    denorm = era5_test_dataset.denormalizer()
    
    # Get ground truth data
    raw_test_data = era5_test_dataset.data  # shape: [N, H, W, C]
    groundtruth = raw_test_data[start_T:start_T + prediction_step, ...]  # (30, 64, 32, 5)
    groundtruth = torch.tensor(groundtruth, dtype=torch.float32).permute(0, 3, 1, 2)  # (30, 5, 64, 32)
    
    print(f"Ground truth shape: {groundtruth.shape}")
    print(f"Ground truth range: [{groundtruth.min():.4f}, {groundtruth.max():.4f}]")
    
    # Normalize ground truth using dataset's normalize function
    normalize_groundtruth = era5_test_dataset.normalize(groundtruth)
    print(f"Normalized ground truth range: [{normalize_groundtruth.min():.4f}, {normalize_groundtruth.max():.4f}]")
    
    # Load model
    forward_model = ERA5_C_FORWARD()
    forward_model.load_state_dict(torch.load('../../../../results/CAE_MLP/ERA5/model_weights_FTF/forward_model.pt', 
                                            weights_only=True, map_location='cpu'))
    forward_model.eval()
    
    print(forward_model)

    inference_stats = {}

    # Warm up
    with torch.no_grad():
        z = forward_model.K_S(normalize_groundtruth[:1])
        _ = forward_model.K_S_preimage(z)
    
    print("\n=== ERA5 Rollout Inference ===")
    predictions = []
    current_state = normalize_groundtruth[0, ...].unsqueeze(0)  # (1, 5, 64, 32)
    n_steps = prediction_step

    start_time = time.time()
    start_cpu, start_gpu = get_memory_usage()
    max_cpu_rollout = start_cpu
    max_gpu_rollout = start_gpu

    step_times = []

    # with torch.no_grad():
    #     for step in range(n_steps):
    #         step_start = time.time()
            
    #         gt_state = normalize_groundtruth[step:step+1]
    #         z_current = forward_model.K_S(gt_state)
    #         reconstructed_state = forward_model.K_S_preimage(z_current)
            
    #         predictions.append(reconstructed_state)
    
    # with torch.no_grad():
    #     for step in range(n_steps):
    #         step_start = time.time()
            
    #         gt_current = normalize_groundtruth[step:step+1]
    #         z_current = forward_model.K_S(gt_current)
    #         z_next = forward_model.latent_forward(z_current)
    #         next_state = forward_model.K_S_preimage(z_next)
            
    #         predictions.append(next_state)
    
    with torch.no_grad():
        for step in range(n_steps):
            step_start = time.time()

            z_current = forward_model.K_S(current_state)
            z_next = forward_model.latent_forward(z_current)
            next_state = forward_model.K_S_preimage(z_next)
            
            predictions.append(next_state)
            current_state = next_state

            step_time = time.time() - step_start
            step_times.append(step_time)
            
            cpu_mem, gpu_mem = get_memory_usage()
            max_cpu_rollout = max(max_cpu_rollout, cpu_mem)
            max_gpu_rollout = max(max_gpu_rollout, gpu_mem)
    
    rollout_time = time.time() - start_time
    avg_step_time = sum(step_times) / len(step_times)
    
    print(f"Rollout total time: {rollout_time:.4f}s")
    print(f"Average time per step: {avg_step_time:.4f}s")
    print(f"Memory usage - CPU: {max_cpu_rollout:.2f}GB, GPU: {max_gpu_rollout:.2f}GB")
    
    inference_stats['rollout'] = {
        'total_time': rollout_time,
        'avg_time_per_step': avg_step_time,
        'max_cpu_memory': max_cpu_rollout,
        'max_gpu_memory': max_gpu_rollout,
        'step_times': step_times
    }

    rollout = torch.cat(predictions, dim=0)  # (30, 5, 64, 32)
    rollout = torch.cat([normalize_groundtruth[0:1, ...], rollout[:-1, ...]])  # Replace first timestep with ground truth

    print(f"Rollout shape: {rollout.shape}")

    # Denormalize for visualization and metrics
    de_rollout = denorm(rollout)
    print(f"Denormalized rollout range: [{de_rollout.min():.4f}, {de_rollout.max():.4f}]")

    # Save results
    np.save(os.path.join(fig_save_path, 'era5_rollout.npy'), de_rollout.numpy())

    inference_stats['config'] = {
        'prediction_step': prediction_step,
        'forward_step': forward_step,
        'start_T': start_T
    }
    
    save_inference_stats(inference_stats, os.path.join(fig_save_path, 'era5_inference_stats.pkl'))

    # Plot comparisons
    plot_era5_comparisons(groundtruth, de_rollout,
                         time_indices=[1, 20, 50, 80, 99], save_dir=fig_save_path)
    
    # Compute metrics - now includes both overall and per-channel metrics
    overall_metrics = compute_era5_metrics(groundtruth, de_rollout)
    
    print("\nOverall metrics:")
    for method, metrics in overall_metrics.items():
        if method == 'rollout_overall':
            print(f"Rollout (Overall):")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.6f}")
            print()
        elif method == 'rollout_per_channel':
            print("Rollout (Per Channel):")
            for channel, channel_metrics in metrics.items():
                print(f"  {channel}:")
                for metric_name, value in channel_metrics.items():
                    print(f"    {metric_name}: {value:.6f}")
            print()
    
    # Compute and plot temporal metrics - now includes both overall and per-channel
    temporal_metrics = compute_era5_temporal_metrics(groundtruth, de_rollout)
    
    # Plot overall temporal metrics
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    metrics_names = ['MSE', 'MAE', 'Relative_L2', 'RRMSE', 'SSIM']
    
    for i, metric_name in enumerate(metrics_names):
        ax = axes[i]
        values = temporal_metrics['rollout_overall'][metric_name]
        ax.plot(values, label='rollout', marker='o', markersize=3)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} over Time (Overall)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_path, "era5_perframe_metric_overall.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot per-channel temporal metrics
    channel_names = ['Geopotential', 'Temperature', 'Humidity', 'Wind_u', 'Wind_v']
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for metric_name in metrics_names:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        for ch_idx, (channel, color) in enumerate(zip(channel_names, colors)):
            values = temporal_metrics['rollout_per_channel'][channel][metric_name]
            ax.plot(values, label=channel, marker='o', markersize=2, color=color)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} over Time (Per Channel)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_save_path, f"era5_perframe_{metric_name.lower()}_per_channel.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"\nResults saved to: {fig_save_path}")
    print("Files generated:")
    print("- era5_rollout_comparison.png")
    print("- era5_rollout_error.png")
    print("- era5_perframe_metric_overall.png")
    print("- era5_perframe_mse_per_channel.png")
    print("- era5_perframe_mae_per_channel.png")
    print("- era5_perframe_relative_l2_per_channel.png")
    print("- era5_perframe_rrmse_per_channel.png")
    print("- era5_perframe_ssim_per_channel.png")
    print("- era5_rollout.npy")
    print("- era5_inference_stats.pkl")