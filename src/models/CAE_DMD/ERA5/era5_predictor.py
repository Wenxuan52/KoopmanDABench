import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import psutil
import time
import pickle
import os
import sys

current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
sys.path.append(src_directory)

from src.utils.Dataset import ERA5Dataset
from src.models.DMD.base import TorchDMD
from src.models.CAE_Koopman.ERA5.era5_model_FTF import ERA5_C_FORWARD

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

def plot_era5_comparisons(raw_data, onestep, rollout, time_indices=[1, 4, 7, 10], save_dir="figures"):
    """Plot ERA5 predictions comparison"""
    os.makedirs(save_dir, exist_ok=True)

    raw = raw_data.numpy()
    one = onestep.detach().numpy()
    roll = rollout.detach().numpy()
    
    channel_names = ['Geopotential', 'Temperature', 'Humidity', 'Wind_u', 'Wind_v']
    
    # Main comparison plot
    fig1, axes1 = plt.subplots(nrows=15, ncols=len(time_indices), figsize=(6*len(time_indices), 30))
    
    for ch_idx in range(5):  # 5 channels
        # Calculate vmin, vmax for this channel across all time steps
        channel_data = np.concatenate([raw[:, ch_idx], one[:, ch_idx], roll[:, ch_idx]], axis=0)
        vmin_ch, vmax_ch = channel_data.min(), channel_data.max()
        
        for col, t in enumerate(time_indices):
            # Ground truth row
            gt_row = ch_idx * 3
            img_gt = raw[t, ch_idx].T
            ax_gt = axes1[gt_row, col]
            im_gt = ax_gt.imshow(img_gt, cmap='viridis', vmin=vmin_ch, vmax=vmax_ch, origin='lower', aspect='auto')
            ax_gt.axis('off')
            ax_gt.set_title(f"GT {channel_names[ch_idx]} t={t}", fontsize=11)
            fig1.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)
            
            # One-step prediction row
            onestep_row = ch_idx * 3 + 1
            img_onestep = one[t, ch_idx].T
            ax_onestep = axes1[onestep_row, col]
            im_onestep = ax_onestep.imshow(img_onestep, cmap='viridis', vmin=vmin_ch, vmax=vmax_ch, origin='lower', aspect='auto')
            ax_onestep.axis('off')
            ax_onestep.set_title(f"One-step {channel_names[ch_idx]} t={t}", fontsize=11)
            fig1.colorbar(im_onestep, ax=ax_onestep, fraction=0.046, pad=0.04)
            
            # Rollout prediction row
            rollout_row = ch_idx * 3 + 2
            img_rollout = roll[t, ch_idx].T
            ax_rollout = axes1[rollout_row, col]
            im_rollout = ax_rollout.imshow(img_rollout, cmap='viridis', vmin=vmin_ch, vmax=vmax_ch, origin='lower', aspect='auto')
            ax_rollout.axis('off')
            ax_rollout.set_title(f"Rollout {channel_names[ch_idx]} t={t}", fontsize=11)
            fig1.colorbar(im_rollout, ax=ax_rollout, fraction=0.046, pad=0.04)
            
            # Add channel labels on the left
            if col == 0:
                ax_gt.set_ylabel(f'GT\n{channel_names[ch_idx]}', rotation=0, labelpad=50, ha='right', va='center', fontsize=10)
                ax_onestep.set_ylabel(f'One-step\n{channel_names[ch_idx]}', rotation=0, labelpad=50, ha='right', va='center', fontsize=10)
                ax_rollout.set_ylabel(f'Rollout\n{channel_names[ch_idx]}', rotation=0, labelpad=50, ha='right', va='center', fontsize=10)

    fig1.suptitle("CAE+DMD ERA5 Prediction Comparison", fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.97, left=0.08)
    fig1.savefig(os.path.join(save_dir, "era5_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close(fig1)

    print(f"[INFO] Comparison plots saved to {save_dir}")

def compute_era5_metrics(groundtruth, onestep, rollout):
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
                if gt_np[t, ch].max() - gt_np[t, ch].min() > 0:
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
            pred_ch = pred[:, ch_idx, :, :]
            gt_ch = gt[:, ch_idx, :, :]
            
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
    
    onestep_overall, onestep_per_channel = calculate_metrics(onestep, groundtruth)
    rollout_overall, rollout_per_channel = calculate_metrics(rollout, groundtruth)
    
    results = {
        'onestep_overall': onestep_overall,
        'onestep_per_channel': onestep_per_channel,
        'rollout_overall': rollout_overall,
        'rollout_per_channel': rollout_per_channel
    }
    
    return results

if __name__ == '__main__':
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    fig_save_path = '../../../../results/CAE_DMD/figures/'
    start_T = 1000
    prediction_step = 100
    
    # Load dataset
    print("[INFO] Loading ERA5 datasets...")
    era5_test_dataset = ERA5Dataset(
        data_path="../../../../data/ERA5/ERA5_data/test_seq_state.h5",
        seq_length=12,  # Not used for prediction but required by dataset
        min_path="../../../../data/ERA5/ERA5_data/min_val.npy",
        max_path="../../../../data/ERA5/ERA5_data/max_val.npy"
    )
    
    denorm = era5_test_dataset.denormalizer()
    
    # Load CAE+DMD models
    print("[INFO] Loading CAE+DMD models...")
    
    # Load CAE encoder/decoder
    cae_model = ERA5_C_FORWARD()
    cae_model.load_state_dict(
        torch.load('../../../../results/CAE_Koopman/ERA5/model_weights_FTF/forward_model.pt', 
                  weights_only=True, map_location=device)
    )
    cae_model.to(device)
    cae_model.eval()
    
    # Load DMD model
    dmd = TorchDMD(svd_rank=508, device=device)
    dmd.load_dmd('../../../../results/CAE_DMD/ERA5/dmd_model.pth')
    
    print(f"[INFO] CAE hidden dimension: {cae_model.hidden_dim}")
    print(f"[INFO] DMD eigenvalues shape: {dmd.eigenvalues.shape}")
    
    # Prepare groundtruth data
    raw_test_data = era5_test_dataset.data  # shape: [N, H, W, C]
    groundtruth = raw_test_data[start_T:start_T + prediction_step, ...]  # (100, 64, 32, 5)
    groundtruth = torch.tensor(groundtruth, dtype=torch.float32).permute(0, 3, 1, 2)  # (100, 5, 64, 32)
    
    print(f"[INFO] Groundtruth shape: {groundtruth.shape}")
    print(f"[INFO] Groundtruth range: [{groundtruth.min():.4f}, {groundtruth.max():.4f}]")
    
    # Normalize data
    normalize_groundtruth = era5_test_dataset.normalize(groundtruth).to(device)
    print(f"[INFO] Normalized groundtruth range: [{normalize_groundtruth.min():.4f}, {normalize_groundtruth.max():.4f}]")
    
    inference_stats = {}
    
    # Warm up
    with torch.no_grad():
        z_init = cae_model.K_S(normalize_groundtruth[0:1])
    
    # === One-step Prediction ===
    print("\n=== CAE+DMD One-step Prediction ===")
    start_time = time.time()
    start_cpu, start_gpu = get_memory_usage()
    
    with torch.no_grad():
        onestep_predictions = []
        
        # First timestep is reconstruction of initial state
        z_0 = cae_model.K_S(normalize_groundtruth[0:1])
        x_0_recon = cae_model.K_S_preimage(z_0)
        onestep_predictions.append(x_0_recon)
        
        # Predict each subsequent timestep from previous groundtruth
        for t in range(1, prediction_step):
            # Encode current groundtruth state
            z_current = cae_model.K_S(normalize_groundtruth[t-1:t])  # [1, hidden_dim]
            
            # Forward in latent space using DMD
            z_next = dmd.predict(z_current.T).T  # DMD expects [hidden_dim, 1]
            
            # Decode to physical space
            x_next = cae_model.K_S_preimage(z_next)
            onestep_predictions.append(x_next)
        
        onestep = torch.cat(onestep_predictions, dim=0)  # [T, C, H, W]
    
    onestep_time = time.time() - start_time
    end_cpu, end_gpu = get_memory_usage()
    
    print(f"One-step time: {onestep_time:.4f}s")
    print(f"Memory usage - CPU: {max(start_cpu, end_cpu):.2f}GB, GPU: {max(start_gpu, end_gpu):.2f}GB")
    
    inference_stats['onestep'] = {
        'total_time': onestep_time,
        'avg_time_per_frame': onestep_time / prediction_step,
        'max_cpu_memory': max(start_cpu, end_cpu),
        'max_gpu_memory': max(start_gpu, end_gpu)
    }
    
    de_onestep = denorm(onestep.cpu())
    
    # # === Rollout Prediction ===
    # print("\n=== CAE+DMD Rollout Prediction (Encode-Forward-Decode-Encode) ===")
    # start_time = time.time()
    # start_cpu, start_gpu = get_memory_usage()
    # max_cpu_rollout = start_cpu
    # max_gpu_rollout = start_gpu
    
    # step_times = []
    
    # with torch.no_grad():
    #     rollout_predictions = []
        
    #     # Initialize with first state (reconstruction)
    #     current_state = normalize_groundtruth[0:1]  # [1, C, H, W]
    #     z_current = cae_model.K_S(current_state)  # [1, hidden_dim]
    #     x_current_recon = cae_model.K_S_preimage(z_current)
    #     rollout_predictions.append(x_current_recon)
    #     current_state = x_current_recon
        
    #     # Rollout with encode-forward-decode-encode cycle
    #     for t in range(1, prediction_step):
    #         step_start = time.time()
            
    #         # 1. Encode current physical state to latent space
    #         z_current = cae_model.K_S(current_state)  # [1, hidden_dim]
            
    #         # 2. Forward in latent space using DMD
    #         z_next = dmd.predict(z_current.T).T  # DMD expects [hidden_dim, 1], return [1, hidden_dim]
            
    #         # 3. Decode latent state back to physical space
    #         current_state = cae_model.K_S_preimage(z_next)  # [1, C, H, W]
            
    #         # 4. Store prediction (the encode step is implicit in next iteration)
    #         rollout_predictions.append(current_state)
            
    #         step_time = time.time() - step_start
    #         step_times.append(step_time)
            
    #         # Monitor memory usage
    #         cpu_mem, gpu_mem = get_memory_usage()
    #         max_cpu_rollout = max(max_cpu_rollout, cpu_mem)
    #         max_gpu_rollout = max(max_gpu_rollout, gpu_mem)
        
    #     # Stack all predictions
    #     rollout = torch.cat(rollout_predictions, dim=0)  # [T, C, H, W]
    
    # rollout_time = time.time() - start_time
    # avg_step_time = sum(step_times) / len(step_times)
    
    # print(f"Rollout total time: {rollout_time:.4f}s")
    # print(f"Average time per step: {avg_step_time:.4f}s")
    # print(f"Memory usage - CPU: {max_cpu_rollout:.2f}GB, GPU: {max_gpu_rollout:.2f}GB")
    
    # inference_stats['rollout'] = {
    #     'total_time': rollout_time,
    #     'avg_time_per_step': avg_step_time,
    #     'max_cpu_memory': max_cpu_rollout,
    #     'max_gpu_memory': max_gpu_rollout,
    #     'step_times': step_times
    # }

    # === Rollout Prediction ===
    print("\n=== CAE+DMD Rollout Prediction (Latent Propagation) ===")
    start_time = time.time()
    start_cpu, start_gpu = get_memory_usage()
    max_cpu_rollout = start_cpu
    max_gpu_rollout = start_gpu

    with torch.no_grad():
        # 1. Encode initial physical state to latent space
        z_0 = cae_model.K_S(normalize_groundtruth[0:1])  # [1, hidden_dim]
        
        # 2. Propagate in latent space using DMD
        # Prepare initial latent state for DMD: [hidden_dim, 1]
        z_current = z_0.T  # [hidden_dim, 1]
        latent_states = [z_current]
        
        for t in range(1, prediction_step):
            z_next = dmd.predict(z_current)  # [hidden_dim, 1]
            latent_states.append(z_next)
            z_current = z_next
            
            # Monitor memory usage
            cpu_mem, gpu_mem = get_memory_usage()
            max_cpu_rollout = max(max_cpu_rollout, cpu_mem)
            max_gpu_rollout = max(max_gpu_rollout, gpu_mem)
        
        # 3. Stack all latent states and decode to physical space
        latent_trajectory = torch.cat(latent_states, dim=1)  # [hidden_dim, T]
        latent_trajectory = latent_trajectory.T  # [T, hidden_dim]
        
        # 4. Decode all latent states to physical space
        rollout = cae_model.K_S_preimage(latent_trajectory)  # [T, C, H, W]

    rollout_time = time.time() - start_time

    print(f"Rollout total time: {rollout_time:.4f}s")
    print(f"Average time per step: {rollout_time/prediction_step:.4f}s")
    print(f"Memory usage - CPU: {max_cpu_rollout:.2f}GB, GPU: {max_gpu_rollout:.2f}GB")

    inference_stats['rollout'] = {
        'total_time': rollout_time,
        'avg_time_per_step': rollout_time / prediction_step,
        'max_cpu_memory': max_cpu_rollout,
        'max_gpu_memory': max_gpu_rollout
    }
    
    de_rollout = denorm(rollout.cpu())
    de_groundtruth = denorm(normalize_groundtruth.cpu())
    
    print(f"[INFO] Rollout output shape: {de_rollout.shape}")
    print(f"[INFO] Denormalized onestep range: [{de_onestep.min():.4f}, {de_onestep.max():.4f}]")
    print(f"[INFO] Denormalized rollout range: [{de_rollout.min():.4f}, {de_rollout.max():.4f}]")
    print(f"[INFO] Denormalized groundtruth range: [{de_groundtruth.min():.4f}, {de_groundtruth.max():.4f}]")
    
    # Save results
    os.makedirs(fig_save_path, exist_ok=True)
    np.save(os.path.join(fig_save_path, 'era5_rollout.npy'), de_rollout.numpy())
    
    # Save inference statistics
    inference_stats['config'] = {
        'prediction_step': prediction_step,
        'start_T': start_T,
        'device': str(device),
        'hidden_dim': cae_model.hidden_dim
    }
    
    save_inference_stats(inference_stats, os.path.join(fig_save_path, 'era5_inference_stats.pkl'))
    
    # Plot comparisons
    print("\n[INFO] Generating plots...")
    plot_era5_comparisons(de_groundtruth, de_onestep, de_rollout,
                         time_indices=[1, 20, 50, 80, 99], save_dir=fig_save_path)
    
    # Compute metrics
    print("\n[INFO] Computing metrics...")
    overall_metrics = compute_era5_metrics(de_groundtruth, de_onestep, de_rollout)
    
    print("\n=== Overall Metrics ===")
    for method, metrics in overall_metrics.items():
        if method.endswith('_overall'):
            method_name = method.replace('_overall', '').title()
            print(f"{method_name} (Overall):")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.6f}")
            print()
        elif method.endswith('_per_channel'):
            method_name = method.replace('_per_channel', '').title()
            print(f"{method_name} (Per Channel):")
            for channel, channel_metrics in metrics.items():
                print(f"  {channel}:")
                for metric_name, value in channel_metrics.items():
                    print(f"    {metric_name}: {value:.6f}")
            print()
    
    print(f"\n[INFO] All results saved to: {fig_save_path}")
    print("[INFO] CAE+DMD ERA5 prediction completed successfully!")