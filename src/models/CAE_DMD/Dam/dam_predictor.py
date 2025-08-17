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

from src.utils.Dataset import DamDynamicsDataset
from src.models.DMD.base import TorchDMD
from src.models.CAE_Koopman.Dam.dam_model import DAM_C_FORWARD

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

def plot_comparisons(raw_data, onestep, rollout, time_indices=[1, 4, 7, 10], save_dir="figures"):
    """Plot comparison between groundtruth and predictions"""
    os.makedirs(save_dir, exist_ok=True)

    raw = raw_data.numpy()
    one = onestep.detach().numpy()
    roll = rollout.detach().numpy()

    titles = ['Groundtruth', 'One-step', 'Rollout']
    datas = [raw, one, roll]

    # Main comparison plot
    fig1, axes1 = plt.subplots(nrows=3, ncols=len(time_indices), figsize=(20, 14))
    all_pred_data = np.concatenate([d[:, 0] for d in datas], axis=0) 
    vmin_pred, vmax_pred = all_pred_data.min(), all_pred_data.max()

    for row in range(3):
        for col, t in enumerate(time_indices):
            img = datas[row][t, 0]
            ax = axes1[row, col]
            im = ax.imshow(img, cmap='viridis', vmin=vmin_pred, vmax=vmax_pred, origin='lower')
            ax.axis('off')
            ax.set_title(f"{titles[row]} t={t}", fontsize=18)
            fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig1.suptitle("CAE+DMD Prediction Comparison", fontsize=24)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig1.savefig(os.path.join(save_dir, "dam_comparison.png"))
    plt.close(fig1)

    # Error comparison plot
    fig2, axes2 = plt.subplots(nrows=3, ncols=len(time_indices), figsize=(20, 14))
    errors = []

    for t in time_indices:
        raw_img = raw[t, 0]
        for pred in [one, roll]:
            err = np.abs(pred[t, 0] - raw_img)
            errors.append(err)
    errors_all = np.stack(errors)
    vmin_err, vmax_err = errors_all.min(), errors_all.max()

    for col, t in enumerate(time_indices):
        img_raw = raw[t, 0]
        ax = axes2[0, col]
        im = ax.imshow(img_raw, cmap='viridis', vmin=vmin_pred, vmax=vmax_pred, origin='lower')
        ax.axis('off')
        ax.set_title(f"Groundtruth t={t}", fontsize=18)
        fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if col == 0:
            ax.set_ylabel('Groundtruth')

        for i, pred in enumerate([one, roll], start=1):
            err = np.abs(pred[t, 0] - img_raw)
            ax = axes2[i, col]
            im_err = ax.imshow(err, cmap='magma', vmin=vmin_err, vmax=vmax_err, origin='lower')
            ax.axis('off')
            ax.set_title(f"Error {titles[i]} t={t}", fontsize=18)
            fig2.colorbar(im_err, ax=ax, fraction=0.046, pad=0.04)
            if col == 0:
                ax.set_ylabel(f"Error {titles[i]}")

    fig2.suptitle("Error Comparison", fontsize=24)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig2.savefig(os.path.join(save_dir, "dam_error.png"))
    plt.close(fig2)

    print(f"[INFO] Comparison plots saved to {save_dir}")

def compute_metrics(groundtruth, onestep, rollout):
    """Compute evaluation metrics"""
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

        # RRMSE
        rmse = torch.sqrt(F.mse_loss(pred, gt))
        gt_mean = torch.mean(gt)
        rrmse = (rmse / gt_mean).item()
        
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
            'RRMSE': rrmse,
            'SSIM': avg_ssim
        }
    
    results = {}
    results['onestep'] = calculate_metrics(onestep, groundtruth)
    results['rollout'] = calculate_metrics(rollout, groundtruth)
    
    return results

if __name__ == '__main__':
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    fig_save_path = '../../../../results/CAE_DMD/figures/'
    start_T = 50
    prediction_step = 50
    val_idx = -1
    
    # Load dataset
    print("[INFO] Loading datasets...")
    dam_train_dataset = DamDynamicsDataset(data_path="../../../../data/dam/dam_train_data.npy",
                seq_length = 12,
                mean=None,
                std=None)
    
    dam_val_dataset = DamDynamicsDataset(data_path="../../../../data/dam/dam_val_data.npy",
                seq_length = 12,
                mean=None,
                std=None)
    
    denorm = dam_val_dataset.denormalizer()
    
    # Load CAE+DMD models
    print("[INFO] Loading CAE+DMD models...")
    
    # Load CAE encoder/decoder
    cae_model = DAM_C_FORWARD()
    cae_model.load_state_dict(
        torch.load('../../../../results/CAE_Koopman/Dam/model_weights/forward_model.pt', 
                  weights_only=True, map_location=device)
    )
    cae_model.to(device)
    cae_model.eval()
    
    # Load DMD model
    dmd = TorchDMD(svd_rank=150, device=device)
    dmd.load_dmd('../../../../results/CAE_DMD/Dam/dmd_model.pth')
    
    print(f"[INFO] CAE hidden dimension: {cae_model.hidden_dim}")
    print(f"[INFO] DMD eigenvalues shape: {dmd.eigenvalues.shape}")
    
    # Prepare groundtruth data
    groundtruth = dam_val_dataset.data[val_idx, start_T:start_T + prediction_step, ...]
    groundtruth = torch.tensor(groundtruth, dtype=torch.float32)
    
    # Convert to velocity magnitude
    raw_data_uv = (groundtruth[:, 0, :, :] ** 2 + groundtruth[:, 1, :, :] ** 2) ** 0.5
    raw_data_uv = raw_data_uv.unsqueeze(1)
    print(f"[INFO] Groundtruth shape: {raw_data_uv.shape}")
    
    # Normalize data
    normalize_groundtruth = dam_val_dataset.normalize(groundtruth).to(device)
    
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
    de_onestep_uv = (de_onestep[:, 0, :, :] ** 2 + de_onestep[:, 1, :, :] ** 2) ** 0.5
    de_onestep_uv = de_onestep_uv.unsqueeze(1)
    
    # === Rollout Prediction ===
    print("\n=== CAE+DMD Rollout Prediction ===")
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
    
    de_rollout = denorm(rollout.cpu())
    de_rollout_uv = (de_rollout[:, 0, :, :] ** 2 + de_rollout[:, 1, :, :] ** 2) ** 0.5
    de_rollout_uv = de_rollout_uv.unsqueeze(1)
    
    print(f"[INFO] Rollout output shape: {de_rollout_uv.shape}")
    
    # Save results
    os.makedirs(fig_save_path, exist_ok=True)
    np.save(os.path.join(fig_save_path, 'dam_rollout.npy'), de_rollout_uv.numpy())
    
    # Save inference statistics
    inference_stats['config'] = {
        'prediction_step': prediction_step,
        'val_idx': val_idx,
        'start_T': start_T,
        'device': str(device),
        'hidden_dim': cae_model.hidden_dim
    }
    
    save_inference_stats(inference_stats, os.path.join(fig_save_path, 'dam_inference_stats.pkl'))
    
    # Plot comparisons
    print("\n[INFO] Generating plots...")
    plot_comparisons(raw_data_uv, de_onestep_uv, de_rollout_uv,
                    time_indices=[1, 10, 20, 30, 40], save_dir=fig_save_path)
    
    # Compute metrics
    print("\n[INFO] Computing metrics...")
    overall_metrics = compute_metrics(raw_data_uv, de_onestep_uv, de_rollout_uv)
    
    print("\n=== Overall Metrics ===")
    for method, metrics in overall_metrics.items():
        print(f"{method}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.6f}")
        print()
    
    print(f"\n[INFO] All results saved to: {fig_save_path}")
    print("[INFO] CAE+DMD prediction completed successfully!")