import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import psutil
import time
import pickle
import os
import sys

current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
sys.path.append(src_directory)

from src.utils.Dataset import KolDynamicsDataset
from src.models.DMD.base import TorchDMD

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
            im = ax.imshow(img, cmap='RdBu_r', vmin=vmin_pred, vmax=vmax_pred)
            ax.axis('off')
            ax.set_title(f"{titles[row]} t={t}", fontsize=18)
            fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig1.suptitle("DMD Prediction Comparison", fontsize=24)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig1.savefig(os.path.join(save_dir, "kol_comparison.png"))
    plt.close(fig1)

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
        im = ax.imshow(img_raw, cmap='RdBu_r', vmin=vmin_pred, vmax=vmax_pred)
        ax.axis('off')
        ax.set_title(f"Groundtruth t={t}", fontsize=18)
        fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if col == 0:
            ax.set_ylabel('Groundtruth')

        for i, pred in enumerate([one, roll], start=1):
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

    print(f"[INFO] Comparison plot saved to {save_dir}")

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
    results['onestep'] = calculate_metrics(onestep, groundtruth)
    results['rollout'] = calculate_metrics(rollout, groundtruth)
    
    return results

def compute_temporal_metrics(groundtruth, onestep, rollout):
    """Compute temporal evaluation metrics"""
    def calculate_temporal_metrics(pred, gt):
        T = pred.shape[0]
        pred_np = pred.detach().cpu().numpy()
        gt_np = gt.detach().cpu().numpy()
        
        mse_list = []
        mae_list = []
        relative_l2_list = []
        ssim_list = []
        
        for t in range(T):
            pred_t = pred[t:t+1]
            gt_t = gt[t:t+1]
            
            mse = F.mse_loss(pred_t, gt_t).item()
            mae = F.l1_loss(pred_t, gt_t).item()
            
            l2_error = torch.norm(pred_t - gt_t)
            l2_gt = torch.norm(gt_t)
            relative_l2 = (l2_error / l2_gt).item()
            
            ssim_val = ssim(gt_np[t, 0], pred_np[t, 0], data_range=gt_np[t, 0].max() - gt_np[t, 0].min())
            
            mse_list.append(mse)
            mae_list.append(mae)
            relative_l2_list.append(relative_l2)
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

# Define encoder, decoder, and latent_forward functions
def encoder(x_t):
    x_t_complex = x_t.to(torch.complex64)
    b_t = torch.linalg.lstsq(dmd.modes, x_t_complex).solution
    return b_t

def decoder(b_t):
    x_t = dmd.modes @ b_t
    return x_t.real

def latent_forward(b_t):
    b_tp = torch.diag(dmd.eigenvalues) @ b_t
    return b_tp

if __name__ == '__main__':
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    fig_save_path = '../../../../results/DMD/figures/'
    start_T = 150
    prediction_step = 100
    val_idx = 3
    
    # Load dataset
    print("[INFO] Loading datasets...")
    kol_train_dataset = KolDynamicsDataset(
        data_path="../../../../data/kolmogorov/RE40_T20/kolmogorov_train_data.npy",
        seq_length=prediction_step,
        mean=None,
        std=None
    )
    
    kol_val_dataset = KolDynamicsDataset(
        data_path="../../../../data/kolmogorov/RE40_T20/kolmogorov_val_data.npy",
        seq_length=prediction_step,
        mean=kol_train_dataset.mean,
        std=kol_train_dataset.std
    )
    
    denorm = kol_val_dataset.denormalizer()
    
    # Load DMD model
    print("[INFO] Loading DMD model...")
    dmd = TorchDMD(svd_rank=512, device=device)
    dmd.load_dmd('../../../../results/DMD/KMG/dmd_model.pth')
    
    # Prepare groundtruth data
    groundtruth = kol_val_dataset.data[val_idx, start_T:start_T + prediction_step, ...]
    groundtruth = torch.tensor(groundtruth, dtype=torch.float32)
    print(f"[INFO] Groundtruth shape: {groundtruth.shape}")
    print(f"[INFO] Groundtruth range: [{groundtruth.min():.4f}, {groundtruth.max():.4f}]")
    
    # Normalize data
    normalize_groundtruth = kol_val_dataset.normalize(groundtruth)
    print(f"[INFO] Normalized shape: {normalize_groundtruth.shape}")
    print(f"[INFO] Normalized range: [{normalize_groundtruth.min():.4f}, {normalize_groundtruth.max():.4f}]")
    
    # Prepare data for DMD: flatten each timestep
    state_flat = prepare_dmd_data(normalize_groundtruth, device)  # [D, T]
    print(f"[INFO] Flattened data shape: {state_flat.shape}")
    
    inference_stats = {}
    
    # === One-step Prediction ===
    print("\n=== DMD One-step Prediction ===")
    start_time = time.time()
    start_cpu, start_gpu = get_memory_usage()
    
    with torch.no_grad():
        onestep_states = []
        # First timestep is the same as groundtruth
        onestep_states.append(state_flat[:, 0:1])
        
        # Predict each subsequent timestep from the previous groundtruth
        for t in range(prediction_step - 1):
            x_t = state_flat[:, t]  # [D]
            # Encode to latent space
            b_t = encoder(x_t)
            # Forward in latent space
            b_tp = latent_forward(b_t)
            # Decode back to physical space
            x_next = decoder(b_tp)
            onestep_states.append(x_next.unsqueeze(1))
        
        onestep_flat = torch.cat(onestep_states, dim=1)  # [D, T]
    
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
    
    onestep = reconstruct_from_dmd(onestep_flat, normalize_groundtruth.shape)
    de_onestep = denorm(onestep)
    
    # === Rollout Prediction ===
    print("\n=== DMD Rollout Prediction ===")
    start_time = time.time()
    start_cpu, start_gpu = get_memory_usage()
    max_cpu_rollout = start_cpu
    max_gpu_rollout = start_gpu
    
    with torch.no_grad():
        rollout_states = []
        # First timestep is the same as groundtruth
        rollout_states.append(state_flat[:, 0:1])
        
        # Initialize with first state
        current_state = state_flat[:, 0]  # [D]
        
        # Rollout with encode-forward-decode at each step
        for t in range(1, prediction_step):
            # Encode current state
            b_current = encoder(current_state)
            # Forward in latent space
            b_next = latent_forward(b_current)
            # Decode to physical space
            current_state = decoder(b_next)
            rollout_states.append(current_state.unsqueeze(1))
        
        rollout_flat = torch.cat(rollout_states, dim=1)  # [D, T]
    
    rollout_time = time.time() - start_time
    cpu_mem, gpu_mem = get_memory_usage()
    max_cpu_rollout = max(max_cpu_rollout, cpu_mem)
    max_gpu_rollout = max(max_gpu_rollout, gpu_mem)
    
    print(f"Rollout total time: {rollout_time:.4f}s")
    print(f"Average time per step: {rollout_time/prediction_step:.4f}s")
    print(f"Memory usage - CPU: {max_cpu_rollout:.2f}GB, GPU: {max_gpu_rollout:.2f}GB")
    
    inference_stats['rollout'] = {
        'total_time': rollout_time,
        'avg_time_per_step': rollout_time / prediction_step,
        'max_cpu_memory': max_cpu_rollout,
        'max_gpu_memory': max_gpu_rollout
    }
    
    rollout = reconstruct_from_dmd(rollout_flat, normalize_groundtruth.shape)
    de_rollout = denorm(rollout)

    np.save(fig_save_path + 'kol_rollout.npy', de_rollout)
    
    # Save inference statistics
    inference_stats['config'] = {
        'prediction_step': prediction_step,
        'val_idx': val_idx,
        'start_T': start_T,
        'device': str(device)
    }
    
    save_inference_stats(inference_stats, os.path.join(fig_save_path, 'kol_inference_stats.pkl'))
    
    # Plot comparisons
    print("\n[INFO] Generating plots...")
    plot_comparisons(groundtruth, de_onestep, de_rollout,
                    time_indices=[1, 30, 60, 90], save_dir=fig_save_path)
    
    # Compute metrics
    print("\n[INFO] Computing metrics...")
    overall_metrics = compute_metrics(groundtruth, de_onestep, de_rollout)
    
    print("\n=== Overall Metrics ===")
    for method, metrics in overall_metrics.items():
        print(f"{method}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.6f}")
        print()
    
    # Temporal metrics and plot
    # temporal_metrics = compute_temporal_metrics(groundtruth, de_onestep, de_rollout)
    
    # fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # metrics_names = ['MSE', 'MAE', 'Relative_L2', 'SSIM']
    
    # for i, metric_name in enumerate(metrics_names):
    #     ax = axes[i//2, i%2]
        
    #     for method in ['reconstruction', 'onestep', 'rollout']:
    #         values = temporal_metrics[method][metric_name]
    #         ax.plot(values, label=method)
        
    #     ax.set_xlabel('Time Step')
    #     ax.set_ylabel(metric_name)
    #     ax.set_title(f'DMD {metric_name} over Time (Kolmogorov)')
    #     ax.legend()
    #     ax.grid(True)
    
    # plt.tight_layout()
    # plt.savefig(os.path.join(fig_save_path, "kol_perframe_metric.png"))
    # plt.close()
    
    print(f"\n[INFO] All results saved to: {fig_save_path}")
    print("[INFO] DMD Kolmogorov prediction completed successfully!")