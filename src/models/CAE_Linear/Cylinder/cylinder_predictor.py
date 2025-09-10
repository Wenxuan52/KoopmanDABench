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

from src.utils.Dataset import CylinderDynamicsDataset

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
        rrmse_list = []
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
            
            # RRMSE
            rmse = torch.sqrt(F.mse_loss(pred_t, gt_t))
            gt_mean = torch.mean(gt_t)
            rrmse = (rmse / gt_mean).item()
            rrmse_list.append(rrmse)
            
            # SSIM
            ssim_val = ssim(gt_np[t, 0], pred_np[t, 0], data_range=gt_np[t, 0].max() - gt_np[t, 0].min())
            ssim_list.append(ssim_val)
        
        return {
            'MSE': mse_list,
            'MAE': mae_list,
            'Relative_L2': relative_l2_list,
            'RRMSE': rrmse_list,
            'SSIM': ssim_list
        }
    
    results = {}
    results['reconstruction'] = calculate_temporal_metrics(reconstruction, groundtruth)
    results['onestep'] = calculate_temporal_metrics(onestep, groundtruth)
    results['rollout'] = calculate_temporal_metrics(rollout, groundtruth)
    
    return results


if __name__ == '__main__':
    from cylinder_model import CYLINDER_C_FORWARD

    fig_save_path = '../../../../results/CAE_Linear/figures/'
    
    start_T = 700
    
    prediction_step = 10
    
    foward_step = 12

    val_idx = 3

    cyl_train_dataset = CylinderDynamicsDataset(data_path="../../../../data/cylinder/cylinder_train_data.npy",
                seq_length = foward_step,
                mean=None,
                std=None)
    
    cyl_val_dataset = CylinderDynamicsDataset(data_path="../../../../data/cylinder/cylinder_val_data.npy",
                seq_length = foward_step,
                mean=cyl_train_dataset.mean,
                std=cyl_train_dataset.std)

    denorm = cyl_val_dataset.denormalizer()

    groundtruth = cyl_val_dataset.data[val_idx, start_T:start_T + prediction_step, ...]
    groundtruth = torch.tensor(groundtruth, dtype=torch.float32)
    # print(groundtruth.shape)
    # print(groundtruth.min())
    # print(groundtruth.max())
    
    raw_data_uv = (groundtruth[:, 0, :, :] ** 2 + groundtruth[:, 1, :, :] ** 2) ** 0.5
    raw_data_uv = raw_data_uv.unsqueeze(1)
    print(raw_data_uv.shape)
    
    forward_model = CYLINDER_C_FORWARD()
    forward_model.load_state_dict(torch.load('../../../../results/CAE_Linear/Cylinder/model_weights/forward_model.pt', weights_only=True, map_location='cpu'))
    forward_model.eval()

    print(forward_model)

    normalize_groundtruth = cyl_val_dataset.normalize(groundtruth)

    print(normalize_groundtruth.shape)
    print(normalize_groundtruth.min())
    print(normalize_groundtruth.max())

    state = normalize_groundtruth

    inference_stats = {}
    
    # Warm Up
    with torch.no_grad():
        z = forward_model.K_S(state)
        reconstruct = forward_model.K_S_preimage(z)

    print("=== Reconstruction Inference ===")
    start_time = time.time()
    start_cpu, start_gpu = get_memory_usage()
    
    with torch.no_grad():

        z = forward_model.K_S(state)
        reconstruct = forward_model.K_S_preimage(z)

    recon_time = time.time() - start_time
    end_cpu, end_gpu = get_memory_usage()
    max_cpu_recon = max(start_cpu, end_cpu)
    max_gpu_recon = max(start_gpu, end_gpu)
    
    print(f"Reconstruction time: {recon_time:.4f}s")
    print(f"Memory usage - CPU: {max_cpu_recon:.2f}GB, GPU: {max_gpu_recon:.2f}GB")
    
    inference_stats['reconstruction'] = {
        'total_time': recon_time,
        'avg_time_per_frame': recon_time / prediction_step,
        'max_cpu_memory': max_cpu_recon,
        'max_gpu_memory': max_gpu_recon
    }

    de_reconstruct = denorm(reconstruct)
    de_reconstruct_uv = (de_reconstruct[:, 0, :, :] ** 2 + de_reconstruct[:, 1, :, :] ** 2) ** 0.5
    de_reconstruct_uv = de_reconstruct_uv.unsqueeze(1)
    # print(de_reconstruct_uv.shape)
    # print(de_reconstruct_uv.min())
    # print(de_reconstruct_uv.max())

    print("\n=== One-step Inference ===")
    start_time = time.time()
    start_cpu, start_gpu = get_memory_usage()

    with torch.no_grad():
        z_current = forward_model.K_S(state)
        z_next = forward_model.latent_forward(z_current)
        onestep = forward_model.K_S_preimage(z_next)

    onestep_time = time.time() - start_time
    end_cpu, end_gpu = get_memory_usage()
    max_cpu_onestep = max(start_cpu, end_cpu)
    max_gpu_onestep = max(start_gpu, end_gpu)
    
    print(f"One-step time: {onestep_time:.4f}s")
    print(f"Memory usage - CPU: {max_cpu_onestep:.2f}GB, GPU: {max_gpu_onestep:.2f}GB")
    
    inference_stats['onestep'] = {
        'total_time': onestep_time,
        'avg_time_per_frame': onestep_time / prediction_step,
        'max_cpu_memory': max_cpu_onestep,
        'max_gpu_memory': max_gpu_onestep
    }

    onestep = torch.cat([normalize_groundtruth[0:1, ...], onestep[:-1, ...]])
    de_onestep = denorm(onestep)
    de_onestep_uv = (de_onestep[:, 0, :, :] ** 2 + de_onestep[:, 1, :, :] ** 2) ** 0.5
    de_onestep_uv = de_onestep_uv.unsqueeze(1)
    # print(de_onestep_uv.shape)
    # print(de_onestep_uv.min())
    # print(de_onestep_uv.max())

    print("\n=== Rollout Inference ===")
    predictions = []
    current_state = state[0, ...].unsqueeze(0)
    # print(current_state.shape)
    n_steps = prediction_step

    start_time = time.time()
    start_cpu, start_gpu = get_memory_usage()
    max_cpu_rollout = start_cpu
    max_gpu_rollout = start_gpu

    # step_times = []
    
    # with torch.no_grad():
    #     for step in range(n_steps):
    #         step_start = time.time()

    #         z_current = forward_model.K_S(current_state)
    #         z_next = forward_model.latent_forward(z_current)
    #         next_state = forward_model.K_S_preimage(z_next)
            
    #         predictions.append(next_state)
    #         current_state = next_state

    #         step_time = time.time() - step_start
    #         step_times.append(step_time)
            
    #         cpu_mem, gpu_mem = get_memory_usage()
    #         max_cpu_rollout = max(max_cpu_rollout, cpu_mem)
    #         max_gpu_rollout = max(max_gpu_rollout, gpu_mem)

    with torch.no_grad():
        step_start = time.time()
        
        z_current = forward_model.K_S(current_state)
        latent_predictions = [z_current]
        
        for step in range(n_steps):
            z_next = forward_model.latent_forward(z_current)
            latent_predictions.append(z_next)
            z_current = z_next
        
        latent_time = time.time() - step_start
        
        decode_start = time.time()
        predictions = []
        for z_pred in latent_predictions[1:]:
            state_pred = forward_model.K_S_preimage(z_pred)
            predictions.append(state_pred)
        
        decode_time = time.time() - decode_start
        total_time = time.time() - step_start
        
        cpu_mem, gpu_mem = get_memory_usage()
        max_cpu_rollout = max(max_cpu_rollout, cpu_mem)
        max_gpu_rollout = max(max_gpu_rollout, gpu_mem)
        
        step_times = [latent_time / n_steps] * n_steps
        print(f"Latent propagation time: {latent_time:.4f}s")
        print(f"Decoding time: {decode_time:.4f}s") 
        print(f"Total time: {total_time:.4f}s")
    
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

    rollout = torch.cat(predictions, dim=0)
    rollout = torch.cat([normalize_groundtruth[0:1, ...], rollout[:-1, ...]])

    # print(rollout.shape)

    de_rollout = denorm(rollout)
    de_rollout_uv = (de_rollout[:, 0, :, :] ** 2 + de_rollout[:, 1, :, :] ** 2) ** 0.5
    de_rollout_uv = de_rollout_uv.unsqueeze(1)
    # print(de_rollout_uv.shape)
    # print(de_rollout_uv.min())
    # print(de_rollout_uv.max())

    np.save(fig_save_path + 'cyl_rollout_new.npy', de_rollout_uv)

    inference_stats['config'] = {
        'prediction_step': prediction_step,
        'forward_step': foward_step,
        'val_idx': val_idx,
        'start_T': start_T
    }
    
    save_inference_stats(inference_stats, os.path.join(fig_save_path, 'cyl_inference_stats.pkl'))

    plot_comparisons(raw_data_uv, de_reconstruct_uv, de_onestep_uv, de_rollout_uv,
                    time_indices=[1, 2, 4, 7, 9], save_dir=fig_save_path)

    # plot_comparisons(raw_data_uv, de_reconstruct_uv, de_onestep_uv, de_rollout_uv,
    #                 time_indices=[1, 5, 10, 15, 20], save_dir=fig_save_path)
    

    # Compute Metric

    overall_metrics = compute_metrics(raw_data_uv, de_reconstruct_uv, de_onestep_uv, de_rollout_uv)
    
    print("Overall metric:")
    for method, metrics in overall_metrics.items():
        print(f"{method}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.6f}")
        print()
    
    temporal_metrics = compute_temporal_metrics(raw_data_uv, de_reconstruct_uv, de_onestep_uv, de_rollout_uv)
    
    fig, axes = plt.subplots(1, 5, figsize=(40, 10))
    metrics_names = ['MSE', 'MAE', 'Relative_L2', 'RRMSE', 'SSIM']
    
    for i, metric_name in enumerate(metrics_names):
        ax = axes[i]
        
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
