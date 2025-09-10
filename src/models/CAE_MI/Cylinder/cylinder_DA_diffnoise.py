#!/usr/bin/env python
# coding: utf-8

"""
Noise Level Data Assimilation Experiment
Testing different noise levels in sparse observations with fixed observation density (~5%)
"""

import random
import os
import sys
import pickle
from time import perf_counter
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

# Set matplotlib backend and config directory
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_config'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors

# Add src directory to path
current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
sys.path.append(src_directory)

from cylinder_model import CYLINDER_C_FORWARD
from src.utils.Dataset import CylinderDynamicsDataset
import torchda


def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def seed_worker(worker_id):
    """Worker seed for DataLoader"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_device():
    """Set device to GPU if available, otherwise CPU"""
    if torch.cuda.device_count() == 0:
        return "cpu"
    torch.set_float32_matmul_precision("high")
    return "cuda"


class NoisySparseObservationHandler:
    """Handler for sparse observations with added noise at fixed density"""
    def __init__(self, obs_ratio: float = 0.05, noise_std: float = 0.0, seed: int = 42):
        self.obs_ratio = obs_ratio
        self.noise_std = noise_std
        self.seed = seed
        self.observation_positions = None
        self.obs_count = 0
        self.total_pixels = 0
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
    
    def generate_observation_positions(self, image_shape: tuple) -> int:
        """Generate fixed observation positions for all time steps"""
        if len(image_shape) == 3:
            C, H, W = image_shape
        else:
            raise ValueError(f"Expected 3D image shape (C, H, W), got {image_shape}")
        
        self.total_pixels = C * H * W
        self.obs_count = int(self.total_pixels * self.obs_ratio)
        self.obs_count = max(1, self.obs_count)  # Ensure at least 1 observation
        
        # Generate fixed random positions (same for all time steps)
        self.observation_positions = torch.randperm(self.total_pixels)[:self.obs_count]
        
        actual_ratio = self.obs_count / self.total_pixels
        print(f"Fixed observation setup: {self.obs_count} observations ({actual_ratio:.3%} ratio)")
        print(f"Noise level: σ = {self.noise_std:.4f}")
        
        return self.obs_count
    
    def get_obs_count(self):
        """Get observation count"""
        return self.obs_count
    
    def apply_observation(self, full_image: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        """Apply observation mask to full image and add noise"""
        if self.observation_positions is None:
            raise ValueError("Observation positions not generated. Call generate_observation_positions first.")
        
        flat_image = full_image.flatten()
        obs_vector = flat_image[self.observation_positions]
        
        # Add noise if requested
        if add_noise and self.noise_std > 0:
            noise = torch.randn_like(obs_vector) * self.noise_std
            obs_vector = obs_vector + noise
        
        return obs_vector
    
    def create_R_matrix(self, base_variance=None):
        """Create observation covariance matrix based on noise level"""
        if base_variance is None:
            # Use noise variance as base variance, with minimum value
            base_variance = max(self.noise_std**2, 1e-6)
        
        R = torch.eye(self.obs_count) * base_variance
        return R
    
    def get_signal_to_noise_ratio(self, signal_std: float):
        """Calculate signal-to-noise ratio"""
        if self.noise_std == 0:
            return float('inf')
        return signal_std / self.noise_std


# Global variables for observation handler
_global_obs_handler = None


def H_sparse_noisy(x):
    """Observation operator for noisy sparse observations"""
    global _global_obs_handler, forward_model
    
    x_reconstructed = forward_model.K_S_preimage(x)
    sparse_obs = _global_obs_handler.apply_observation(x_reconstructed.squeeze(), add_noise=False)  # Noise already added to observations
    return sparse_obs.unsqueeze(0)


def dmd_wrapper(z_t, time_fw=None, *args):
    """Wrapper for DMD forward model"""
    if time_fw is None:
        if z_t.ndim == 1:
            z_t = z_t.unsqueeze(0)
        z_tp = forward_model.latent_forward(z_t)
    else:
        if z_t.ndim == 1:
            z_t = z_t.unsqueeze(0)

        z_tp = torch.empty((time_fw.shape[0], z_t.shape[0], z_t.shape[1]), device=z_t.device)
        
        current_state = forward_model.K_S_preimage(z_t)
        
        for i in range(int(time_fw.shape[0])):
            z_current = forward_model.K_S(current_state)
            z_tp[i] = z_current
            
            if i < int(time_fw.shape[0]) - 1:
                z_next = forward_model.latent_forward(z_current)
                current_state = forward_model.K_S_preimage(z_next)
    
    return z_tp


def run_single_noise_da_experiment(
    noise_std: float,
    obs_ratio: float = 0.05,
    model_name: str = "CAE_DMD",
    start_da_end_idxs: tuple = (700, 800, 900),
    time_obs: list = None,
    gaps: list = None,
    early_stop_config: tuple = (100, 1e-2),
    device: str = "cuda"
):
    """Run single data assimilation experiment with specific noise level"""
    
    # Set default values if not provided
    if time_obs is None:
        time_obs = [
            start_da_end_idxs[1],
            start_da_end_idxs[1] + 10,
            start_da_end_idxs[1] + 20,
        ]
    
    if gaps is None:
        gaps = [10] * (len(time_obs) - 1)
    
    global forward_model, _global_obs_handler
    
    print(f"\n{'='*60}")
    print(f"Running DA experiment with noise level: σ = {noise_std:.4f}")
    print(f"Fixed observation ratio: {obs_ratio:.1%}")
    print(f"{'='*60}")
    
    # Initialize observation handler with noise
    obs_handler = NoisySparseObservationHandler(
        obs_ratio=obs_ratio, 
        noise_std=noise_std, 
        seed=42
    )
    _global_obs_handler = obs_handler
    
    # Load datasets
    forward_step = 12
    val_idx = 3
    
    cyl_train_dataset = CylinderDynamicsDataset(
        data_path="../../../../data/cylinder/cylinder_train_data.npy",
        seq_length=forward_step,
        mean=None,
        std=None
    )
    
    cyl_val_dataset = CylinderDynamicsDataset(
        data_path="../../../../data/cylinder/cylinder_val_data.npy",
        seq_length=forward_step,
        mean=cyl_train_dataset.mean,
        std=cyl_train_dataset.std
    )
    
    # Create a device-aware denormalizer
    def safe_denorm(x):
        """Device-aware denormalization"""
        if isinstance(x, torch.Tensor):
            x_cpu = x.cpu() if x.is_cuda else x
            mean = cyl_val_dataset.mean.reshape(1, -1, 1, 1)
            std = cyl_val_dataset.std.reshape(1, -1, 1, 1)
            return (x_cpu * std + mean).cpu()
        else:
            return x
    
    # Get ground truth data
    groundtruth = cyl_val_dataset.data[val_idx, ...]
    groundtruth = torch.from_numpy(groundtruth)
    print(f"Ground truth shape: {groundtruth.shape}")
    
    # Prepare observation data
    full_y_data = [
        cyl_val_dataset.normalize(groundtruth[i+1, ...])
        for i in range(groundtruth.shape[0])
        if i in time_obs
    ]
    full_y_data = torch.cat(full_y_data).to(device)
    
    # Generate observation positions (fixed for all time steps)
    sample_image_shape = full_y_data[0].shape
    obs_count = obs_handler.generate_observation_positions(sample_image_shape)
    
    # Create noisy observations
    noisy_y_data = []
    clean_y_data = []
    signal_stds = []
    
    for full_img in full_y_data:
        # Get clean observations
        clean_obs = obs_handler.apply_observation(full_img, add_noise=False)
        clean_y_data.append(clean_obs)
        
        # Get noisy observations
        noisy_obs = obs_handler.apply_observation(full_img, add_noise=True)
        noisy_y_data.append(noisy_obs)
        
        # Calculate signal standard deviation for SNR
        signal_stds.append(clean_obs.std().item())
    
    # Calculate Signal-to-Noise Ratio
    avg_signal_std = np.mean(signal_stds)
    snr = obs_handler.get_signal_to_noise_ratio(avg_signal_std)
    
    print(f"Average signal std: {avg_signal_std:.4f}")
    print(f"Noise std: {noise_std:.4f}")
    print(f"Signal-to-Noise Ratio: {snr:.2f}" if snr != float('inf') else "Signal-to-Noise Ratio: ∞ (no noise)")
    
    # Stack noisy observations for 4D-Var
    noisy_y_data_tensor = torch.stack(noisy_y_data)
    
    # Set up DA matrices
    latent_dim = forward_model.C_forward.shape[0]
    B = torch.eye(latent_dim, device=device)
    R = obs_handler.create_R_matrix().to(device)
    
    print(f"Observation dimension: {obs_count}")
    print(f"R matrix diagonal value: {R[0,0].item():.6f}")
    
    # Configure 4D-Var
    case_to_run = (
        torchda.CaseBuilder()
        .set_observation_time_steps(time_obs)
        .set_gaps(gaps)
        .set_forward_model(dmd_wrapper)
        .set_observation_model(H_sparse_noisy)
        .set_background_covariance_matrix(B)
        .set_observation_covariance_matrix(R)
        .set_observations(noisy_y_data_tensor)
        .set_optimizer_cls(torch.optim.Adam)
        .set_optimizer_args({"lr": 0.05})
        .set_max_iterations(5000)
        .set_early_stop(early_stop_config)
        .set_algorithm(torchda.Algorithms.Var4D)
        .set_device(torchda.Device.GPU)
        .set_output_sequence_length(1)
    )
    
    # Run 4D-Var assimilation
    print("Running 4D-Var data assimilation...")
    outs_4d_da = []
    start_time = perf_counter()
    
    current_state = cyl_val_dataset.normalize(groundtruth[start_da_end_idxs[0]]).to(device)
    
    da_metrics = {
        'final_cost': None,
        'num_iterations': None,
        'avg_time_per_iteration': None,
        'total_da_time': None,
        'noise_std': noise_std,
        'snr': snr,
        'avg_signal_std': avg_signal_std
    }
    
    for i in range(start_da_end_idxs[0], start_da_end_idxs[-1] + 1):
        z_current = forward_model.K_S(current_state)
        
        if i == start_da_end_idxs[1]:
            case_to_run.set_background_state(z_current.ravel())
            
            da_start_time = perf_counter()
            result = case_to_run.execute()
            da_time = perf_counter() - da_start_time
            z_assimilated = result["assimilated_state"]
            
            intermediate_results = result["intermediate_results"]
            final_cost = intermediate_results["J"][-1]
            num_iterations = len(intermediate_results['J'])
            avg_time_per_iteration = da_time / num_iterations

            da_metrics['final_cost'] = final_cost
            da_metrics['num_iterations'] = num_iterations
            da_metrics['avg_time_per_iteration'] = avg_time_per_iteration
            da_metrics['total_da_time'] = da_time
            
            outs_4d_da.append(z_assimilated)
            current_state = forward_model.K_S_preimage(z_assimilated)
        else:
            outs_4d_da.append(z_current)
            z_next = dmd_wrapper(z_current)
            current_state = forward_model.K_S_preimage(z_next)
    
    total_time = perf_counter() - start_time
    print(f"Total experiment time: {total_time:.2f}s")
    
    # Run baseline (no DA)
    print("Running baseline (no DA)...")
    outs_no_4d_da = []
    
    with torch.no_grad():
        current_state = cyl_val_dataset.normalize(groundtruth[start_da_end_idxs[0]]).to(device)
        
        for i in range(start_da_end_idxs[0], start_da_end_idxs[-1] + 1):
            z_current = forward_model.K_S(current_state)
            outs_no_4d_da.append(z_current)
            
            z_next = forward_model.latent_forward(z_current)
            next_state = forward_model.K_S_preimage(z_next)
            current_state = next_state
    
    # Compute metrics
    print("Computing metrics...")
    diffs_da_real_mse = []
    diffs_noda_real_mse = []
    diffs_da_real_rrmse = []
    diffs_noda_real_rrmse = []
    diffs_da_real_ssim = []
    diffs_noda_real_ssim = []
    
    with torch.no_grad():
        for i, (no_da, da) in enumerate(zip(outs_no_4d_da, outs_4d_da), start=start_da_end_idxs[0]):
            da_img = forward_model.K_S_preimage(da).view(2, 64, 64).cpu()
            noda_img = forward_model.K_S_preimage(no_da).view(2, 64, 64).cpu()
            
            de_da_img = safe_denorm(da_img)
            de_noda_img = safe_denorm(noda_img)

            if i == 800:
                da_minus_real_img_square = (de_da_img[0] - groundtruth[i+1]) ** 2
                noda_minus_real_img_square = (de_noda_img[0] - groundtruth[i]) ** 2
            else:
                da_minus_real_img_square = (de_da_img[0] - groundtruth[i]) ** 2
                noda_minus_real_img_square = (de_noda_img[0] - groundtruth[i]) ** 2
            
            diffs_da_real_mse.append(da_minus_real_img_square.mean().item())
            diffs_noda_real_mse.append(noda_minus_real_img_square.mean().item())

            if i == 800:
                diffs_da_real_rrmse.append((da_minus_real_img_square.sum()/((groundtruth[i+1]**2).sum())).sqrt().item())
                diffs_noda_real_rrmse.append((noda_minus_real_img_square.sum()/((groundtruth[i]**2).sum())).sqrt().item())
                
                diffs_da_real_ssim.append(ssim(groundtruth[i+1].numpy(), de_da_img[0].numpy(), data_range=1, channel_axis=0))
                diffs_noda_real_ssim.append(ssim(groundtruth[i].numpy(), de_noda_img[0].numpy(), data_range=1, channel_axis=0))
            else:
                diffs_da_real_rrmse.append((da_minus_real_img_square.sum()/((groundtruth[i]**2).sum())).sqrt().item())
                diffs_noda_real_rrmse.append((noda_minus_real_img_square.sum()/((groundtruth[i]**2).sum())).sqrt().item())
                
                diffs_da_real_ssim.append(ssim(groundtruth[i].numpy(), de_da_img[0].numpy(), data_range=1, channel_axis=0))
                diffs_noda_real_ssim.append(ssim(groundtruth[i].numpy(), de_noda_img[0].numpy(), data_range=1, channel_axis=0))
    
    # Package results
    experiment_results = {
        'noise_std': noise_std,
        'obs_ratio': obs_ratio,
        'snr': snr,
        'avg_signal_std': avg_signal_std,
        'da_metrics': da_metrics,
        'diffs_da_real_mse': diffs_da_real_mse,
        'diffs_noda_real_mse': diffs_noda_real_mse,
        'diffs_da_real_rrmse': diffs_da_real_rrmse,
        'diffs_noda_real_rrmse': diffs_noda_real_rrmse,
        'diffs_da_real_ssim': diffs_da_real_ssim,
        'diffs_noda_real_ssim': diffs_noda_real_ssim,
        'time_obs': time_obs,
        'start_da_end_idxs': start_da_end_idxs
    }
    
    return experiment_results


def plot_noise_level_comparison(all_results, model_name="CAE_DMD", model_display_name="CAE DMD"):
    """Plot comparison of different noise levels with gradient red colors"""
    
    # Create gradient red colormap
    noise_stds = [result['noise_std'] for result in all_results]
    n_noise_levels = len(noise_stds)
    
    # Generate gradient colors from blue (low noise) to red (high noise)
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, n_noise_levels))
    
    # Get step indices from first result
    start_da_end_idxs = all_results[0]['start_da_end_idxs']
    step_idxs = list(range(start_da_end_idxs[0] + 1, start_da_end_idxs[-1] + 2))
    time_obs = all_results[0]['time_obs']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot MSE comparison for DA
    for i, result in enumerate(all_results):
        noise_std = result['noise_std']
        snr = result['snr']
        color = colors[i]
        
        if snr == float('inf'):
            label = f'σ = {noise_std:.4f} (No Noise)'
        else:
            label = f'σ = {noise_std:.4f} (SNR: {snr:.1f})'
        
        ax1.plot(step_idxs, result['diffs_da_real_mse'], 
                color=color, linewidth=2, 
                label=label, alpha=0.8)
    
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('MSE', fontsize=12)
    ax1.set_title(f'{model_display_name}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Add observation time markers
    for x in time_obs:
        ax1.axvline(x=x+1, color="red", linestyle="--", alpha=0.6, linewidth=1)
    
    # Plot MSE comparison for No DA (baseline)
    for i, result in enumerate(all_results):
        noise_std = result['noise_std']
        snr = result['snr']
        color = colors[i]
        
        if snr == float('inf'):
            label = f'σ = {noise_std:.4f} (No Noise)'
        else:
            label = f'σ = {noise_std:.4f} (SNR: {snr:.1f})'
        
        ax2.plot(step_idxs, result['diffs_noda_real_mse'], 
                color=color, linewidth=2, 
                label=label, alpha=0.8)
    
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('MSE', fontsize=12)
    ax2.set_title(f'{model_display_name}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Add observation time markers
    for x in time_obs:
        ax2.axvline(x=x+1, color="red", linestyle="--", alpha=0.6, linewidth=1)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = f'../../../../results/{model_name}/DA/cyl_diffnoise_comparison.png'
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Noise level comparison plot saved to: {plot_path}")


def plot_single_model_noise_comparison(all_results, model_name="CAE_DMD", model_display_name="CAE DMD"):
    """Plot single model with different noise levels (recommended visualization)"""
    
    # Create gradient colormap
    noise_stds = [result['noise_std'] for result in all_results]
    n_noise_levels = len(noise_stds)
    
    # Generate gradient colors from blue (low noise) to red (high noise)
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, n_noise_levels))
    
    # Get step indices from first result
    start_da_end_idxs = all_results[0]['start_da_end_idxs']
    step_idxs = list(range(start_da_end_idxs[0] + 1, start_da_end_idxs[-1] + 2))
    time_obs = all_results[0]['time_obs']
    obs_ratio = all_results[0]['obs_ratio']
    
    # Create single figure focusing on DA performance
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot MSE for different noise levels
    for i, result in enumerate(all_results):
        noise_std = result['noise_std']
        snr = result['snr']
        color = colors[i]
        
        if snr == float('inf'):
            label = f'σ = {noise_std:.4f} (No Noise)'
        else:
            label = f'σ = {noise_std:.4f} (SNR: {snr:.1f})'
        
        ax.plot(step_idxs, result['diffs_da_real_mse'], 
               color=color, linewidth=3, 
               label=label, alpha=0.9,
               marker='o', markersize=4, markevery=5)
    
    # Plot baseline (No DA) for reference with dashed line
    ax.plot(step_idxs, all_results[0]['diffs_noda_real_mse'], 
           color='black', linewidth=2, linestyle='--',
           label='No DA (Baseline)', alpha=0.8)
    
    ax.set_xlabel('Time Step', fontsize=14)
    ax.set_ylabel('MSE', fontsize=14)
    ax.set_title(f'{model_display_name}', 
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='upper right')
    
    # Add observation time markers
    for i, x in enumerate(time_obs):
        ax.axvline(x=x+1, color="black", linestyle=":", alpha=0.7, linewidth=2)
        ax.text(x+1, ax.get_ylim()[1]*0.9, f'Obs {i+1}', 
               rotation=90, ha='right', va='top', fontsize=10)
    
    # Enhance aesthetics
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = f'../../../../results/{model_name}/DA/cyl_diffnoise_performance.png'
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Single model noise level comparison plot saved to: {plot_path}")


def plot_snr_vs_performance(all_results, model_name="CAE_DMD", model_display_name="CAE DMD"):
    """Plot Signal-to-Noise Ratio vs. Performance metrics"""
    
    # Extract data
    snrs = []
    final_mse_da = []
    final_mse_noda = []
    improvements = []
    noise_stds = []
    
    for result in all_results:
        snr = result['snr']
        if snr != float('inf'):  # Skip no-noise case for SNR plot
            snrs.append(snr)
            final_mse_da.append(result['diffs_da_real_mse'][-1])
            final_mse_noda.append(result['diffs_noda_real_mse'][-1])
            improvement = ((result['diffs_noda_real_mse'][-1] - result['diffs_da_real_mse'][-1]) / 
                          result['diffs_noda_real_mse'][-1] * 100)
            improvements.append(improvement)
            noise_stds.append(result['noise_std'])
    
    if not snrs:  # If all results are no-noise
        print("No SNR plot generated: all experiments are noise-free")
        return
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: SNR vs Final MSE
    ax1.scatter(snrs, final_mse_da, color='blue', s=100, alpha=0.7, label='4D-Var DA')
    ax1.scatter(snrs, final_mse_noda, color='red', s=100, alpha=0.7, label='No DA (Baseline)')
    
    ax1.set_xlabel('Signal-to-Noise Ratio', fontsize=12)
    ax1.set_ylabel('Final MSE', fontsize=12)
    ax1.set_title(f'{model_display_name} - SNR vs Final MSE', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add annotations for noise levels
    for i, (snr, mse, noise_std) in enumerate(zip(snrs, final_mse_da, noise_stds)):
        ax1.annotate(f'σ={noise_std:.3f}', (snr, mse), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8, alpha=0.7)
    
    # Plot 2: SNR vs Improvement
    ax2.scatter(snrs, improvements, color='green', s=100, alpha=0.7)
    
    ax2.set_xlabel('Signal-to-Noise Ratio', fontsize=12)
    ax2.set_ylabel('Improvement (%)', fontsize=12)
    ax2.set_title(f'{model_display_name} - SNR vs DA Improvement', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Add annotations for noise levels
    for i, (snr, improvement, noise_std) in enumerate(zip(snrs, improvements, noise_stds)):
        ax2.annotate(f'σ={noise_std:.3f}', (snr, improvement), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = f'../../../../results/{model_name}/DA/cyl_snr_vs_performance.png'
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SNR vs Performance plot saved to: {plot_path}")


def plot_noise_impact_summary(all_results, model_name="CAE_DMD", model_display_name="CAE DMD"):
    """Plot comprehensive summary of noise impact on DA performance"""
    
    # Extract data
    noise_stds = [result['noise_std'] for result in all_results]
    snrs = [result['snr'] for result in all_results]
    final_mse_da = [result['diffs_da_real_mse'][-1] for result in all_results]
    final_mse_noda = [result['diffs_noda_real_mse'][-1] for result in all_results]
    improvements = [((noda - da) / noda * 100) for da, noda in zip(final_mse_da, final_mse_noda)]
    iterations = [result['da_metrics']['num_iterations'] for result in all_results]
    final_costs = [result['da_metrics']['final_cost'] for result in all_results]
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: Noise std vs Final MSE
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(noise_stds, final_mse_da, 'bo-', linewidth=2, markersize=8, label='4D-Var DA')
    ax1.plot(noise_stds, final_mse_noda, 'ro-', linewidth=2, markersize=8, label='No DA')
    ax1.set_xlabel('Noise Standard Deviation (σ)')
    ax1.set_ylabel('Final MSE')
    ax1.set_title('Noise Level vs Final MSE')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Noise std vs Improvement
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(noise_stds, improvements, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Noise Standard Deviation (σ)')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Noise Level vs DA Improvement')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Plot 3: Noise std vs Iterations
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(noise_stds, iterations, 'mo-', linewidth=2, markersize=8)
    ax3.set_xlabel('Noise Standard Deviation (σ)')
    ax3.set_ylabel('DA Iterations')
    ax3.set_title('Noise Level vs Convergence')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: SNR vs Final MSE (log-log, excluding infinite SNR)
    ax4 = fig.add_subplot(gs[1, 0])
    finite_snr_mask = [snr != float('inf') for snr in snrs]
    finite_snrs = [snr for snr, mask in zip(snrs, finite_snr_mask) if mask]
    finite_mse_da = [mse for mse, mask in zip(final_mse_da, finite_snr_mask) if mask]
    finite_mse_noda = [mse for mse, mask in zip(final_mse_noda, finite_snr_mask) if mask]
    
    if finite_snrs:
        ax4.loglog(finite_snrs, finite_mse_da, 'bo-', linewidth=2, markersize=8, label='4D-Var DA')
        ax4.loglog(finite_snrs, finite_mse_noda, 'ro-', linewidth=2, markersize=8, label='No DA')
        ax4.set_xlabel('Signal-to-Noise Ratio')
        ax4.set_ylabel('Final MSE')
        ax4.set_title('SNR vs Final MSE (Log-Log)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No finite SNR data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('SNR vs Final MSE (No Data)')
    
    # Plot 5: Final Cost vs Noise
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(noise_stds, final_costs, 'co-', linewidth=2, markersize=8)
    ax5.set_xlabel('Noise Standard Deviation (σ)')
    ax5.set_ylabel('Final Cost Function Value')
    ax5.set_title('Noise Level vs Final Cost')
    ax5.set_yscale('log')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Performance degradation
    ax6 = fig.add_subplot(gs[1, 2])
    if noise_stds[0] == 0:  # If first experiment is noise-free
        baseline_improvement = improvements[0]
        degradation = [baseline_improvement - imp for imp in improvements]
        ax6.plot(noise_stds[1:], degradation[1:], 'ro-', linewidth=2, markersize=8)
        ax6.set_xlabel('Noise Standard Deviation (σ)')
        ax6.set_ylabel('Performance Degradation (%)')
        ax6.set_title('DA Performance Degradation')
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'No noise-free baseline', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Performance Degradation (No Baseline)')
    
    # Plot 7: Time series comparison for different noise levels
    ax7 = fig.add_subplot(gs[2, :])
    
    start_da_end_idxs = all_results[0]['start_da_end_idxs']
    step_idxs = list(range(start_da_end_idxs[0] + 1, start_da_end_idxs[-1] + 2))
    time_obs = all_results[0]['time_obs']
    
    # Generate colors
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(all_results)))
    
    for i, result in enumerate(all_results):
        noise_std = result['noise_std']
        snr = result['snr']
        color = colors[i]
        
        if snr == float('inf'):
            label = f'σ = {noise_std:.4f} (No Noise)'
        else:
            label = f'σ = {noise_std:.4f} (SNR: {snr:.1f})'
        
        ax7.plot(step_idxs, result['diffs_da_real_mse'], 
                color=color, linewidth=2, label=label, alpha=0.8)
    
    # Add baseline
    ax7.plot(step_idxs, all_results[0]['diffs_noda_real_mse'], 
            color='black', linewidth=2, linestyle='--',
            label='No DA (Baseline)', alpha=0.8)
    
    ax7.set_xlabel('Time Step')
    ax7.set_ylabel('MSE')
    ax7.set_title('MSE Evolution Over Time for Different Noise Levels')
    ax7.grid(True, alpha=0.3)
    ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add observation time markers
    for x in time_obs:
        ax7.axvline(x=x+1, color="red", linestyle=":", alpha=0.6, linewidth=1)
    
    plt.suptitle(f'{model_display_name} - Comprehensive Noise Impact Analysis', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Save plot
    plot_path = f'../../../../results/{model_name}/DA/cyl_noise_impact_summary.png'
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comprehensive noise impact summary plot saved to: {plot_path}")


def run_noise_level_experiments(
    noise_levels=[0.0, 0.005, 0.01, 0.02, 0.05, 0.1],  # Different noise standard deviations
    obs_ratio=0.05,  # Fixed observation ratio (~5%)
    model_name="CAE_DMD",
    model_display_name="CAE DMD"
):
    """Run complete noise level experiments with fixed observation density"""
    
    print(f"\n{'='*80}")
    print(f"NOISE LEVEL DATA ASSIMILATION EXPERIMENTS")
    print(f"Model: {model_display_name}")
    print(f"Fixed Observation Ratio: {obs_ratio:.1%}")
    print(f"Noise Levels (σ): {noise_levels}")
    print(f"{'='*80}")
    
    # Set seed and device
    set_seed(42)
    device = set_device()
    print(f"Using device: {device}")
    
    # Load forward model (global)
    global forward_model
    forward_model = CYLINDER_C_FORWARD().to(device)
    forward_model.load_state_dict(torch.load(f'../../../../results/{model_name}/Cylinder/cyl_model_weights/forward_model.pt', 
                                            weights_only=True, map_location=device))
    forward_model.C_forward = torch.load(f'../../../../results/{model_name}/Cylinder/cyl_model_weights/C_forward.pt', 
                                       weights_only=True, map_location=device).to(device)
    forward_model.eval()
    print("Forward model loaded successfully")
    
    # Run experiments for each noise level
    all_results = []
    
    for noise_std in noise_levels:
        try:
            print(f"\n{'='*50}")
            print(f"Noise Level: σ = {noise_std:.4f}")
            print(f"{'='*50}")
            
            result = run_single_noise_da_experiment(
                noise_std=noise_std,
                obs_ratio=obs_ratio,
                model_name=model_name,
                device=device
            )
            all_results.append(result)
            
            print(f"\nExperiment completed:")
            print(f"  Noise std: {noise_std:.4f}")
            print(f"  SNR: {result['snr']:.2f}" if result['snr'] != float('inf') else "  SNR: ∞ (no noise)")
            print(f"  Final cost: {result['da_metrics']['final_cost']:.6f}")
            print(f"  Iterations: {result['da_metrics']['num_iterations']}")
            print(f"  Avg time per iteration: {result['da_metrics']['avg_time_per_iteration']:.6f}s")
            
        except Exception as e:
            print(f"Error in experiment with noise level {noise_std:.4f}: {str(e)}")
            continue
    
    if not all_results:
        print("No experiments completed successfully!")
        return
    
    # Save all results
    results_path = f'../../../../results/{model_name}/DA/cyl_diffnoise_results.pkl'
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nAll results saved to: {results_path}")
    
    # Generate plots
    print("\nGenerating comparison plots...")
    
    # Plot 1: Side-by-side comparison (DA vs No DA)
    plot_noise_level_comparison(all_results, model_name, model_display_name)
    
    # Plot 2: Single model focus (recommended)
    plot_single_model_noise_comparison(all_results, model_name, model_display_name)
    
    # Plot 3: SNR vs Performance
    plot_snr_vs_performance(all_results, model_name, model_display_name)
    
    # Plot 4: Comprehensive summary
    plot_noise_impact_summary(all_results, model_name, model_display_name)
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    for result in all_results:
        noise_std = result['noise_std']
        snr = result['snr']
        start_da_end_idxs = all_results[0]['start_da_end_idxs']
        da_time_idx = start_da_end_idxs[1] - start_da_end_idxs[0]
        final_mse_da = result['diffs_da_real_mse'][da_time_idx]
        final_mse_noda = result['diffs_noda_real_mse'][da_time_idx]
        improvement = ((final_mse_noda - final_mse_da) / final_mse_noda * 100)
        
        print(f"Noise Level: σ = {noise_std:.4f}")
        if snr != float('inf'):
            print(f"  SNR: {snr:.2f}")
        else:
            print(f"  SNR: ∞ (no noise)")
        print(f"  Final MSE (DA): {final_mse_da:.6f}")
        print(f"  Final MSE (No DA): {final_mse_noda:.6f}")
        print(f"  Improvement: {improvement:.2f}%")
        print(f"  DA Iterations: {result['da_metrics']['num_iterations']}")
        print(f"  Final Cost: {result['da_metrics']['final_cost']:.6f}")
        print()
    
    # Analyze trends
    if len(all_results) > 1:
        print("TREND ANALYSIS:")
        noise_stds = [r['noise_std'] for r in all_results]
        improvements = [((r['diffs_noda_real_mse'][-1] - r['diffs_da_real_mse'][-1]) / 
                        r['diffs_noda_real_mse'][-1] * 100) for r in all_results]
        
        # Find optimal noise level
        max_improvement_idx = np.argmax(improvements)
        optimal_noise = noise_stds[max_improvement_idx]
        max_improvement = improvements[max_improvement_idx]
        
        print(f"  Best performance at σ = {optimal_noise:.4f} with {max_improvement:.2f}% improvement")
        
        # Performance degradation analysis
        if noise_stds[0] == 0:  # If we have noise-free baseline
            baseline_improvement = improvements[0]
            print(f"  Noise-free baseline improvement: {baseline_improvement:.2f}%")
            
            # Find where performance drops below 90% of baseline
            threshold = 0.9 * baseline_improvement
            degraded_indices = [i for i, imp in enumerate(improvements) if imp < threshold]
            if degraded_indices:
                critical_noise = noise_stds[degraded_indices[0]]
                print(f"  Performance drops below 90% of baseline at σ = {critical_noise:.4f}")
    
    print("Noise level experiments completed successfully!")
    return all_results


if __name__ == "__main__":
    # Run experiments with different noise levels
    # Fixed observation ratio around 5% (4.5%-5.5% range)
    results = run_noise_level_experiments(
        noise_levels=[0.0, 0.005, 0.01, 0.05, 0.1],  # σ values
        obs_ratio=0.05,  # Fixed 5% observation ratio
        model_name="CAE_Koopman",
        model_display_name="Koopman ROM"
    )