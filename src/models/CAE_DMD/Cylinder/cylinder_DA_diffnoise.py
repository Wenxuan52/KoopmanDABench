#!/usr/bin/env python
# coding: utf-8

"""
Noise Level Data Assimilation Experiment for CAE+DMD model
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

from src.models.CAE_Koopman.Cylinder.cylinder_model import CYLINDER_C_FORWARD
from src.models.DMD.base import TorchDMD
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


# Global variables for observation handler and models
_global_obs_handler = None
_global_time_idx = 0
_global_cae_model = None
_global_dmd_model = None
_global_image_shape = None


def update_observation_time_index(time_idx: int):
    """Update global time index for observations"""
    global _global_time_idx
    _global_time_idx = time_idx


def encoder(x_t):
    """Encode physical state to CAE+DMD latent space"""
    global _global_cae_model
    if x_t.dim() == 1:
        x_t = x_t.view(1, *_global_image_shape)
    elif x_t.dim() == 3:
        x_t = x_t.unsqueeze(0)
    
    z_t = _global_cae_model.K_S(x_t)
    return z_t.squeeze()


def decoder(z_t):
    """Decode from CAE+DMD latent space to physical state"""
    global _global_cae_model
    if z_t.dim() == 1:
        z_t = z_t.unsqueeze(0)
    
    x_t = _global_cae_model.K_S_preimage(z_t)
    return x_t.squeeze()


def latent_forward(z_t):
    """Forward propagation in latent space using DMD"""
    global _global_dmd_model
    if z_t.dim() == 1:
        z_t = z_t.unsqueeze(0)
    
    # DMD expects [hidden_dim, 1] format
    z_tp = _global_dmd_model.predict(z_t.T).T
    return z_tp.squeeze()


def H_sparse_noisy(z_t):
    """Observation operator for noisy sparse observations"""
    global _global_obs_handler, _global_image_shape
    
    x_reconstructed = decoder(z_t)
    if x_reconstructed.dim() == 1:
        x_reconstructed = x_reconstructed.view(_global_image_shape)
    
    sparse_obs = _global_obs_handler.apply_observation(x_reconstructed, add_noise=False)  # Noise already added to observations
    return sparse_obs.unsqueeze(0)


def dmd_wrapper(z_t, time_fw=None, *args):
    """Wrapper for CAE+DMD forward model"""
    if z_t.dim() > 1:
        z_t = z_t.squeeze()
    
    if time_fw is None:
        z_tp = latent_forward(z_t)
        return z_tp
    else:
        num_steps = int(time_fw.shape[0])
        latent_dim = z_t.shape[0]
        z_tp = torch.empty((num_steps, latent_dim), device=z_t.device)
        
        z_current = z_t
        
        for i in range(num_steps):
            z_tp[i] = z_current
            
            if i < num_steps - 1:
                z_current = latent_forward(z_current)
        
        return z_tp.unsqueeze(1)


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
    
    global _global_cae_model, _global_dmd_model, _global_obs_handler, _global_image_shape
    
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
    _global_image_shape = (2, 64, 64)
    
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
    
    denorm = cyl_val_dataset.denormalizer()
    
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
    latent_dim = _global_cae_model.hidden_dim
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
        .set_max_iterations(500)
        .set_early_stop(early_stop_config)
        .set_algorithm(torchda.Algorithms.Var4D)
        .set_device(torchda.Device.GPU if device == "cuda" else torchda.Device.CPU)
        .set_output_sequence_length(1)
    )
    
    # Run 4D-Var assimilation
    print("Running 4D-Var data assimilation...")
    outs_4d_da = []
    start_time = perf_counter()
    
    # Initialize with encoding of the initial state
    current_state = cyl_val_dataset.normalize(groundtruth[start_da_end_idxs[0]]).to(device)
    z_current = encoder(current_state)
    
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
        print(f"Processing step {i}")
        
        if i == start_da_end_idxs[1]:
            # Perform data assimilation at this time step
            case_to_run.set_background_state(z_current)
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
            
            print(f"Final cost function: {final_cost}")
            print(f"Number of iterations: {num_iterations}")
            print(f"Average time per iteration: {avg_time_per_iteration:.6f}s")
            
            outs_4d_da.append(z_assimilated)
            
            # Continue propagation in latent space
            z_current = z_assimilated
        else:
            # Store current latent state
            outs_4d_da.append(z_current)
        
        # Propagate forward in latent space for next iteration (if not last step)
        if i < start_da_end_idxs[-1]:
            z_current = latent_forward(z_current)
        
        print("=" * 50)
    
    total_time = perf_counter() - start_time
    print(f"Total experiment time: {total_time:.2f}s")
    
    # Run baseline (no DA)
    print("Running baseline (no DA)...")
    outs_no_4d_da = []
    
    with torch.no_grad():
        # Initialize with encoding of the initial state
        current_state = cyl_val_dataset.normalize(groundtruth[start_da_end_idxs[0]]).to(device)
        z_current = encoder(current_state)
        
        for i in range(start_da_end_idxs[0], start_da_end_idxs[-1] + 1):
            print(f"Step {i}")
            
            # Store current latent state
            outs_no_4d_da.append(z_current)
            
            # Propagate forward in latent space for next iteration (if not last step)
            if i < start_da_end_idxs[-1]:
                z_current = latent_forward(z_current)
            
            print("=" * 30)
    
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
            da_img = decoder(da).view(2, 64, 64).cpu()
            noda_img = decoder(no_da).view(2, 64, 64).cpu()
            
            de_da_img = denorm(da_img)
            de_noda_img = denorm(noda_img)

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


def plot_single_model_noise_comparison(all_results, model_name="CAE_DMD", model_display_name="CAE DMD"):
    """Plot single model with different noise levels"""
    
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
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
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
    
    # Plot 6: Time series comparison for different noise levels
    ax6 = fig.add_subplot(gs[1, 2])
    
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
            label = f'σ = {noise_std:.4f}'
        
        ax6.plot(step_idxs, result['diffs_da_real_mse'], 
                color=color, linewidth=2, label=label, alpha=0.8)
    
    ax6.set_xlabel('Time Step')
    ax6.set_ylabel('MSE')
    ax6.set_title('MSE Evolution Over Time')
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=8)
    
    # Add observation time markers
    for x in time_obs:
        ax6.axvline(x=x+1, color="red", linestyle=":", alpha=0.6, linewidth=1)
    
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
    
    # Initialize globals
    global _global_cae_model, _global_dmd_model, _global_obs_handler, _global_image_shape
    
    # Load CAE model
    _global_cae_model = CYLINDER_C_FORWARD()
    _global_cae_model.load_state_dict(
        torch.load('../../../../results/CAE_Koopman/Cylinder/cyl_model_weights/forward_model.pt', 
                  weights_only=True, map_location=device)
    )
    _global_cae_model.to(device)
    _global_cae_model.eval()
    print("CAE model loaded")
    
    # Load DMD model
    _global_dmd_model = TorchDMD(svd_rank=93, device=device)
    _global_dmd_model.load_dmd('../../../../results/CAE_DMD/Cylinder/dmd_model.pth')
    print("DMD model loaded")
    
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
    
    # Plot 1: Single model focus (recommended)
    plot_single_model_noise_comparison(all_results, model_name, model_display_name)
    
    # Plot 2: Comprehensive summary
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
        model_name="CAE_DMD",
        model_display_name="DMD ROM"
    )