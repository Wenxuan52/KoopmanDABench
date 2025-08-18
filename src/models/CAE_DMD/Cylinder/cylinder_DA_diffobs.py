#!/usr/bin/env python
# coding: utf-8

"""
Sparse Observation Data Assimilation Experiment for CAE+DMD model
Testing different observation densities with gradient blue visualization
Fixed to handle variable observation dimensions properly
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


def set_device():
    """Set device to GPU if available, otherwise CPU"""
    if torch.cuda.device_count() == 0:
        return "cpu"
    torch.set_float32_matmul_precision("high")
    return "cuda"


class UnifiedRandomSparseObservationHandler:
    """
    Unified handler for random sparse observations with variable density
    Uses fixed maximum observation count with variable valid observations per time step
    """
    def __init__(self, target_obs_ratio: float, ratio_range: float = 0.005, seed: int = 42):
        self.target_obs_ratio = target_obs_ratio
        self.ratio_range = ratio_range
        self.seed = seed
        self.time_masks = {}
        self.max_obs_count = 0
        self.total_pixels = 0
        self.fixed_positions = None
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
    
    def generate_unified_observations(self, image_shape: tuple, time_steps: int) -> int:
        """
        Generate unified observation positions for all time steps
        Similar to cylinder_DA.py but with random observation counts per time step
        """
        if len(image_shape) == 3:
            C, H, W = image_shape
        else:
            raise ValueError(f"Expected 3D image shape (C, H, W), got {image_shape}")
        
        self.total_pixels = C * H * W
        
        # Calculate the maximum possible observations needed (use upper bound of ratio range)
        max_ratio = min(self.target_obs_ratio + self.ratio_range, 1.0)
        self.max_obs_count = int(self.total_pixels * max_ratio)
        
        # Generate fixed observation positions (like in cylinder_DA.py)
        self.fixed_positions = torch.randperm(self.total_pixels)[:self.max_obs_count]
        print(f"Fixed observation positions generated: {self.max_obs_count} positions")
        
        # Generate random observation counts for each time step
        for t in range(time_steps):
            # Random ratio within the specified range
            min_ratio = max(self.target_obs_ratio - self.ratio_range, 0.001)
            max_ratio_curr = min(self.target_obs_ratio + self.ratio_range, 1.0)
            actual_ratio = np.random.uniform(min_ratio, max_ratio_curr)
            
            # Number of valid observations for this time step
            num_valid = int(self.total_pixels * actual_ratio)
            num_valid = max(1, num_valid)  # Ensure at least 1 observation
            num_valid = min(num_valid, self.max_obs_count)  # Don't exceed max
            
            # Random selection of which fixed positions are valid
            valid_indices = torch.randperm(self.max_obs_count)[:num_valid]
            
            self.time_masks[t] = {
                'num_valid': num_valid,
                'valid_indices': valid_indices,
                'actual_ratio': actual_ratio
            }
            
            print(f"Time step {t}: {num_valid}/{self.max_obs_count} observations ({actual_ratio:.3%} ratio, target: {self.target_obs_ratio:.1%})")
        
        return self.max_obs_count
    
    def apply_unified_observation(self, full_image: torch.Tensor, time_step: int) -> torch.Tensor:
        """
        Apply observation mask to full image for specific time step
        Returns a vector of size max_obs_count with zeros for invalid observations
        """
        if time_step not in self.time_masks:
            raise ValueError(f"Time step {time_step} not found in masks")
        
        mask_info = self.time_masks[time_step]
        
        # Get observations at fixed positions
        flat_image = full_image.flatten()
        fixed_obs = flat_image[self.fixed_positions.to(full_image.device)]
        
        # Create observation vector with zeros for invalid observations
        obs_vector = torch.zeros(self.max_obs_count, device=full_image.device)
        valid_indices = mask_info['valid_indices'].to(full_image.device)
        obs_vector[valid_indices] = fixed_obs[valid_indices]
        
        return obs_vector
    
    def create_block_R_matrix(self, base_variance=1e-3):
        """
        Create unified observation covariance matrix
        Same size for all time steps (max_obs_count x max_obs_count)
        """
        R = torch.eye(self.max_obs_count) * base_variance
        return R
    
    def get_actual_ratio(self, time_step: int):
        """Get actual observation ratio for specific time step"""
        if time_step not in self.time_masks:
            return 0.0
        return self.time_masks[time_step]['actual_ratio']


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


def H_unified(z_t):
    """
    Unified observation operator for sparse observations
    Uses the current time index to determine which observations are valid
    """
    global _global_time_idx, _global_obs_handler, _global_image_shape
    
    x_reconstructed = decoder(z_t)
    if x_reconstructed.dim() == 1:
        x_reconstructed = x_reconstructed.view(_global_image_shape)
    
    sparse_obs = _global_obs_handler.apply_unified_observation(
        x_reconstructed, _global_time_idx
    )
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


def run_single_da_experiment(
    target_obs_ratio: float,
    ratio_range: float = 0.005,
    model_name: str = "CAE_DMD",
    start_da_end_idxs: tuple = (700, 800, 900),
    time_obs: list = None,
    gaps: list = None,
    early_stop_config: tuple = (100, 1e-2),
    device: str = "cuda"
):
    """Run single data assimilation experiment with random observation ratios"""
    
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
    print(f"Running DA experiment with target observation ratio: {target_obs_ratio:.1%}")
    print(f"Random range: ±{ratio_range:.1%}")
    print(f"{'='*60}")
    
    # Initialize unified observation handler
    obs_handler = UnifiedRandomSparseObservationHandler(
        target_obs_ratio=target_obs_ratio, 
        ratio_range=ratio_range, 
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
    print(f"Full observation data shape: {full_y_data.shape}")
    
    # Generate unified sparse observations
    sample_image_shape = full_y_data[0].shape
    max_obs_count = obs_handler.generate_unified_observations(sample_image_shape, len(time_obs))
    
    # Create sparse observation data using unified approach
    sparse_y_data = []
    actual_ratios = []
    for i, full_img in enumerate(full_y_data):
        sparse_obs = obs_handler.apply_unified_observation(full_img, i)
        sparse_y_data.append(sparse_obs)
        actual_ratios.append(obs_handler.get_actual_ratio(i))
    
    # Stack sparse observations - now they all have the same dimension
    sparse_y_data = torch.stack(sparse_y_data).to(device)
    print(f"Sparse observation shape: {sparse_y_data.shape}")
    
    print(f"Actual observation ratios: {[f'{ratio:.3%}' for ratio in actual_ratios]}")
    print(f"Average ratio: {np.mean(actual_ratios):.3%} (target: {target_obs_ratio:.1%})")
    
    # Set up DA matrices
    latent_dim = _global_cae_model.hidden_dim
    B = torch.eye(latent_dim, device=device)
    R = obs_handler.create_block_R_matrix(base_variance=1e-3).to(device)
    
    print(f"Background covariance B shape: {B.shape}")
    print(f"Observation covariance R shape: {R.shape}")
    print(f"R matrix condition number: {torch.linalg.cond(R):.2e}")
    
    # Configure 4D-Var
    case_to_run = (
        torchda.CaseBuilder()
        .set_observation_time_steps(time_obs)
        .set_gaps(gaps)
        .set_forward_model(dmd_wrapper)
        .set_observation_model(H_unified)
        .set_background_covariance_matrix(B)
        .set_observation_covariance_matrix(R)
        .set_observations(sparse_y_data)
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
    
    current_state = cyl_val_dataset.normalize(groundtruth[start_da_end_idxs[0]]).to(device)
    z_current = encoder(current_state)
    
    da_metrics = {
        'final_cost': None,
        'num_iterations': None,
        'avg_time_per_iteration': None,
        'total_da_time': None
    }
    
    for i in range(start_da_end_idxs[0], start_da_end_idxs[-1] + 1):
        print(f"Processing step {i}")
        
        if i == start_da_end_idxs[1]:
            # Perform data assimilation at this time step
            update_observation_time_index(0)
            
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
            z_current = z_assimilated
        else:
            outs_4d_da.append(z_current)
        
        # Propagate forward in latent space for next iteration (if not last step)
        if i < start_da_end_idxs[-1]:
            z_current = latent_forward(z_current)
        
        print("=" * 50)
    
    total_time = perf_counter() - start_time
    print(f"4D-Var time: {total_time:.2f}s")
    
    # Run baseline (no DA)
    print("Running baseline (no DA)...")
    outs_no_4d_da = []
    
    with torch.no_grad():
        current_state = cyl_val_dataset.normalize(groundtruth[start_da_end_idxs[0]]).to(device)
        z_current = encoder(current_state)
        
        for i in range(start_da_end_idxs[0], start_da_end_idxs[-1] + 1):
            print(f"Step {i}")
            outs_no_4d_da.append(z_current)
            
            # Propagate forward in latent space for next iteration (if not last step)
            if i < start_da_end_idxs[-1]:
                z_current = latent_forward(z_current)
            
            print("=" * 30)
    
    print(f"Baseline time: {perf_counter() - start_time:.2f}s")
    
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
        'target_obs_ratio': target_obs_ratio,
        'actual_obs_ratios': actual_ratios,
        'avg_actual_ratio': np.mean(actual_ratios),
        'ratio_range': ratio_range,
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


def plot_sparse_observation_comparison(all_results, model_name="CAE_DMD", model_display_name="CAE DMD"):
    """Plot comparison of different observation densities with gradient blue colors"""
    
    # Create gradient blue colormap
    target_ratios = [result['target_obs_ratio'] for result in all_results]
    n_ratios = len(target_ratios)
    
    # Generate gradient blue colors from light to dark
    blues = plt.cm.Blues(np.linspace(0.3, 1.0, n_ratios))
    
    # Get step indices from first result
    start_da_end_idxs = all_results[0]['start_da_end_idxs']
    step_idxs = list(range(start_da_end_idxs[0] + 1, start_da_end_idxs[-1] + 2))
    time_obs = all_results[0]['time_obs']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot MSE comparison for DA
    for i, result in enumerate(all_results):
        target_ratio = result['target_obs_ratio']
        avg_actual_ratio = result['avg_actual_ratio']
        color = blues[i]
        
        ax1.plot(step_idxs, result['diffs_da_real_mse'], 
                color=color, linewidth=2, 
                label=f'{target_ratio:.1%} target (avg: {avg_actual_ratio:.2%})', alpha=0.8)
    
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('MSE', fontsize=12)
    ax1.set_title(f'{model_display_name} - 4D-Var', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Add observation time markers
    for x in time_obs:
        ax1.axvline(x=x+1, color="red", linestyle="--", alpha=0.6, linewidth=1)
    
    # Plot MSE comparison for No DA (baseline)
    for i, result in enumerate(all_results):
        target_ratio = result['target_obs_ratio']
        avg_actual_ratio = result['avg_actual_ratio']
        color = blues[i]
        
        ax2.plot(step_idxs, result['diffs_noda_real_mse'], 
                color=color, linewidth=2, 
                label=f'{target_ratio:.1%} target (avg: {avg_actual_ratio:.2%})', alpha=0.8)
    
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('MSE', fontsize=12)
    ax2.set_title(f'{model_display_name} - No DA', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Add observation time markers
    for x in time_obs:
        ax2.axvline(x=x+1, color="red", linestyle="--", alpha=0.6, linewidth=1)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = f'../../../../results/{model_name}/DA/cyl_diffobs_comparison.png'
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Random sparse observation comparison plot saved to: {plot_path}")


def plot_single_model_sparse_comparison(all_results, model_name="CAE_DMD", model_display_name="CAE DMD"):
    """Plot single model with different observation densities (recommended visualization)"""
    
    # Create gradient blue colormap
    target_ratios = [result['target_obs_ratio'] for result in all_results]
    n_ratios = len(target_ratios)
    
    # Generate gradient blue colors from light to dark
    blues = plt.cm.Blues(np.linspace(0.4, 1.0, n_ratios))
    
    # Get step indices from first result
    start_da_end_idxs = all_results[0]['start_da_end_idxs']
    step_idxs = list(range(start_da_end_idxs[0] + 1, start_da_end_idxs[-1] + 2))
    time_obs = all_results[0]['time_obs']
    
    # Create single figure focusing on DA performance
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot MSE for different observation densities
    for i, result in enumerate(all_results):
        target_ratio = result['target_obs_ratio']
        avg_actual_ratio = result['avg_actual_ratio']
        color = blues[i]
        
        ax.plot(step_idxs, result['diffs_da_real_mse'], 
               color=color, linewidth=3, 
               label=f'{target_ratio:.1%} Target (Avg: {avg_actual_ratio:.2%})', alpha=0.9,
               marker='o', markersize=4, markevery=5)
    
    # Plot baseline (No DA) for reference with dashed line
    ax.plot(step_idxs, all_results[0]['diffs_noda_real_mse'], 
           color='red', linewidth=2, linestyle='--',
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
    plot_path = f'../../../../results/{model_name}/DA/cyl_diffobs_density_comparison.png'
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Single model random sparse observation comparison plot saved to: {plot_path}")


def run_sparse_observation_experiments(
    target_observation_ratios=[0.1, 0.05, 0.01, 0.005],  # 10%, 5%, 1%, 0.5%
    ratio_ranges=[0.005, 0.005, 0.001, 0.001],  # ±0.5%, ±0.5%, ±0.1%, ±0.1%
    model_name="CAE_DMD",
    model_display_name="CAE DMD"
):
    """Run complete sparse observation experiments with random densities"""
    
    # Set default ratio ranges if not provided
    if len(ratio_ranges) == 1:
        ratio_ranges = ratio_ranges * len(target_observation_ratios)
    elif len(ratio_ranges) != len(target_observation_ratios):
        ratio_ranges = [0.005] * len(target_observation_ratios)
    
    print(f"\n{'='*80}")
    print(f"RANDOM SPARSE OBSERVATION DATA ASSIMILATION EXPERIMENTS")
    print(f"Model: {model_display_name}")
    print(f"Target Ratios: {[f'{ratio:.1%}' for ratio in target_observation_ratios]}")
    print(f"Random Ranges: {[f'±{range_val:.1%}' for range_val in ratio_ranges]}")
    print(f"{'='*80}")
    
    # Set seed and device
    set_seed(42)
    device = set_device()
    print(f"Using device: {device}")
    
    # Load CAE+DMD models (global)
    global _global_cae_model, _global_dmd_model
    
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
    
    # Run experiments for each target observation ratio
    all_results = []
    
    for target_ratio, ratio_range in zip(target_observation_ratios, ratio_ranges):
        try:
            print(f"\n{'='*50}")
            print(f"Target: {target_ratio:.1%} (±{ratio_range:.1%})")
            print(f"Range: {target_ratio-ratio_range:.1%} - {target_ratio+ratio_range:.1%}")
            print(f"{'='*50}")
            
            result = run_single_da_experiment(
                target_obs_ratio=target_ratio,
                ratio_range=ratio_range,
                model_name=model_name,
                device=device
            )
            all_results.append(result)
            
            print(f"\nExperiment completed:")
            print(f"  Target ratio: {target_ratio:.1%}")
            print(f"  Actual ratios: {[f'{r:.3%}' for r in result['actual_obs_ratios']]}")
            print(f"  Average actual ratio: {result['avg_actual_ratio']:.3%}")
            print(f"  Final cost: {result['da_metrics']['final_cost']:.6f}")
            print(f"  Iterations: {result['da_metrics']['num_iterations']}")
            print(f"  Avg time per iteration: {result['da_metrics']['avg_time_per_iteration']:.6f}s")
            
        except Exception as e:
            print(f"Error in experiment with target {target_ratio:.1%} observations: {str(e)}")
            continue
    
    if not all_results:
        print("No experiments completed successfully!")
        return
    
    # Save all results
    results_path = f'../../../../results/{model_name}/DA/cyl_diffobs_results.pkl'
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nAll results saved to: {results_path}")
    
    # Generate plots
    print("\nGenerating comparison plots...")
    
    # Plot 1: Side-by-side comparison (DA vs No DA)
    plot_sparse_observation_comparison(all_results, model_name, model_display_name)
    
    # Plot 2: Single model focus (recommended)
    plot_single_model_sparse_comparison(all_results, model_name, model_display_name)
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    for result in all_results:
        target_ratio = result['target_obs_ratio']
        avg_actual_ratio = result['avg_actual_ratio']
        start_da_end_idxs = all_results[0]['start_da_end_idxs']
        da_time_idx = start_da_end_idxs[1] - start_da_end_idxs[0]
        final_mse_da = result['diffs_da_real_mse'][da_time_idx]
        final_mse_noda = result['diffs_noda_real_mse'][da_time_idx]
        improvement = ((final_mse_noda - final_mse_da) / final_mse_noda * 100)
        
        print(f"Target Ratio: {target_ratio:.1%}")
        print(f"  Actual ratios: {[f'{r:.3%}' for r in result['actual_obs_ratios']]}")
        print(f"  Average actual: {avg_actual_ratio:.3%}")
        print(f"  Final MSE (DA): {final_mse_da:.6f}")
        print(f"  Final MSE (No DA): {final_mse_noda:.6f}")
        print(f"  Improvement: {improvement:.2f}%")
        print(f"  DA Iterations: {result['da_metrics']['num_iterations']}")
        print()
    
    print("Random sparse observation experiments completed successfully!")
    return all_results


if __name__ == "__main__":
    # Run experiments with random observation densities
    # Each target ratio has its corresponding range for randomness
    results = run_sparse_observation_experiments(
        target_observation_ratios=[0.1, 0.05, 0.01, 0.005],  # 10%, 5%, 1%, 0.5%
        ratio_ranges=[0.005, 0.005, 0.001, 0.001],  # ±0.5%, ±0.5%, ±0.1%, ±0.1%
        model_name="CAE_DMD",
        model_display_name="DMD ROM"
    )