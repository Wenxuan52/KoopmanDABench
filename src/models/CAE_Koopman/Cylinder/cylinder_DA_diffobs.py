#!/usr/bin/env python
# coding: utf-8

"""
Sparse Observation Data Assimilation Experiment
Testing different observation densities with gradient blue visualization
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


class RandomSparseObservationHandler:
    """Handler for random sparse observations with variable density"""
    def __init__(self, target_obs_ratio: float, ratio_range: float = 0.005, seed: int = 42):
        self.target_obs_ratio = target_obs_ratio
        self.ratio_range = ratio_range
        self.seed = seed
        self.time_masks = {}
        self.max_obs_count = 0
        self.total_pixels = 0
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
    
    def generate_observation_positions(self, image_shape: tuple, time_steps: int) -> int:
        """Generate random observation positions for each time step"""
        if len(image_shape) == 3:
            C, H, W = image_shape
        else:
            raise ValueError(f"Expected 3D image shape (C, H, W), got {image_shape}")
        
        self.total_pixels = C * H * W
        
        # Calculate the maximum possible observations needed
        max_ratio = min(self.target_obs_ratio + self.ratio_range, 1.0)
        self.max_obs_count = int(self.total_pixels * max_ratio)
        
        # Generate random observation ratios and positions for each time step
        for t in range(time_steps):
            # Random ratio within the specified range
            min_ratio = max(self.target_obs_ratio - self.ratio_range, 0.001)
            max_ratio = min(self.target_obs_ratio + self.ratio_range, 1.0)
            actual_ratio = np.random.uniform(min_ratio, max_ratio)
            
            # Number of observations for this time step
            obs_count = int(self.total_pixels * actual_ratio)
            obs_count = max(1, obs_count)  # Ensure at least 1 observation
            
            # Random positions
            positions = torch.randperm(self.total_pixels)[:obs_count]
            
            self.time_masks[t] = {
                'positions': positions,
                'obs_count': obs_count,
                'actual_ratio': actual_ratio
            }
            
            print(f"Time step {t}: {obs_count} observations ({actual_ratio:.3%} ratio, target: {self.target_obs_ratio:.1%})")
        
        return self.max_obs_count
    
    def get_max_obs_count(self):
        """Get maximum observation count across all time steps"""
        if not self.time_masks:
            return 0
        return max(mask_info['obs_count'] for mask_info in self.time_masks.values())
    
    def apply_observation(self, full_image: torch.Tensor, time_step: int) -> torch.Tensor:
        """Apply observation mask to full image for specific time step"""
        if time_step not in self.time_masks:
            raise ValueError(f"Time step {time_step} not found in masks")
        
        mask_info = self.time_masks[time_step]
        flat_image = full_image.flatten()
        obs_vector = flat_image[mask_info['positions']]
        return obs_vector
    
    def create_R_matrix(self, time_step: int, base_variance=1e-3):
        """Create observation covariance matrix for specific time step"""
        if time_step not in self.time_masks:
            raise ValueError(f"Time step {time_step} not found in masks")
        
        obs_count = self.time_masks[time_step]['obs_count']
        R = torch.eye(obs_count) * base_variance
        return R
    
    def get_actual_ratio(self, time_step: int):
        """Get actual observation ratio for specific time step"""
        if time_step not in self.time_masks:
            return 0.0
        return self.time_masks[time_step]['actual_ratio']


# Global variables for observation handler
_global_obs_handler = None
_global_time_step = 0


def update_observation_time_step(time_step: int):
    """Update global time step for observations"""
    global _global_time_step
    _global_time_step = time_step


def H_sparse(x):
    """Observation operator for sparse observations"""
    global _global_obs_handler, _global_time_step, forward_model
    
    x_reconstructed = forward_model.K_S_preimage(x)
    sparse_obs = _global_obs_handler.apply_observation(x_reconstructed.squeeze(), _global_time_step)
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
    
    global forward_model, _global_obs_handler
    
    print(f"\n{'='*60}")
    print(f"Running DA experiment with target observation ratio: {target_obs_ratio:.1%}")
    print(f"Random range: ±{ratio_range:.1%}")
    print(f"{'='*60}")
    
    # Initialize observation handler with random ratios
    obs_handler = RandomSparseObservationHandler(
        target_obs_ratio=target_obs_ratio, 
        ratio_range=ratio_range, 
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
    
    # Generate random sparse observations for each time step
    sample_image_shape = full_y_data[0].shape
    max_obs_count = obs_handler.generate_observation_positions(sample_image_shape, len(time_obs))
    
    sparse_y_data = []
    actual_ratios = []
    for i, full_img in enumerate(full_y_data):
        sparse_obs = obs_handler.apply_observation(full_img, i)
        sparse_y_data.append(sparse_obs)
        actual_ratios.append(obs_handler.get_actual_ratio(i))
    
    print(f"Actual observation ratios: {[f'{ratio:.3%}' for ratio in actual_ratios]}")
    print(f"Average ratio: {np.mean(actual_ratios):.3%} (target: {target_obs_ratio:.1%})")
    
    # Set up DA matrices (using first time step dimensions)
    latent_dim = forward_model.C_forward.shape[0]
    B = torch.eye(latent_dim, device=device)
    R = obs_handler.create_R_matrix(0, base_variance=1e-3).to(device)  # Use first time step
    
    sparse_y_data_tensor = torch.stack(sparse_y_data).to(device)
    
    # Configure 4D-Var
    case_to_run = (
        torchda.CaseBuilder()
        .set_observation_time_steps(time_obs)
        .set_gaps(gaps)
        .set_forward_model(dmd_wrapper)
        .set_observation_model(H_sparse)
        .set_background_covariance_matrix(B)
        .set_observation_covariance_matrix(R)
        .set_observations(sparse_y_data_tensor)
        .set_optimizer_cls(torch.optim.Adam)
        .set_optimizer_args({"lr": 0.005})
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
        'total_da_time': None
    }
    
    for i in range(start_da_end_idxs[0], start_da_end_idxs[-1] + 1):
        z_current = forward_model.K_S(current_state)
        
        if i == start_da_end_idxs[1]:
            # Update observation time step for the observation operator
            update_observation_time_step(0)
            
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
    ax1.set_title(f'{model_display_name}', fontsize=14, fontweight='bold')
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
    ax2.set_title(f'{model_display_name}', fontsize=14, fontweight='bold')
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
    
    # Load forward model (global)
    global forward_model
    forward_model = CYLINDER_C_FORWARD().to(device)
    forward_model.load_state_dict(torch.load(f'../../../../results/{model_name}/Cylinder/cyl_model_weights/forward_model.pt', 
                                            weights_only=True, map_location=device))
    forward_model.C_forward = torch.load(f'../../../../results/{model_name}/Cylinder/cyl_model_weights/C_forward.pt', 
                                       weights_only=True, map_location=device).to(device)
    forward_model.eval()
    print("Forward model loaded successfully")
    
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
        model_name="CAE_Koopman",
        model_display_name="Koopman ROM"
    )