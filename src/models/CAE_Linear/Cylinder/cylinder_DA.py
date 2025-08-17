#!/usr/bin/env python
# coding: utf-8

"""
Data Assimilation for DMD cylinder model
"""

import random
import os
import sys
import pickle
from time import perf_counter

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Set matplotlib backend and config directory
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_config'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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


class UnifiedDynamicSparseObservationHandler:
    """Handler for dynamic sparse observations"""
    def __init__(self, max_obs_ratio: float = 0.15, min_obs_ratio: float = 0.05, seed: int = 42):
        self.max_obs_ratio = max_obs_ratio
        self.min_obs_ratio = min_obs_ratio
        self.seed = seed
        self.fixed_positions = None
        self.max_obs_count = 0
        self.time_masks = {}
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
    
    def generate_unified_observations(self, image_shape: tuple, time_steps: list) -> int:
        """Generate unified observation positions for all time steps"""
        if len(image_shape) == 3:
            C, H, W = image_shape
        else:
            raise ValueError(f"Expected 3D image shape (C, H, W), got {image_shape}")
        
        total_pixels = C * H * W
        self.max_obs_count = int(total_pixels * self.max_obs_ratio)
        
        self.fixed_positions = torch.randperm(total_pixels)[:self.max_obs_count]
        print(f"Fixed observation positions generated: {self.max_obs_count} positions")
        
        for i, t in enumerate(time_steps):
            obs_ratio = np.random.uniform(self.min_obs_ratio, self.max_obs_ratio)
            num_valid = int(total_pixels * obs_ratio)
            num_valid = min(num_valid, self.max_obs_count)
            
            valid_indices = torch.randperm(self.max_obs_count)[:num_valid]
            
            self.time_masks[i] = {
                'num_valid': num_valid,
                'valid_indices': valid_indices,
                'obs_ratio': obs_ratio
            }
            
            print(f"Time step {t}: {num_valid}/{self.max_obs_count} observations ({obs_ratio:.3f} ratio)")
        
        return self.max_obs_count
    
    def apply_unified_observation(self, full_image: torch.Tensor, time_step_idx: int) -> torch.Tensor:
        """Apply observation mask to full image"""
        if time_step_idx not in self.time_masks:
            raise ValueError(f"Time step index {time_step_idx} not found in masks")
        
        mask_info = self.time_masks[time_step_idx]
        
        flat_image = full_image.flatten()
        fixed_obs = flat_image[self.fixed_positions]
        
        obs_vector = torch.zeros(self.max_obs_count, device=full_image.device)
        valid_indices = mask_info['valid_indices']
        obs_vector[valid_indices] = fixed_obs[valid_indices]
        
        return obs_vector
    
    def create_block_R_matrix(self, base_variance=1e-3):
        """Create block diagonal R matrix"""
        R = torch.eye(self.max_obs_count) * base_variance
        return R


# Global variables for observation handler
_global_obs_handler = None
_global_time_idx = 0


def update_observation_time_index(time_idx: int):
    """Update global time index for observations"""
    global _global_time_idx
    _global_time_idx = time_idx


def H_unified(x):
    """Observation operator for unified sparse observations"""
    global _global_time_idx, _global_obs_handler
    
    x_reconstructed = forward_model.K_S_preimage(x)
    sparse_obs = _global_obs_handler.apply_unified_observation(
        x_reconstructed.squeeze(), _global_time_idx
    )
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


def run_data_assimilation(
    max_obs_ratio: float = 0.11,
    min_obs_ratio: float = 0.09,
    start_da_end_idxs: tuple = (700, 800, 900),
    time_obs: list = None,
    gaps: list = None,
    early_stop_config: tuple = (10, 1e-4),
    model_name: str = "CAE_DMD",
    model_display_name: str = "CAE DMD",
    residual_vmin: float = 0,
    residual_vmax: float = 5
):
    
    # Set default values if not provided
    if time_obs is None:
        time_obs = [
            start_da_end_idxs[1],
            start_da_end_idxs[1] + 10,
            start_da_end_idxs[1] + 20,
        ]
    
    if gaps is None:
        gaps = [10] * (len(time_obs) - 1)
    
    # Initialize
    global forward_model, _global_obs_handler
    
    # Set random seed and device
    set_seed(42)
    g = torch.Generator()
    g.manual_seed(42)
    device = set_device()
    print(f"Using device: {device}")
    
    # Load forward model
    forward_model = CYLINDER_C_FORWARD().to(device)
    forward_model.load_state_dict(torch.load(f'../../../../results/{model_name}/Cylinder/jointly_model_weights/forward_model.pt', 
                                            weights_only=True, map_location=device))
    forward_model.eval()
    print("Forward model loaded")
    
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
    
    denorm = cyl_val_dataset.denormalizer()
    
    # Create a device-aware denormalizer
    def safe_denorm(x):
        """Device-aware denormalization"""
        if isinstance(x, torch.Tensor):
            # Ensure x is on CPU for denormalization
            x_cpu = x.cpu() if x.is_cuda else x
            mean = cyl_val_dataset.mean.reshape(1, -1, 1, 1)
            std = cyl_val_dataset.std.reshape(1, -1, 1, 1)
            return (x_cpu * std + mean).cpu()
        else:
            return denorm(x)
    
    # Get ground truth data
    groundtruth = cyl_val_dataset.data[val_idx, ...]
    groundtruth = torch.from_numpy(groundtruth)
    print(f"Ground truth shape: {groundtruth.shape}")
    
    # Initialize observation handler with provided parameters
    obs_handler = UnifiedDynamicSparseObservationHandler(
        max_obs_ratio=max_obs_ratio, 
        min_obs_ratio=min_obs_ratio, 
        seed=42
    )
    _global_obs_handler = obs_handler
    
    # Prepare observation data
    full_y_data = [
        cyl_val_dataset.normalize(groundtruth[i+1, ...])
        for i in range(groundtruth.shape[0])
        if i in time_obs
    ]
    full_y_data = torch.cat(full_y_data).to(device)
    print(f"Full observation data shape: {full_y_data.shape}")
    
    # Generate sparse observations
    sample_image_shape = full_y_data[0].shape
    max_obs_count = obs_handler.generate_unified_observations(sample_image_shape, range(len(time_obs)))
    
    sparse_y_data = []
    for i, full_img in enumerate(full_y_data):
        sparse_obs = obs_handler.apply_unified_observation(full_img, i)
        sparse_y_data.append(sparse_obs)
    sparse_y_data = torch.stack(sparse_y_data).to(device)
    print(f"Sparse observation shape: {sparse_y_data.shape}")
    
    # Set up DA matrices
    latent_dim = int(forward_model.C_forward.in_features)
    B = torch.eye(latent_dim, device=device)
    R = obs_handler.create_block_R_matrix(base_variance=1e-3).to(device)
    
    print(f"Background covariance B shape: {B.shape}")
    print(f"Observation covariance R shape: {R.shape}")
    
    # Configure 4D-Var with provided early stop configuration
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
        .set_max_iterations(5000)
        .set_early_stop(early_stop_config)
        .set_algorithm(torchda.Algorithms.Var4D)
        .set_device(torchda.Device.GPU)
        .set_output_sequence_length(1)
    )
    
    # Run 4D-Var assimilation
    print("\nRunning 4D-Var data assimilation...")
    outs_4d_da = []
    start_time = perf_counter()
    
    current_state = cyl_val_dataset.normalize(groundtruth[start_da_end_idxs[0]]).to(device)
    
    for i in range(start_da_end_idxs[0], start_da_end_idxs[-1] + 1):
        print(f"Processing step {i}")
        
        z_current = forward_model.K_S(current_state)
        
        if i == start_da_end_idxs[1]:
            update_observation_time_index(0)
            
            case_to_run.set_background_state(z_current.ravel())
            da_start_time = perf_counter()
            result = case_to_run.execute()
            da_time = perf_counter() - da_start_time
            z_assimilated = result["assimilated_state"]
            
            intermediate_results = result["intermediate_results"]
            final_cost = intermediate_results["J"][-1]
            num_iterations = len(intermediate_results['J'])
            avg_time_per_iteration = da_time / num_iterations
            print(f"Final cost function: {final_cost}")
            print(f"Number of iterations: {num_iterations}")
            print(f"Average time per iteration: {avg_time_per_iteration:.6f}s")
            
            outs_4d_da.append(z_assimilated)
            current_state = forward_model.K_S_preimage(z_assimilated)
        else:
            outs_4d_da.append(z_current)
            z_next = dmd_wrapper(z_current)
            current_state = forward_model.K_S_preimage(z_next)
        
        print("=" * 50)
    
    print(f"4D-Var time: {perf_counter() - start_time:.2f}s")
    
    # Run baseline (no DA)
    print("\nRunning baseline (no DA)...")
    outs_no_4d_da = []
    start_time = perf_counter()
    
    with torch.no_grad():
        current_state = cyl_val_dataset.normalize(groundtruth[start_da_end_idxs[0]]).to(device)
        
        for i in range(start_da_end_idxs[0], start_da_end_idxs[-1] + 1):
            print(f"Step {i}")
            
            z_current = forward_model.K_S(current_state)
            outs_no_4d_da.append(z_current)
            
            z_next = forward_model.latent_forward(z_current)
            next_state = forward_model.K_S_preimage(z_next)
            current_state = next_state
            print("=" * 30)
    
    print(f"Baseline time: {perf_counter() - start_time:.2f}s")
    
    # Compute metrics
    print("\nComputing metrics...")
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

    print(diffs_da_real_mse[100])
    print(diffs_noda_real_mse[100])
    print(diffs_da_real_ssim[100])
    print(diffs_noda_real_ssim[100])

    print(diffs_da_real_mse[101])
    print(diffs_noda_real_mse[101])
    print(diffs_da_real_ssim[101])
    print(diffs_noda_real_ssim[101])

    print(diffs_da_real_mse[102])
    print(diffs_noda_real_mse[102])
    print(diffs_da_real_ssim[102])
    print(diffs_noda_real_ssim[102])
    
    # Save results with model-specific path
    results_data = {
        'diffs_da_real_mse': diffs_da_real_mse,
        'diffs_noda_real_mse': diffs_noda_real_mse,
        'diffs_da_real_rrmse': diffs_da_real_rrmse,
        'diffs_noda_real_rrmse': diffs_noda_real_rrmse,
        'diffs_da_real_ssim': diffs_da_real_ssim,
        'diffs_noda_real_ssim': diffs_noda_real_ssim
    }
    
    results_path = f'../../../../results/{model_name}/DA/cyl_comp_data.pkl'
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'wb') as f:
        pickle.dump(results_data, f)
    print(f"Results saved to {results_path}")
    
    # Plot metrics
    plot_metrics(diffs_da_real_mse, diffs_noda_real_mse,
                diffs_da_real_rrmse, diffs_noda_real_rrmse,
                diffs_da_real_ssim, diffs_noda_real_ssim,
                start_da_end_idxs, time_obs, model_name)
    
    # Generate comparison figure
    da_idxs = [100, 110, 120]
    generate_comparison_figure(groundtruth, outs_4d_da, outs_no_4d_da, 
                             da_idxs, time_obs, forward_model, safe_denorm,
                             model_name, model_display_name, 
                             residual_vmin, residual_vmax)
    
    print("\nData assimilation completed!")


def plot_metrics(diffs_da_real_mse, diffs_noda_real_mse,
                diffs_da_real_rrmse, diffs_noda_real_rrmse,
                diffs_da_real_ssim, diffs_noda_real_ssim,
                start_da_end_idxs, time_obs, model_name):
    """Plot comparison metrics"""
    step_idxs = list(range(start_da_end_idxs[0] + 1, start_da_end_idxs[-1] + 2))
    
    _, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    ax[0].plot(step_idxs, diffs_da_real_mse, color="#e377c2", label="4D-Var")
    ax[0].plot(step_idxs, diffs_noda_real_mse, color="#1f77b4", label="No DA")
    
    ax[1].plot(step_idxs, diffs_da_real_rrmse, color="#e377c2", label="4D-Var")
    ax[1].plot(step_idxs, diffs_noda_real_rrmse, color="#1f77b4", label="No DA")
    
    ax[2].plot(step_idxs, diffs_da_real_ssim, color="#e377c2", label="4D-Var")
    ax[2].plot(step_idxs, diffs_noda_real_ssim, color="#1f77b4", label="No DA")
    
    for i, name in enumerate(["MSE", "RRMSE", "SSIM"]):
        ax[i].set_xlabel("Step Index")
        ax[i].set_ylabel(name)
        ax[i].grid(True)
        ax[i].legend()
        for x in time_obs:
            ax[i].axvline(x=x+1, color="k", linestyle="--")
    
    plt.tight_layout()
    
    # Save with model-specific path
    metrics_path = f'../../../../results/{model_name}/DA/cyl_metrics_comparison.png'
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    plt.savefig(metrics_path, dpi=300)
    plt.close()
    print(f"Metrics plot saved to {metrics_path}")


def generate_comparison_figure(groundtruth, outs_4d_da, outs_no_4d_da, 
                             da_idxs, time_obs, forward_model, denorm,
                             model_name, model_display_name,
                             residual_vmin, residual_vmax):
    """Generate comparison figure"""
    # Create a device-aware denormalizer
    def safe_denorm(x):
        """Device-aware denormalization"""
        if isinstance(x, torch.Tensor):
            x_cpu = x.cpu() if x.is_cuda else x
            # Get mean and std from the dataset through denorm closure
            # This is a bit hacky but works with the existing denormalizer
            try:
                return denorm(x_cpu)
            except:
                # If denorm fails, return as is
                return x_cpu
        else:
            return denorm(x)
    
    n_times = len(da_idxs)
    figsize = (1.5 * n_times + 0.5, 7)
    threshold = 0.1
    dpi = 300
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(5, n_times, figure=fig, hspace=0.02, wspace=0.01, 
                  left=0.1, right=0.88, top=0.90, bottom=0.06)
    
    # Use model display name for title
    fig.suptitle(model_display_name, fontsize=14, fontweight='bold', y=0.96)
    
    # Create subplots
    ax = []
    for i in range(5):
        row = []
        for j in range(n_times):
            row.append(fig.add_subplot(gs[i, j]))
        ax.append(row)
    ax = np.array(ax)
    
    # Remove ticks and spines
    for axes in ax.flat:
        axes.set_xticks([])
        axes.set_yticks([])
        for spine in axes.spines.values():
            spine.set_visible(False)
    
    im1 = None
    im2 = None
    
    with torch.no_grad():
        for i, da_idx in enumerate(da_idxs):
            if i == 0:
                img_tensor = (groundtruth[time_obs[i]+1, 0, ...] ** 2 + 
                            groundtruth[time_obs[i]+1, 1, ...] ** 2) ** 0.5
            else:
                img_tensor = (groundtruth[time_obs[i], 0, ...] ** 2 + 
                            groundtruth[time_obs[i], 1, ...] ** 2) ** 0.5
            
            # Plot ground truth
            im1 = ax[0, i].imshow(img_tensor.reshape(64, 64), cmap="viridis", aspect='equal')
            
            # Get reconstructions
            no_da = forward_model.K_S_preimage(outs_no_4d_da[da_idx]).cpu()
            da = forward_model.K_S_preimage(outs_4d_da[da_idx]).cpu()
            
            de_no_da = safe_denorm(no_da)
            de_da = safe_denorm(da)
            
            image_noda = (de_no_da[0, 0, ...] ** 2 + de_no_da[0, 1, ...] ** 2) ** 0.5
            image_da = (de_da[0, 0, ...] ** 2 + de_da[0, 1, ...] ** 2) ** 0.5
            
            # Plot reconstructions
            ax[1, i].imshow(image_noda.reshape(64, 64), cmap="viridis", aspect='equal')
            ax[2, i].imshow(image_da.reshape(64, 64), cmap="viridis", aspect='equal')
            
            # Calculate and plot residuals with provided vmin/vmax
            res_no_da = (img_tensor.reshape(64, 64) - image_noda.reshape(64, 64)).abs()
            res_no_da = torch.where(res_no_da > threshold, res_no_da, 0)
            
            res_da = (img_tensor.reshape(64, 64) - image_da.reshape(64, 64)).abs()
            res_da = torch.where(res_da > threshold, res_da, 0)
            
            im2 = ax[3, i].imshow(res_no_da, cmap="magma", aspect='equal', 
                                vmin=residual_vmin, vmax=residual_vmax)
            ax[4, i].imshow(res_da, cmap="magma", aspect='equal', 
                          vmin=residual_vmin, vmax=residual_vmax)
    
    # Add titles
    for i in range(n_times):
        ax[0, i].set_title(f"$t_{i+1}$ = {time_obs[i]}", fontsize=11, pad=2)
    
    # Add row labels
    row_labels = ["Ground Truth", "No DA", "4D Var", "No DA Error", "4D Var Error"]
    for i, label in enumerate(row_labels):
        ax[i, 0].text(-0.15, 0.5, label, transform=ax[i, 0].transAxes,
                     fontsize=10, ha='right', va='center', rotation=90)
    
    # Add colorbars
    cax1 = fig.add_axes([0.89, 0.48, 0.015, 0.38])
    cbar1 = fig.colorbar(im1, cax=cax1)
    cbar1.ax.tick_params(labelsize=8)
    cbar1.set_label('Magnitude', rotation=270, labelpad=15, fontsize=9)
    
    cax2 = fig.add_axes([0.89, 0.08, 0.015, 0.3])
    cbar2 = fig.colorbar(im2, cax=cax2)
    cbar2.ax.tick_params(labelsize=8)
    cbar2.set_label('|Residual|', rotation=270, labelpad=15, fontsize=9)
    
    # Save figure with model-specific path and filename
    save_filename = f"../../../../results/{model_name}/DA/cyl_{model_name}.png"
    os.makedirs(os.path.dirname(save_filename), exist_ok=True)
    plt.savefig(save_filename, dpi=dpi, bbox_inches='tight', pad_inches=0.05)
    print(f"Comparison figure saved to: {save_filename}")
    plt.close()


if __name__ == "__main__":
    # Example usage with different model configurations
    
    # CAE DMD (default)
    run_data_assimilation(
        max_obs_ratio=0.055,
        min_obs_ratio=0.045,
        start_da_end_idxs=(700, 800, 900),
        time_obs=None,  # Will use default
        gaps=None,      # Will use default
        early_stop_config=(100, 1e-2),
        model_name="CAE_Linear",
        model_display_name="Linear ROM",
        residual_vmin=0,
        residual_vmax=5
    )