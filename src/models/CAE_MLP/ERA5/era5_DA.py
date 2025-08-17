#!/usr/bin/env python
# coding: utf-8

"""
Data Assimilation for DMD ERA5 model
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

from era5_model_FTF import ERA5_C_FORWARD
from src.utils.Dataset import ERA5Dataset
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
    start_da_end_idxs: tuple = (50, 60, 90),
    time_obs: list = None,
    gaps: list = None,
    early_stop_config: tuple = (10, 1e-4),
    model_name: str = "CAE_DMD",
    model_display_name: str = "CAE DMD",
    start_T: int = 1000,
    prediction_step: int = 100
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
    forward_model = ERA5_C_FORWARD().to(device)
    forward_model.load_state_dict(torch.load(f'../../../../results/{model_name}/ERA5/model_weights_FTF/forward_model.pt', 
                                            weights_only=True, map_location=device))
    forward_model.eval()
    print("Forward model loaded")
    
    # Load datasets
    forward_step = 12
    
    era5_test_dataset = ERA5Dataset(
        data_path="../../../../data/ERA5/ERA5_data/test_seq_state.h5",
        seq_length=forward_step,
        min_path="../../../../data/ERA5/ERA5_data/min_val.npy",
        max_path="../../../../data/ERA5/ERA5_data/max_val.npy"
    )
    
    denorm = era5_test_dataset.denormalizer()
    
    # Create a device-aware denormalizer
    def safe_denorm(x):
        """Device-aware denormalization"""
        if isinstance(x, torch.Tensor):
            # Ensure x is on CPU for denormalization
            x_cpu = x.cpu() if x.is_cuda else x
            min_val = era5_test_dataset.min.reshape(-1, 1, 1)
            max_val = era5_test_dataset.max.reshape(-1, 1, 1)
            return (x_cpu * (max_val - min_val) + min_val).cpu()
        else:
            return denorm(x)
    
    # Get ground truth data
    raw_test_data = era5_test_dataset.data  # shape: [N, H, W, C]
    groundtruth = raw_test_data[start_T:start_T + prediction_step, ...]  # (100, 64, 32, 5)
    groundtruth = torch.tensor(groundtruth, dtype=torch.float32).permute(0, 3, 1, 2)  # (100, 5, 64, 32)
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
        era5_test_dataset.normalize(groundtruth[i+1, ...])
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
    latent_dim = int(forward_model.hidden_dim)
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
    
    current_state = era5_test_dataset.normalize(groundtruth[start_da_end_idxs[0]]).to(device)
    
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
            
            z_assimilated = z_assimilated.unsqueeze(0)
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
        current_state = era5_test_dataset.normalize(groundtruth[start_da_end_idxs[0]]).to(device)
        
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
    def compute_channel_wise_metrics():
        """Compute metrics for each channel separately"""
        # Initialize channel-wise metrics dictionaries
        channel_names = ['Temperature', 'U_wind', 'V_wind', 'Humidity', 'Pressure']  # Adjust names as needed

        diffs_da_real_mse_channels = {f'channel_{c}': [] for c in range(5)}
        diffs_noda_real_mse_channels = {f'channel_{c}': [] for c in range(5)}
        diffs_da_real_rrmse_channels = {f'channel_{c}': [] for c in range(5)}
        diffs_noda_real_rrmse_channels = {f'channel_{c}': [] for c in range(5)}
        diffs_da_real_ssim_channels = {f'channel_{c}': [] for c in range(5)}
        diffs_noda_real_ssim_channels = {f'channel_{c}': [] for c in range(5)}

        # Overall metrics (average across channels)
        diffs_da_real_mse = []
        diffs_noda_real_mse = []
        diffs_da_real_rrmse = []
        diffs_noda_real_rrmse = []
        diffs_da_real_ssim = []
        diffs_noda_real_ssim = []

        with torch.no_grad():
            for i, (no_da, da) in enumerate(zip(outs_no_4d_da, outs_4d_da), start=start_da_end_idxs[0]):
                da_img = forward_model.K_S_preimage(da).view(5, 64, 32).cpu()
                noda_img = forward_model.K_S_preimage(no_da).view(5, 64, 32).cpu()
                
                de_da_img = safe_denorm(da_img)
                de_noda_img = safe_denorm(noda_img)

                # Compute metrics for each channel
                channel_mse_da = []
                channel_mse_noda = []
                channel_rrmse_da = []
                channel_rrmse_noda = []
                channel_ssim_da = []
                channel_ssim_noda = []
                
                for c in range(5):
                    # MSE for this channel
                    da_minus_real_img_square_c = (de_da_img[c] - groundtruth[i][c]) ** 2
                    noda_minus_real_img_square_c = (de_noda_img[c] - groundtruth[i][c]) ** 2
                    
                    mse_da_c = da_minus_real_img_square_c.mean().item()
                    mse_noda_c = noda_minus_real_img_square_c.mean().item()
                    
                    # RRMSE for this channel
                    rrmse_da_c = (da_minus_real_img_square_c.sum() / ((groundtruth[i][c]**2).sum())).sqrt().item()
                    rrmse_noda_c = (noda_minus_real_img_square_c.sum() / ((groundtruth[i][c]**2).sum())).sqrt().item()
                    
                    # SSIM for this channel
                    data_range_c = groundtruth[i][c].max().item() - groundtruth[i][c].min().item()
                    if data_range_c > 0:
                        ssim_da_c = ssim(groundtruth[i][c].numpy(), de_da_img[c].numpy(), data_range=data_range_c)
                        ssim_noda_c = ssim(groundtruth[i][c].numpy(), de_noda_img[c].numpy(), data_range=data_range_c)
                    else:
                        ssim_da_c = 1.0  # Perfect similarity if no variation
                        ssim_noda_c = 1.0
                    
                    # Store channel-wise metrics
                    diffs_da_real_mse_channels[f'channel_{c}'].append(mse_da_c)
                    diffs_noda_real_mse_channels[f'channel_{c}'].append(mse_noda_c)
                    diffs_da_real_rrmse_channels[f'channel_{c}'].append(rrmse_da_c)
                    diffs_noda_real_rrmse_channels[f'channel_{c}'].append(rrmse_noda_c)
                    diffs_da_real_ssim_channels[f'channel_{c}'].append(ssim_da_c)
                    diffs_noda_real_ssim_channels[f'channel_{c}'].append(ssim_noda_c)
                    
                    # For overall average
                    channel_mse_da.append(mse_da_c)
                    channel_mse_noda.append(mse_noda_c)
                    channel_rrmse_da.append(rrmse_da_c)
                    channel_rrmse_noda.append(rrmse_noda_c)
                    channel_ssim_da.append(ssim_da_c)
                    channel_ssim_noda.append(ssim_noda_c)
                
                # Overall metrics (average across channels)
                diffs_da_real_mse.append(np.mean(channel_mse_da))
                diffs_noda_real_mse.append(np.mean(channel_mse_noda))
                diffs_da_real_rrmse.append(np.mean(channel_rrmse_da))
                diffs_noda_real_rrmse.append(np.mean(channel_rrmse_noda))
                diffs_da_real_ssim.append(np.mean(channel_ssim_da))
                diffs_noda_real_ssim.append(np.mean(channel_ssim_noda))

        return (diffs_da_real_mse_channels, diffs_noda_real_mse_channels,
                diffs_da_real_rrmse_channels, diffs_noda_real_rrmse_channels,
                diffs_da_real_ssim_channels, diffs_noda_real_ssim_channels,
                diffs_da_real_mse, diffs_noda_real_mse,
                diffs_da_real_rrmse, diffs_noda_real_rrmse,
                diffs_da_real_ssim, diffs_noda_real_ssim)


    def plot_channel_wise_metrics(diffs_da_real_mse_channels, diffs_noda_real_mse_channels,
                                diffs_da_real_rrmse_channels, diffs_noda_real_rrmse_channels,
                                diffs_da_real_ssim_channels, diffs_noda_real_ssim_channels,
                                start_da_end_idxs, time_obs, model_name):
        """Plot channel-wise comparison metrics in a 5x3 grid"""
        step_idxs = list(range(start_da_end_idxs[0] + 1, start_da_end_idxs[-1] + 2))
        channel_names = ['Temperature', 'U_wind', 'V_wind', 'Humidity', 'Pressure']

        # Create 5x3 subplot grid
        fig, axes = plt.subplots(5, 3, figsize=(18, 20))

        for c in range(5):
            channel_key = f'channel_{c}'
            
            # MSE plot
            axes[c, 0].plot(step_idxs, diffs_da_real_mse_channels[channel_key], 
                            color="#e377c2", label="4D-Var", linewidth=2)
            axes[c, 0].plot(step_idxs, diffs_noda_real_mse_channels[channel_key], 
                            color="#1f77b4", label="No DA", linewidth=2)
            axes[c, 0].set_title(f'{channel_names[c]} - MSE', fontsize=12, fontweight='bold')
            axes[c, 0].set_ylabel('MSE', fontsize=10)
            axes[c, 0].grid(True, alpha=0.3)
            axes[c, 0].legend(fontsize=9)
            
            # RRMSE plot
            axes[c, 1].plot(step_idxs, diffs_da_real_rrmse_channels[channel_key], 
                            color="#e377c2", label="4D-Var", linewidth=2)
            axes[c, 1].plot(step_idxs, diffs_noda_real_rrmse_channels[channel_key], 
                            color="#1f77b4", label="No DA", linewidth=2)
            axes[c, 1].set_title(f'{channel_names[c]} - RRMSE', fontsize=12, fontweight='bold')
            axes[c, 1].set_ylabel('RRMSE', fontsize=10)
            axes[c, 1].grid(True, alpha=0.3)
            axes[c, 1].legend(fontsize=9)
            
            # SSIM plot
            axes[c, 2].plot(step_idxs, diffs_da_real_ssim_channels[channel_key], 
                            color="#e377c2", label="4D-Var", linewidth=2)
            axes[c, 2].plot(step_idxs, diffs_noda_real_ssim_channels[channel_key], 
                            color="#1f77b4", label="No DA", linewidth=2)
            axes[c, 2].set_title(f'{channel_names[c]} - SSIM', fontsize=12, fontweight='bold')
            axes[c, 2].set_ylabel('SSIM', fontsize=10)
            axes[c, 2].grid(True, alpha=0.3)
            axes[c, 2].legend(fontsize=9)
            
            # Add observation time lines to all subplots in this row
            for col in range(3):
                for x in time_obs:
                    axes[c, col].axvline(x=x+1, color="k", linestyle="--", alpha=0.7, linewidth=1)

        # Set x-labels only for bottom row
        for col in range(3):
            axes[4, col].set_xlabel('Step Index', fontsize=10)

        # Adjust layout
        plt.tight_layout(pad=2.0)

        # Save the figure
        metrics_path = f'../../../../results/{model_name}/DA/era5_channel_wise_metrics_comparison.png'
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Channel-wise metrics plot saved to {metrics_path}")


    # Modified save section
    def save_channel_wise_results(diffs_da_real_mse_channels, diffs_noda_real_mse_channels,
                                diffs_da_real_rrmse_channels, diffs_noda_real_rrmse_channels,
                                diffs_da_real_ssim_channels, diffs_noda_real_ssim_channels,
                                diffs_da_real_mse, diffs_noda_real_mse,
                                diffs_da_real_rrmse, diffs_noda_real_rrmse,
                                diffs_da_real_ssim, diffs_noda_real_ssim,
                                model_name):
        """Save both channel-wise and overall results"""

        # Save channel-wise results
        channel_results_data = {
            'diffs_da_real_mse_channels': diffs_da_real_mse_channels,
            'diffs_noda_real_mse_channels': diffs_noda_real_mse_channels,
            'diffs_da_real_rrmse_channels': diffs_da_real_rrmse_channels,
            'diffs_noda_real_rrmse_channels': diffs_noda_real_rrmse_channels,
            'diffs_da_real_ssim_channels': diffs_da_real_ssim_channels,
            'diffs_noda_real_ssim_channels': diffs_noda_real_ssim_channels,
            'channel_names': ['Temperature', 'U_wind', 'V_wind', 'Humidity', 'Pressure']
        }

        # Save overall results (backward compatibility)
        overall_results_data = {
            'diffs_da_real_mse': diffs_da_real_mse,
            'diffs_noda_real_mse': diffs_noda_real_mse,
            'diffs_da_real_rrmse': diffs_da_real_rrmse,
            'diffs_noda_real_rrmse': diffs_noda_real_rrmse,
            'diffs_da_real_ssim': diffs_da_real_ssim,
            'diffs_noda_real_ssim': diffs_noda_real_ssim
        }

        # Save channel-wise data
        channel_results_path = f'../../../../results/{model_name}/DA/era5_channel_wise_comp_data.pkl'
        os.makedirs(os.path.dirname(channel_results_path), exist_ok=True)
        with open(channel_results_path, 'wb') as f:
            pickle.dump(channel_results_data, f)
        print(f"Channel-wise results saved to {channel_results_path}")

        # Save overall data
        overall_results_path = f'../../../../results/{model_name}/DA/era5_comp_data.pkl'
        with open(overall_results_path, 'wb') as f:
            pickle.dump(overall_results_data, f)
        print(f"Overall results saved to {overall_results_path}")

    # Compute channel-wise metrics
    (diffs_da_real_mse_channels, diffs_noda_real_mse_channels,
    diffs_da_real_rrmse_channels, diffs_noda_real_rrmse_channels,
    diffs_da_real_ssim_channels, diffs_noda_real_ssim_channels,
    diffs_da_real_mse, diffs_noda_real_mse,
    diffs_da_real_rrmse, diffs_noda_real_rrmse,
    diffs_da_real_ssim, diffs_noda_real_ssim) = compute_channel_wise_metrics()

    # Print some sample metrics
    print(f"DA MSE (overall) at step {len(diffs_da_real_mse)//2}: {diffs_da_real_mse[len(diffs_da_real_mse)//2]}")
    print(f"No DA MSE (overall) at step {len(diffs_noda_real_mse)//2}: {diffs_noda_real_mse[len(diffs_noda_real_mse)//2]}")

    # Save results
    save_channel_wise_results(diffs_da_real_mse_channels, diffs_noda_real_mse_channels,
                            diffs_da_real_rrmse_channels, diffs_noda_real_rrmse_channels,
                            diffs_da_real_ssim_channels, diffs_noda_real_ssim_channels,
                            diffs_da_real_mse, diffs_noda_real_mse,
                            diffs_da_real_rrmse, diffs_noda_real_rrmse,
                            diffs_da_real_ssim, diffs_noda_real_ssim,
                            model_name)

    # Plot channel-wise metrics
    plot_channel_wise_metrics(diffs_da_real_mse_channels, diffs_noda_real_mse_channels,
                            diffs_da_real_rrmse_channels, diffs_noda_real_rrmse_channels,
                            diffs_da_real_ssim_channels, diffs_noda_real_ssim_channels,
                            start_da_end_idxs, time_obs, model_name)
    
    print("\nData assimilation completed!")


if __name__ == "__main__":
    # Example usage with ERA5 configuration
    
    # CAE DMD for ERA5
    run_data_assimilation(
        max_obs_ratio=0.205,
        min_obs_ratio=0.195,
        start_da_end_idxs=(50, 60, 90),
        time_obs=None,  # Will use default
        gaps=None,      # Will use default
        early_stop_config=(100, 1e-2),
        model_name="CAE_MLP",
        model_display_name="MLP ROM",
        start_T=1000,
        prediction_step=100
    )