#!/usr/bin/env python
# coding: utf-8

"""
Data Assimilation for DMD cylinder model - Latent Space Analysis
"""

import random
import os
import sys
from time import perf_counter

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE

# Set matplotlib backend and config directory
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_config'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

import pickle
from datetime import datetime

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

def save_tsne_data(gt_latents, no_da_latents, da_latents, model_name, 
                   tsne_2d_results=None, tsne_3d_results=None):
    """
    Save t-SNE dimensionality reduction results to pickle file
    
    Args:
        gt_latents: Ground truth latent representations list
        no_da_latents: No data assimilation latent representations list  
        da_latents: Data assimilation latent representations list
        model_name: Model name string
        tsne_2d_results: 2D t-SNE results dictionary
        tsne_3d_results: 3D t-SNE results dictionary
    
    Returns:
        str: Path to saved pickle file
    """
    
    # Convert original latent representations to numpy arrays
    gt_latents_np = np.array([lat.cpu().numpy().flatten() for lat in gt_latents])
    no_da_latents_np = np.array([lat.cpu().numpy().flatten() for lat in no_da_latents])
    da_latents_np = np.array([lat.detach().cpu().numpy().flatten() for lat in da_latents])
    
    # Prepare data dictionary to save
    save_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'num_gt_samples': len(gt_latents_np),
            'num_no_da_samples': len(no_da_latents_np), 
            'num_da_samples': len(da_latents_np),
            'latent_dimension': gt_latents_np.shape[1] if len(gt_latents_np) > 0 else 0
        },
        'original_latents': {
            'ground_truth': gt_latents_np,
            'no_data_assimilation': no_da_latents_np,
            'data_assimilation': da_latents_np
        }
    }
    
    # Add 2D t-SNE results
    if tsne_2d_results is not None:
        save_data['tsne_2d'] = tsne_2d_results
    
    # Add 3D t-SNE results  
    if tsne_3d_results is not None:
        save_data['tsne_3d'] = tsne_3d_results
    
    # Create save directory
    save_dir = f'../../../../results/{model_name}/DA/'
    os.makedirs(save_dir, exist_ok=True)
    
    # Save as pickle file
    pickle_path = os.path.join(save_dir, 'latent_space_analysis_data.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"Data saved to: {pickle_path}")
    
    return pickle_path


def plot_latent_space_tsne_with_save(gt_latents, no_da_latents, da_latents, 
                                    start_da_end_idxs, model_name, model_display_name):
    """Modified 2D t-SNE plotting function that returns dimensionality reduction results"""

    # Convert to numpy arrays
    gt_latents_np = np.array([lat.cpu().numpy().flatten() for lat in gt_latents])
    no_da_latents_np = np.array([lat.cpu().numpy().flatten() for lat in no_da_latents])
    da_latents_np = np.array([lat.detach().cpu().numpy().flatten() for lat in da_latents])

    # Combine all latent vectors for t-SNE
    all_latents = np.vstack([gt_latents_np, no_da_latents_np, da_latents_np])

    # Create labels for each group
    n_gt, n_no_da, n_da = len(gt_latents_np), len(no_da_latents_np), len(da_latents_np)

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, all_latents.shape[0] // 4))
    latents_2d = tsne.fit_transform(all_latents)

    # Split back into groups
    gt_2d = latents_2d[:n_gt]
    no_da_2d = latents_2d[n_gt:n_gt + n_no_da]
    da_2d = latents_2d[n_gt + n_no_da:]

    # === Original plotting code remains unchanged ===
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 7))

    colors = {
        'One step': '#808080',   # Gray
        'No Data Assimilation': 'blue',  # Bright blue
        '4D-Var': 'red'          # Bright red
    }
    alphas = {
        'One step': 0.4,
        'No Data Assimilation': 0.7,
        '4D-Var': 0.7
    }
    markers = {
        'One step': 'o',
        'No Data Assimilation': 's',
        '4D-Var': '^'
    }
    sizes = {
        'One step': 70,
        'No Data Assimilation': 70,
        '4D-Var': 70
    }

    # Plot
    ax.scatter(gt_2d[:, 0], gt_2d[:, 1],
               c=colors['One step'], marker=markers['One step'],
               s=sizes['One step'], alpha=alphas['One step'],
               edgecolors='white', linewidth=0.6, label='One step')

    ax.scatter(no_da_2d[:, 0], no_da_2d[:, 1],
               c=colors['No Data Assimilation'], marker=markers['No Data Assimilation'],
               s=sizes['No Data Assimilation'], alpha=alphas['No Data Assimilation'],
               edgecolors='white', linewidth=0.6, label='No Data Assimilation')

    ax.scatter(da_2d[:, 0], da_2d[:, 1],
               c=colors['4D-Var'], marker=markers['4D-Var'],
               s=sizes['4D-Var'], alpha=alphas['4D-Var'],
               edgecolors='white', linewidth=0.6, label='4D-Var')

    # Labels
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(f'{model_display_name}', fontsize=15, fontweight='bold', pad=15)

    # Legend
    legend = ax.legend(loc='upper right', frameon=True, fontsize=11)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('lightgray')
    legend.get_frame().set_alpha(0.95)

    # Styling
    ax.set_facecolor('white')
    ax.grid(True, linestyle='--', alpha=0.25)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color('gray')

    ax.tick_params(axis='both', which='major', labelsize=11, width=1.1)
    plt.tight_layout()

    # Save figure
    save_path = f'../../../../results/{model_name}/DA/latent_space_tsne.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Return 2D t-SNE results
    return {
        'ground_truth_2d': gt_2d,
        'no_data_assimilation_2d': no_da_2d, 
        'data_assimilation_2d': da_2d,
        'tsne_parameters': {
            'n_components': 2,
            'random_state': 42,
            'perplexity': min(30, all_latents.shape[0] // 4)
        }
    }


def plot_latent_space_tsne_3d_with_save(gt_latents, no_da_latents, da_latents, 
                                       start_da_end_idxs, model_name, model_display_name):
    """Modified 3D t-SNE plotting function that returns dimensionality reduction results"""
    
    # Convert to numpy arrays
    gt_latents_np = np.array([lat.cpu().numpy().flatten() for lat in gt_latents])
    no_da_latents_np = np.array([lat.cpu().numpy().flatten() for lat in no_da_latents])
    da_latents_np = np.array([lat.detach().cpu().numpy().flatten() for lat in da_latents])
    
    # Combine all latent vectors for t-SNE
    all_latents = np.vstack([gt_latents_np, no_da_latents_np, da_latents_np])
    
    # Group sizes
    n_gt = len(gt_latents_np)
    n_no_da = len(no_da_latents_np)
    n_da = len(da_latents_np)
    
    # 3D t-SNE
    tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, all_latents.shape[0]//4))
    latents_3d = tsne.fit_transform(all_latents)
    
    # Split back into groups
    gt_3d = latents_3d[:n_gt]
    no_da_3d = latents_3d[n_gt:n_gt+n_no_da]
    da_3d = latents_3d[n_gt+n_no_da:]
    
    # === Original plotting code remains unchanged ===
    colors = {
        'One step': '#808080',   # Gray
        'No Data Assimilation': 'blue',  # Bright blue
        '4D-Var': 'red'          # Bright red
    }
    alphas = {
        'One step': 0.4,
        'No Data Assimilation': 0.5,
        '4D-Var': 0.7
    }
    markers = {
        'One step': 'o',
        'No Data Assimilation': 's', 
        '4D-Var': '^'
    }
    sizes = {
        'One step': 40,
        'No Data Assimilation': 50,
        '4D-Var': 70
    }
    
    # Create plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter points only
    ax.scatter(gt_3d[:, 0], gt_3d[:, 1], gt_3d[:, 2],
               c=colors['One step'], marker=markers['One step'],
               s=sizes['One step'], alpha=alphas['One step'],
               edgecolors='white', linewidth=0.5, label='One step')
    ax.scatter(no_da_3d[:, 0], no_da_3d[:, 1], no_da_3d[:, 2],
               c=colors['No Data Assimilation'], marker=markers['No Data Assimilation'],
               s=sizes['No Data Assimilation'], alpha=alphas['No Data Assimilation'],
               edgecolors='white', linewidth=0.5, label='No Data Assimilation')
    ax.scatter(da_3d[:, 0], da_3d[:, 1], da_3d[:, 2],
               c=colors['4D-Var'], marker=markers['4D-Var'],
               s=sizes['4D-Var'], alpha=alphas['4D-Var'],
               edgecolors='white', linewidth=0.5, label='4D-Var')
    
    # Labels
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_title(f'{model_display_name}', fontsize=16, fontweight='bold', pad=20)
    
    # Legend
    legend = ax.legend(loc='upper left', frameon=True, shadow=True, fontsize=11, title_fontsize=12,
                       bbox_to_anchor=(0.02, 0.98))
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_edgecolor('gray')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    save_path = f'../../../../results/{model_name}/DA/latent_space_3d_tsne.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Return 3D t-SNE results
    return {
        'ground_truth_3d': gt_3d,
        'no_data_assimilation_3d': no_da_3d,
        'data_assimilation_3d': da_3d,
        'tsne_parameters': {
            'n_components': 3,
            'random_state': 42,
            'perplexity': min(30, all_latents.shape[0] // 4)
        }
    }


def run_latent_space_analysis(
    max_obs_ratio: float = 0.11,
    min_obs_ratio: float = 0.09,
    start_da_end_idxs: tuple = (700, 800, 900),
    time_obs: list = None,
    gaps: list = None,
    early_stop_config: tuple = (10, 1e-4),
    model_name: str = "CAE_DMD",
    model_display_name: str = "CAE DMD"
):
    """Run data assimilation and analyze latent space distributions"""
    
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
    
    # Get ground truth data
    groundtruth = cyl_val_dataset.data[val_idx, ...]
    groundtruth = torch.from_numpy(groundtruth)
    print(f"Ground truth shape: {groundtruth.shape}")
    
    # Initialize observation handler
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
        .set_optimizer_args({"lr": 0.01})
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
            
            print(f"DA completed in {da_time:.2f}s")
            
            outs_4d_da.append(z_assimilated)
            current_state = forward_model.K_S_preimage(z_assimilated)
        else:
            outs_4d_da.append(z_current)
            z_next = dmd_wrapper(z_current)
            current_state = forward_model.K_S_preimage(z_next)
    
    print(f"4D-Var completed in {perf_counter() - start_time:.2f}s")
    
    # Run baseline (no DA)
    print("\nRunning baseline (no DA)...")
    outs_no_4d_da = []
    
    with torch.no_grad():
        current_state = cyl_val_dataset.normalize(groundtruth[start_da_end_idxs[0]]).to(device)
        
        for i in range(start_da_end_idxs[0], start_da_end_idxs[-1] + 1):
            z_current = forward_model.K_S(current_state)
            outs_no_4d_da.append(z_current)
            
            z_next = forward_model.latent_forward(z_current)
            next_state = forward_model.K_S_preimage(z_next)
            current_state = next_state
    
    # Generate ground truth latent trajectory with forward propagation
    print("\nGenerating ground truth latent trajectory...")
    gt_latents = []
    
    with torch.no_grad():
        # Start from the same initial condition as other methods
        current_state = cyl_val_dataset.normalize(groundtruth[start_da_end_idxs[0]]).to(device)
        
        for i in range(start_da_end_idxs[0], start_da_end_idxs[-1] + 1):
            # Encode current ground truth to latent space
            gt_normalized = cyl_val_dataset.normalize(groundtruth[i]).to(device)
            gt_latent = forward_model.K_S(gt_normalized)
            gt_latents.append(gt_latent)
            
            # For next iteration, use forward propagation in latent space
            # This ensures time alignment with the DA methods
            if i < start_da_end_idxs[-1]:
                gt_latent_next = forward_model.latent_forward(gt_latent)
                # Convert back to image space for next iteration's encoding
                # This creates a consistent latent trajectory that follows the model dynamics
                current_state = forward_model.K_S_preimage(gt_latent_next)
    
    print(f"Generated {len(gt_latents)} ground truth latent representations")
    print(f"Generated {len(outs_no_4d_da)} no-DA latent representations") 
    print(f"Generated {len(outs_4d_da)} DA latent representations")
    
    # Plot latent space analysis
    print("\nCreating t-SNE visualization and saving data...")
    
    # Generate 2D t-SNE and get dimensionality reduction results
    tsne_2d_results = plot_latent_space_tsne_with_save(
        gt_latents, outs_no_4d_da, outs_4d_da,
        start_da_end_idxs, model_name, model_display_name
    )
    
    # Generate 3D t-SNE and get dimensionality reduction results  
    tsne_3d_results = plot_latent_space_tsne_3d_with_save(
        gt_latents, outs_no_4d_da, outs_4d_da,
        start_da_end_idxs, model_name, model_display_name
    )
    
    # Save all data to pickle file
    pickle_path = save_tsne_data(
        gt_latents, outs_no_4d_da, outs_4d_da, model_name,
        tsne_2d_results, tsne_3d_results
    )

    print(f"\nAll data saved to: {pickle_path}")
    
    print("\nLatent space analysis completed!")


if __name__ == "__main__":
    # Run latent space analysis
    run_latent_space_analysis(
        max_obs_ratio=0.055,
        min_obs_ratio=0.045,
        start_da_end_idxs=(700, 800, 900),
        time_obs=None,  # Will use default
        gaps=None,      # Will use default
        early_stop_config=(100, 1e-2),
        model_name="CAE_Linear",
        model_display_name="Linear ROM"
    )