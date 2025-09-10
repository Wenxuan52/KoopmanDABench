#!/usr/bin/env python
# coding: utf-8

"""
Multi-Rollout Latent Space Visualization - Simple Version with Data Saving
"""

import os
import sys
import torch
import numpy as np
import pickle
from datetime import datetime
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Add src directory to path
current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
sys.path.append(src_directory)

from cylinder_model import CYLINDER_C_FORWARD
from src.utils.Dataset import CylinderDynamicsDataset


def state_propagation_rollout(forward_model, initial_state, n_steps):
    """State space propagation rollout"""
    latents = []
    current_state = initial_state
    
    with torch.no_grad():
        for step in range(n_steps):
            z = forward_model.K_S(current_state)
            latents.append(z.squeeze(0))
            z_next = forward_model.latent_forward(z)
            current_state = forward_model.K_S_preimage(z_next)
    
    return latents


def latent_propagation_rollout(forward_model, initial_state, n_steps):
    """Latent space propagation rollout"""
    latents = []
    
    with torch.no_grad():
        z = forward_model.K_S(initial_state)
        latents.append(z.squeeze(0))
        
        for step in range(n_steps - 1):
            z = forward_model.latent_forward(z)
            latents.append(z.squeeze(0))
    
    return latents


def save_tsne_data(gt_latents, state_latents, latent_latents, model_name, 
                   tsne_2d_results=None, tsne_3d_results=None):
    
    # Convert original latent representations to numpy arrays
    gt_latents_np = np.array([lat.cpu().numpy().flatten() for lat in gt_latents])
    state_latents_np = np.array([lat.cpu().numpy().flatten() for lat in state_latents])
    latent_latents_np = np.array([lat.cpu().numpy().flatten() for lat in latent_latents])
    
    # Prepare data dictionary to save
    save_data = {
        'metadata': {
            'model_name': model_name,
            'num_gt_samples': len(gt_latents_np),
            'num_state_propagation_samples': len(state_latents_np), 
            'num_latent_propagation_samples': len(latent_latents_np),
            'latent_dimension': gt_latents_np.shape[1] if len(gt_latents_np) > 0 else 0
        },
        'original_latents': {
            'ground_truth': gt_latents_np,
            'state_propagation': state_latents_np,
            'latent_propagation': latent_latents_np
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
    pickle_path = os.path.join(save_dir, 'rollout_latent_space_analysis_data.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"Rollout analysis data saved to: {pickle_path}")
    
    return pickle_path


def plot_tsne_2d(gt_latents, state_latents, latent_latents, save_path):
    """Optimized 2D t-SNE comparison with time-based color gradients"""
    
    size = 150
    
    # Convert to numpy
    gt_np = np.array([lat.cpu().numpy().flatten() for lat in gt_latents])
    state_np = np.array([lat.cpu().numpy().flatten() for lat in state_latents])
    latent_np = np.array([lat.cpu().numpy().flatten() for lat in latent_latents])
    
    # Combine and run 2D t-SNE
    all_latents = np.vstack([gt_np, state_np, latent_np])
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latents_2d = tsne.fit_transform(all_latents)
    
    # Split results
    n_gt, n_state = len(gt_np), len(state_np)
    gt_2d = latents_2d[:n_gt]
    state_2d = latents_2d[n_gt:n_gt + n_state]
    latent_2d = latents_2d[n_gt + n_state:]
    
    # Create time-based colors
    n_steps = len(gt_latents)
    time_colors_gt = np.linspace(0.3, 0.8, n_steps)  # Gray gradient
    time_colors_state = np.linspace(0.3, 1.0, n_steps)  # Blue gradient
    time_colors_latent = np.linspace(0.3, 1.0, n_steps)  # Red gradient
    
    # Plot 2D
    plt.figure(figsize=(12, 8))
    
    # Ground Truth - gray gradient
    plt.scatter(gt_2d[:, 0], gt_2d[:, 1], c=time_colors_gt, 
                cmap='gray', marker='o', s=size, alpha=0.7)
    
    # State Propagation - blue gradient
    plt.scatter(state_2d[:, 0], state_2d[:, 1], c=time_colors_state, 
                cmap='Blues', marker='s', s=size, alpha=0.8)
    
    # Latent Propagation - red gradient
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=time_colors_latent, 
                cmap='Reds', marker='^', s=size, alpha=0.8)
    
    # Title bigger
    plt.title('Koopman ROM Latent Space', fontsize=34, weight='bold', pad=40)
    
    # Remove axis labels, keep ticks
    plt.xlabel('')
    plt.ylabel('')
    
    # Grid slightly stronger
    plt.grid(True, alpha=0.5, linewidth=0.7)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Return t-SNE results for saving
    tsne_2d_results = {
        'ground_truth': gt_2d,
        'state_propagation': state_2d,
        'latent_propagation': latent_2d,
        'time_colors_gt': time_colors_gt,
        'time_colors_state': time_colors_state,
        'time_colors_latent': time_colors_latent,
        'tsne_params': {
            'n_components': 2,
            'random_state': 42,
            'perplexity': 30
        }
    }
    
    return tsne_2d_results


def plot_tsne_3d(gt_latents, state_latents, latent_latents, save_path):
    """Plot 3D t-SNE comparison with time-based color gradients"""
    # Convert to numpy
    gt_np = np.array([lat.cpu().numpy().flatten() for lat in gt_latents])
    state_np = np.array([lat.cpu().numpy().flatten() for lat in state_latents])
    latent_np = np.array([lat.cpu().numpy().flatten() for lat in latent_latents])
    
    # Combine and run 3D t-SNE
    all_latents = np.vstack([gt_np, state_np, latent_np])
    tsne = TSNE(n_components=3, random_state=42, perplexity=30)
    latents_3d = tsne.fit_transform(all_latents)
    
    # Split results
    n_gt, n_state = len(gt_np), len(state_np)
    gt_3d = latents_3d[:n_gt]
    state_3d = latents_3d[n_gt:n_gt + n_state]
    latent_3d = latents_3d[n_gt + n_state:]
    
    # Create time-based colors
    n_steps = len(gt_latents)
    time_colors_gt = np.linspace(0.3, 0.8, n_steps)  # Gray gradient
    time_colors_state = np.linspace(0.3, 1.0, n_steps)  # Blue gradient
    time_colors_latent = np.linspace(0.3, 1.0, n_steps)  # Red gradient
    
    # Plot 3D
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Ground Truth - gray gradient
    scatter_gt = ax.scatter(gt_3d[:, 0], gt_3d[:, 1], gt_3d[:, 2], c=time_colors_gt, 
                           cmap='gray', marker='o', s=60, alpha=0.7, label='Ground Truth')
    
    # State Propagation - blue gradient
    scatter_state = ax.scatter(state_3d[:, 0], state_3d[:, 1], state_3d[:, 2], c=time_colors_state, 
                              cmap='Blues', marker='s', s=60, alpha=0.8, label='Manifold regularization')
    
    # Latent Propagation - red gradient
    scatter_latent = ax.scatter(latent_3d[:, 0], latent_3d[:, 1], latent_3d[:, 2], c=time_colors_latent, 
                               cmap='Reds', marker='^', s=60, alpha=0.8, label='Latent Propagation')
    
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3')
    ax.set_title('Rollout Methods - 3D Latent Space Distribution (Time Gradient)')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter_state, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label('Time Step', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Return t-SNE results for saving
    tsne_3d_results = {
        'ground_truth': gt_3d,
        'state_propagation': state_3d,
        'latent_propagation': latent_3d,
        'time_colors_gt': time_colors_gt,
        'time_colors_state': time_colors_state,
        'time_colors_latent': time_colors_latent,
        'tsne_params': {
            'n_components': 3,
            'random_state': 42,
            'perplexity': 30
        }
    }
    
    return tsne_3d_results


if __name__ == "__main__":
    # Config
    start_T = 700
    n_steps = 200
    val_idx = 3
    model_name = "CAE_Koopman"
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    forward_model = CYLINDER_C_FORWARD().to(device)
    forward_model.load_state_dict(torch.load(f'../../../../results/{model_name}/Cylinder/model_weights/forward_model.pt', 
                                            weights_only=True, map_location=device))
    forward_model.C_forward = torch.load(f'../../../../results/{model_name}/Cylinder/model_weights/C_forward.pt', 
                                       weights_only=True, map_location=device).to(device)
    forward_model.eval()
    
    # Load data
    cyl_train_dataset = CylinderDynamicsDataset(
        data_path="../../../../data/cylinder/cylinder_train_data.npy",
        seq_length=12, mean=None, std=None)
    
    cyl_val_dataset = CylinderDynamicsDataset(
        data_path="../../../../data/cylinder/cylinder_val_data.npy",
        seq_length=12, mean=cyl_train_dataset.mean, std=cyl_train_dataset.std)
    
    # Get ground truth
    groundtruth = cyl_val_dataset.data[val_idx, start_T:start_T + n_steps, ...]
    groundtruth = torch.from_numpy(groundtruth).to(device)
    normalize_groundtruth = cyl_val_dataset.normalize(groundtruth)
    
    # Generate ground truth latents
    gt_latents = []
    with torch.no_grad():
        for i in range(n_steps):
            z = forward_model.K_S(normalize_groundtruth[i:i+1])
            gt_latents.append(z.squeeze(0))
    
    # Run rollouts
    initial_state = normalize_groundtruth[0:1]
    state_latents = state_propagation_rollout(forward_model, initial_state, n_steps)
    latent_latents = latent_propagation_rollout(forward_model, initial_state, n_steps)
    
    # Plot both 2D and 3D comparisons and get t-SNE results
    save_path_2d = f'../../../../results/{model_name}/rollout_comparison_2d.png'
    save_path_3d = f'../../../../results/{model_name}/rollout_comparison_3d.png'
    
    tsne_2d_results = plot_tsne_2d(gt_latents, state_latents, latent_latents, save_path_2d)
    tsne_3d_results = plot_tsne_3d(gt_latents, state_latents, latent_latents, save_path_3d)
    
    # Save all data including t-SNE results
    saved_pickle_path = save_tsne_data(gt_latents, state_latents, latent_latents, model_name, 
                                      tsne_2d_results, tsne_3d_results)
    
    print(f"2D t-SNE comparison saved to: {save_path_2d}")
    print(f"3D t-SNE comparison saved to: {save_path_3d}")
    print(f"All analysis data saved to: {saved_pickle_path}")