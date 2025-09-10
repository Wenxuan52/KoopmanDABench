import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

import psutil
import time
import pickle

from cylinder_model import CYLINDER_C_FORWARD

import os
import sys
current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
sys.path.append(src_directory)

from src.utils.Dataset import CylinderDynamicsDataset


def visualize_latent_matrix_eigenvalues(forward_model, save_dir="figures", save_name="latent_eigenvalues.png"):
    """
    Visualize eigenvalues of the C_forward matrix to check if they lie near the unit circle
    
    Args:
        forward_model: The trained model containing C_forward matrix
        save_dir: Directory to save the plot
        save_name: Name of the saved plot file
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import os
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract C_forward matrix
    C_forward = forward_model.C_forward.detach().cpu().numpy()
    print(f"C_forward shape: {C_forward.shape}")
    print(f"C_forward norm: {np.linalg.norm(C_forward):.6f}")
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(C_forward)
    print(f"Number of eigenvalues: {len(eigenvalues)}")
    
    # Separate real and imaginary parts
    real_parts = eigenvalues.real
    imag_parts = eigenvalues.imag
    
    # Compute magnitudes
    magnitudes = np.abs(eigenvalues)
    
    # Statistics
    max_magnitude = np.max(magnitudes)
    min_magnitude = np.min(magnitudes)
    mean_magnitude = np.mean(magnitudes)
    std_magnitude = np.std(magnitudes)
    
    # Count eigenvalues near unit circle (within tolerance)
    tolerance = 0.1
    near_unit_circle = np.sum(np.abs(magnitudes - 1.0) < tolerance)
    percentage_near_unit = (near_unit_circle / len(eigenvalues)) * 100
    
    print(f"\nEigenvalue Statistics:")
    print(f"  Max magnitude: {max_magnitude:.6f}")
    print(f"  Min magnitude: {min_magnitude:.6f}")
    print(f"  Mean magnitude: {mean_magnitude:.6f}")
    print(f"  Std magnitude: {std_magnitude:.6f}")
    print(f"  Eigenvalues near unit circle (|λ-1| < {tolerance}): {near_unit_circle}/{len(eigenvalues)} ({percentage_near_unit:.1f}%)")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Eigenvalues in complex plane with unit circle
    ax1 = axes[0]
    
    # Draw unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    unit_circle_real = np.cos(theta)
    unit_circle_imag = np.sin(theta)
    ax1.plot(unit_circle_real, unit_circle_imag, 'k--', linewidth=2, alpha=0.7, label='Unit Circle')
    
    # Plot eigenvalues
    scatter = ax1.scatter(real_parts, imag_parts, c=magnitudes, cmap='viridis', 
                         s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax1.set_xlabel('Real Part', fontsize=12)
    ax1.set_ylabel('Imaginary Part', fontsize=12)
    ax1.set_title('Eigenvalues in Complex Plane', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    ax1.legend()
    
    # Add colorbar
    cbar1 = plt.colorbar(scatter, ax=ax1)
    cbar1.set_label('Magnitude |λ|', fontsize=12)
    
    # Plot 2: Histogram of eigenvalue magnitudes
    ax2 = axes[1]
    n_bins = min(50, len(eigenvalues) // 3)  # Adaptive number of bins
    counts, bins, patches = ax2.hist(magnitudes, bins=n_bins, alpha=0.7, color='skyblue', 
                                    edgecolor='black', linewidth=0.5)
    
    # Highlight unit circle region
    ax2.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Unit Circle (|λ|=1)')
    ax2.axvspan(1.0-tolerance, 1.0+tolerance, alpha=0.2, color='red', 
                label=f'Near Unit Circle (|λ-1|<{tolerance})')
    
    ax2.set_xlabel('Eigenvalue Magnitude |λ|', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Eigenvalue Magnitudes', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add statistics text
    stats_text = f'Mean: {mean_magnitude:.3f}\nStd: {std_magnitude:.3f}\nNear unit: {percentage_near_unit:.1f}%'
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 3: Eigenvalue magnitudes vs index (sorted)
    ax3 = axes[2]
    sorted_magnitudes = np.sort(magnitudes)[::-1]  # Sort in descending order
    indices = np.arange(len(sorted_magnitudes))
    
    ax3.plot(indices, sorted_magnitudes, 'b-', linewidth=2, alpha=0.7, label='Eigenvalue Magnitudes')
    ax3.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Unit Circle (|λ|=1)')
    ax3.axhspan(1.0-tolerance, 1.0+tolerance, alpha=0.2, color='red', 
                label=f'Near Unit Circle')
    
    ax3.set_xlabel('Eigenvalue Index (sorted by magnitude)', fontsize=12)
    ax3.set_ylabel('Eigenvalue Magnitude |λ|', fontsize=12)
    ax3.set_title('Sorted Eigenvalue Magnitudes', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Highlight the largest few eigenvalues
    n_highlight = min(10, len(sorted_magnitudes))
    ax3.scatter(indices[:n_highlight], sorted_magnitudes[:n_highlight], 
                color='orange', s=60, zorder=5, alpha=0.8)
    
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[INFO] Eigenvalue visualization saved to: {save_path}")
    plt.close()
    
    # Additional analysis: spectral radius
    spectral_radius = max_magnitude
    print(f"\nSpectral Analysis:")
    print(f"  Spectral radius: {spectral_radius:.6f}")
    if spectral_radius <= 1.0:
        print("  ✓ Spectral radius ≤ 1: System is stable")
    else:
        print("  ⚠ Spectral radius > 1: System may be unstable")
    
    # Return statistics for further analysis if needed
    eigenvalue_stats = {
        'eigenvalues': eigenvalues,
        'magnitudes': magnitudes,
        'max_magnitude': max_magnitude,
        'min_magnitude': min_magnitude,
        'mean_magnitude': mean_magnitude,
        'std_magnitude': std_magnitude,
        'spectral_radius': spectral_radius,
        'near_unit_circle_count': near_unit_circle,
        'near_unit_circle_percentage': percentage_near_unit,
        'tolerance': tolerance
    }
    
    return eigenvalue_stats


# Add this section after loading the model and before starting inference

if __name__ == '__main__':
    fig_save_path = '../../../../results/CAE_MI/figures/'
    
    forward_model = CYLINDER_C_FORWARD()
    forward_model.load_state_dict(torch.load('../../../../results/CAE_MI/Cylinder/cylinder_model_weights_L512/forward_model.pt', weights_only=True, map_location='cpu'))
    forward_model.C_forward = torch.load('../../../../results/CAE_MI/Cylinder/cylinder_model_weights_L512/C_forward.pt', weights_only=True, map_location='cpu')
    forward_model.eval()

    print(torch.norm(forward_model.C_forward))
    print(forward_model)

    # Add eigenvalue visualization here
    print("\n" + "="*50)
    print("LATENT MATRIX EIGENVALUE ANALYSIS")
    print("="*50)
    
    eigenvalue_stats = visualize_latent_matrix_eigenvalues(
        forward_model, 
        save_dir=fig_save_path, 
        save_name="cyl_latent_eigenvalues_L512.png"
    )
    
    # Save eigenvalue statistics
    eigenvalue_save_path = os.path.join(fig_save_path, 'cyl_eigenvalue_stats.pkl')
    with open(eigenvalue_save_path, 'wb') as f:
        pickle.dump(eigenvalue_stats, f)
    print(f"[INFO] Eigenvalue statistics saved to: {eigenvalue_save_path}")
    
    print("\n" + "="*50)
    print("STARTING INFERENCE EVALUATION")  
    print("="*50)
    
    # ... (rest of your existing code continues from here)