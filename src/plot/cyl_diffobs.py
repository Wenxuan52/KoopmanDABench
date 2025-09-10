#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_multi_model_sparse_comparison():
    """Plot 2x3 grid comparing all models with different observation densities"""
    
    # Model configurations
    models = [
        ("DMD", "DMD"),
        ("CAE_DMD", "DMD ROM"), 
        ("CAE_Koopman", "Koopman ROM"),
        ("CAE_Linear", "Linear ROM"),
        ("CAE_Weaklinear", "Weaklinear ROM"),
        ("CAE_MLP", "MLP ROM")
    ]
    
    # Create 2x3 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    # fig.suptitle('Sparse Observation Data Assimilation Performance Comparison', fontsize=28, fontweight='bold', y=0.99)
    
    # Common variables for legend
    target_ratios = None
    blues = None
    step_idxs = None
    time_obs = None
    
    # Plot each model
    for idx, (model_name, model_display_name) in enumerate(models):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Load results for current model
        results_path = f'../../results/{model_name}/DA/cyl_diffobs_results.pkl'
        
        try:
            with open(results_path, 'rb') as f:
                all_results = pickle.load(f)
            
            # Extract common information from first model for legend
            if target_ratios is None:
                target_ratios = [result['target_obs_ratio'] for result in all_results]
                n_ratios = len(target_ratios)
                blues = plt.cm.Blues(np.linspace(1.0, 0.4, n_ratios))  # Dark to light blue
                
                start_da_end_idxs = all_results[0]['start_da_end_idxs']
                step_idxs = list(range(start_da_end_idxs[0] + 1, start_da_end_idxs[-1] + 2))
                time_obs = all_results[0]['time_obs']
            
            # Plot MSE for different observation densities
            for i, result in enumerate(all_results):
                target_ratio = result['target_obs_ratio']
                color = blues[i]
                
                ax.plot(step_idxs, result['diffs_da_real_mse'], 
                       color=color, linewidth=3, alpha=0.7,
                       marker='o', markersize=2, markevery=5)
            
            # Plot baseline (No DA) for reference
            ax.plot(step_idxs, all_results[0]['diffs_noda_real_mse'], 
                   color='red', linewidth=2, linestyle='--', alpha=0.8)
            
            # Add observation time markers
            for x in time_obs:
                ax.axvline(x=x+1, color="black", linestyle=":", alpha=0.7, linewidth=1.5)
            
        except FileNotFoundError:
            print(f"Warning: Could not find file for {model_name}")
            ax.text(0.5, 0.5, f'{model_display_name}\nData Not Available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            continue
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            continue
        
        # Customize subplot
        ax.set_xlabel('Time Step', fontsize=18)
        ax.set_ylabel('MSE', fontsize=18)
        ax.set_title(f'{model_display_name}', fontsize=20, fontweight='bold')
        ax.tick_params(labelsize=16)
        ax.grid(True, alpha=0.5)
        
        # Remove individual legends
        # ax.legend() is not called
        
        # Enhance aesthetics
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
    
    # Create unified legend at the bottom
    if target_ratios is not None and blues is not None:
        legend_elements = []
        
        # Add observation density lines
        for i, target_ratio in enumerate(target_ratios):
            legend_elements.append(
                plt.Line2D([0], [0], color=blues[i], linewidth=3, 
                          label=f'{target_ratio:.1%} Observations')
            )
        
        # Add baseline line
        legend_elements.append(
            plt.Line2D([0], [0], color='red', linewidth=2, linestyle='--',
                      label='No DA (Baseline)')
        )
        
        # Add legend below the plots
        fig.legend(handles=legend_elements, loc='lower center', 
                bbox_to_anchor=(0.5, -0.02), ncol=len(legend_elements), 
                fontsize=20, frameon=True)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.10, hspace=0.3, wspace=0.3)
    
    # Save plot
    save_path = '../../results/Comparison/figures/multi_model_sparse_comparison.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
    
    print(f"Multi-model sparse observation comparison plot saved to: {save_path}")

if __name__ == "__main__":
    plot_multi_model_sparse_comparison()