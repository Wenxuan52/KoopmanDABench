#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_multi_model_noise_comparison():
    """Plot 2x3 grid comparing all models with different observation noise levels"""
    
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
    # fig.suptitle('Observation Noise Data Assimilation Performance Comparison', fontsize=28, fontweight='bold', y=0.99)
    
    # Common variables for legend
    noise_stds = None
    plasmas = None
    step_idxs = None
    time_obs = None
    
    # Plot each model
    for idx, (model_name, model_display_name) in enumerate(models):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Load results for current model
        results_path = f'../../results/{model_name}/DA/cyl_diffnoise_results.pkl'
        
        try:
            with open(results_path, 'rb') as f:
                all_results = pickle.load(f)
            
            # Extract common information from first model for legend
            if noise_stds is None:
                noise_stds = [result['noise_std'] for result in all_results]
                n_levels = len(noise_stds)
                plasmas = plt.cm.plasma(np.linspace(0.1, 0.9, n_levels))  # Bright to dark plasma
                
                start_da_end_idxs = all_results[0]['start_da_end_idxs']
                step_idxs = list(range(start_da_end_idxs[0] + 1, start_da_end_idxs[-1] + 2))
                time_obs = all_results[0]['time_obs']
            
            # Plot MSE for different noise levels
            for i, result in enumerate(all_results):
                noise_std = result['noise_std']
                color = plasmas[i]
                
                ax.plot(step_idxs, result['diffs_da_real_mse'], 
                       color=color, linewidth=3, alpha=0.9,
                       marker='o', markersize=2, markevery=5)
            
            # Plot baseline (No DA) for reference
            ax.plot(step_idxs, all_results[0]['diffs_noda_real_mse'], 
                   color='black', linewidth=2, linestyle='--', alpha=0.8)
            
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
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=16)
        
        # Remove individual legends
        # ax.legend() is not called
        
        # Enhance aesthetics
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
    
    # Create unified legend at the bottom
    if noise_stds is not None and plasmas is not None:
        legend_elements = []
        
        # Add noise level lines
        for i, noise_std in enumerate(noise_stds):
            if noise_std == 0:
                label = 'No Noise (σ = 0)'
            else:
                label = f'σ = {noise_std:.4f}'
            
            legend_elements.append(
                plt.Line2D([0], [0], color=plasmas[i], linewidth=3, label=label)
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
    save_path = '../../results/Comparison/figures/multi_model_noise_comparison.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
    
    print(f"Multi-model noise comparison plot saved to: {save_path}")


def analyze_noise_experiments(model_name="CAE_Koopman"):
    """Analyze noise level experimental results for a single model"""
    
    results_path = f'../../results/{model_name}/DA/cyl_diffnoise_results.pkl'
    
    try:
        with open(results_path, 'rb') as f:
            all_results = pickle.load(f)
        print(f"Successfully loaded results for {model_name}")
        print(f"Contains {len(all_results)} noise level experiments")
    except FileNotFoundError:
        print(f"Could not find file for {model_name}")
        return None
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None
    
    print(f"\n{'='*80}")
    print(f"NOISE LEVEL EXPERIMENTS SUMMARY - {model_name}")
    print(f"{'='*80}")
    
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
    
    print("Noise level experiments analysis completed!")
    return all_results


if __name__ == "__main__":
    # Plot multi-model comparison
    print("Generating multi-model noise comparison plot...")
    plot_multi_model_noise_comparison()

    # print("\nAnalyzing individual model results...")
    # for model_name in ["DMD", "CAE_DMD", "CAE_Koopman", "CAE_Linear", "CAE_Weaklinear", "CAE_MLP"]:
    #     analyze_noise_experiments(model_name)