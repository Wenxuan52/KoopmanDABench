import torch
import torch.nn.functional as F
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

if __name__ == '__main__':
    fig_save_path = '../../results/Comparison/figures/'
    
    model_paths = [
        '../../results/DMD/DA/cyl_comp_data.pkl',
        '../../results/CAE_DMD/DA/cyl_comp_data.pkl',
        '../../results/CAE_Linear/DA/cyl_comp_data.pkl',
        '../../results/CAE_Weaklinear/DA/cyl_comp_data.pkl'
    ]

    model_names = ['DMD', 'CAE_DMD', 'CAE_Linear', 'CAE_Weaklinear']
    
    # Define colors for each model
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D']
    
    # Storage for all model data
    all_models_data = {}
    
    # Convert tensors to numpy arrays if needed
    def tensor_to_numpy(data):
        if hasattr(data[0], 'item'):
            return np.array([x.item() for x in data])
        elif hasattr(data[0], 'numpy'):
            return np.array([x.numpy() for x in data])
        else:
            return np.array(data)
    
    # Load data from all models
    for i, (path, name) in enumerate(zip(model_paths, model_names)):
        try:
            with open(path, 'rb') as f:
                loaded_data = pickle.load(f)
            
            # Extract and convert data
            da_mse = tensor_to_numpy(loaded_data['diffs_da_real_mse'])
            noda_mse = tensor_to_numpy(loaded_data['diffs_noda_real_mse'])
            da_rrmse = tensor_to_numpy(loaded_data['diffs_da_real_rrmse'])
            noda_rrmse = tensor_to_numpy(loaded_data['diffs_noda_real_rrmse'])
            da_ssim = np.array(loaded_data['diffs_da_real_ssim'])
            noda_ssim = np.array(loaded_data['diffs_noda_real_ssim'])
            
            all_models_data[name] = {
                'da_mse': da_mse,
                'noda_mse': noda_mse,
                'da_rrmse': da_rrmse,
                'noda_rrmse': noda_rrmse,
                'da_ssim': da_ssim,
                'noda_ssim': noda_ssim,
                'color': colors[i],
                'linestyle': line_styles[i],
                'marker': markers[i]
            }
            print(f"Successfully loaded {name}")
            
        except FileNotFoundError:
            print(f"Warning: Could not find file for {name} at {path}")
            continue
        except Exception as e:
            print(f"Error loading {name}: {e}")
            continue
    
    if not all_models_data:
        print("No data loaded successfully!")
        exit()
    
    # Get time steps (assuming all models have same length)
    first_model = next(iter(all_models_data.values()))
    n_points = len(first_model['da_mse'])
    time_steps = np.arange(n_points)
    
    # Plot 1: Comprehensive comparison for 4D Var performance
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('4D VAR Performance Comparison', fontsize=18, fontweight='bold')
    
    time_steps_offset = time_steps + 700
    
    da_times = [800, 810, 820]
    
    # MSE comparison (4D Var only)
    for name, data in all_models_data.items():
        axes[0, 0].plot(time_steps_offset, data['da_mse'], 
                       color=data['color'], linestyle=data['linestyle'], 
                       marker=data['marker'], label=name, linewidth=2, markersize=4)
    
    for da_time in da_times:
        axes[0, 0].axvline(x=da_time, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].set_title('4D Var - Mean Squared Error')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # RRMSE comparison (4D Var only)
    for name, data in all_models_data.items():
        axes[0, 1].plot(time_steps_offset, data['da_rrmse'], 
                       color=data['color'], linestyle=data['linestyle'], 
                       marker=data['marker'], label=name, linewidth=2, markersize=4)
    
    for da_time in da_times:
        axes[0, 1].axvline(x=da_time, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('RRMSE')
    axes[0, 1].set_title('4D Var - Relative Root Mean Squared Error')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # SSIM comparison (4D Var only)
    for name, data in all_models_data.items():
        axes[1, 0].plot(time_steps_offset, data['da_ssim'], 
                       color=data['color'], linestyle=data['linestyle'], 
                       marker=data['marker'], label=name, linewidth=2, markersize=4)
    
    for da_time in da_times:
        axes[1, 0].axvline(x=da_time, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('SSIM')
    axes[1, 0].set_title('4D Var - Structural Similarity Index')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Improvement ratio (4D Var / No DA)
    for name, data in all_models_data.items():
        mse_ratio = data['da_mse'] / data['noda_mse']
        axes[1, 1].plot(time_steps_offset, mse_ratio, 
                       color=data['color'], linestyle=data['linestyle'], 
                       marker=data['marker'], label=f'{name} MSE', linewidth=2, markersize=4)
    
    for da_time in da_times:
        axes[1, 1].axvline(x=da_time, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
    axes[1, 1].axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Equal Performance')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Ratio (4D Var / No DA)')
    axes[1, 1].set_title('MSE Improvement Ratio')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{fig_save_path}cyl_4dvar_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Average performance bar chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Average Performance Metrics', fontsize=16, fontweight='bold')
    
    metrics = ['MSE', 'RRMSE', 'SSIM']
    metric_keys = [('da_mse', 'noda_mse'), ('da_rrmse', 'noda_rrmse'), ('da_ssim', 'noda_ssim')]
    
    for i, (metric, (da_key, noda_key)) in enumerate(zip(metrics, metric_keys)):
        model_names_list = list(all_models_data.keys())
        da_means = [np.mean(all_models_data[name][da_key]) for name in model_names_list]
        noda_means = [np.mean(all_models_data[name][noda_key]) for name in model_names_list]
        da_stds = [np.std(all_models_data[name][da_key]) for name in model_names_list]
        noda_stds = [np.std(all_models_data[name][noda_key]) for name in model_names_list]
        
        x = np.arange(len(model_names_list))
        width = 0.35
        
        bars1 = axes[i].bar(x - width/2, da_means, width, yerr=da_stds, 
                           capsize=5, label='4D Var', alpha=0.8, color='blue')
        bars2 = axes[i].bar(x + width/2, noda_means, width, yerr=noda_stds, 
                           capsize=5, label='No DA', alpha=0.8, color='red')
        
        axes[i].set_xlabel('Models')
        axes[i].set_ylabel(f'{metric} Value')
        axes[i].set_title(f'{metric} Comparison')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(model_names_list, rotation=45, ha='right')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{fig_save_path}cyl_average_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 3: Improvement percentage comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('4D VAR Improvement Over No DA (%)', fontsize=16, fontweight='bold')
    
    for i, (metric, (da_key, noda_key)) in enumerate(zip(metrics, metric_keys)):
        model_names_list = list(all_models_data.keys())
        improvements = []
        
        for name in model_names_list:
            da_mean = np.mean(all_models_data[name][da_key])
            noda_mean = np.mean(all_models_data[name][noda_key])
            
            if metric == 'SSIM':  # For SSIM, higher is better
                improvement = ((da_mean - noda_mean) / noda_mean * 100)
            else:  # For MSE and RRMSE, lower is better
                improvement = ((noda_mean - da_mean) / noda_mean * 100)
            
            improvements.append(improvement)
        
        bars = axes[i].bar(model_names_list, improvements, 
                          color=[all_models_data[name]['color'] for name in model_names_list],
                          alpha=0.8)
        
        axes[i].set_xlabel('Models')
        axes[i].set_ylabel('Improvement (%)')
        axes[i].set_title(f'{metric} Improvement')
        axes[i].set_xticks(range(len(model_names_list)))
        axes[i].set_xticklabels(model_names_list, rotation=45, ha='right')
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Add value labels on bars
        for bar, improvement in zip(bars, improvements):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -1.5),
                        f'{improvement:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
    
    plt.tight_layout()
    plt.savefig(f'{fig_save_path}cyl_improvement_percentage.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistical summary
    print("="*80)
    print("MULTI-MODEL PERFORMANCE SUMMARY")
    print("="*80)
    
    for name, data in all_models_data.items():
        print(f"\n{name.upper()}:")
        print("-" * 40)
        
        # MSE
        da_mse_mean, da_mse_std = np.mean(data['da_mse']), np.std(data['da_mse'])
        noda_mse_mean, noda_mse_std = np.mean(data['noda_mse']), np.std(data['noda_mse'])
        mse_improvement = ((noda_mse_mean - da_mse_mean) / noda_mse_mean * 100)
        print(f"MSE - 4D Var: {da_mse_mean:.6f}±{da_mse_std:.6f}, No DA: {noda_mse_mean:.6f}±{noda_mse_std:.6f}, Improvement: {mse_improvement:.2f}%")
        
        # RRMSE
        da_rrmse_mean, da_rrmse_std = np.mean(data['da_rrmse']), np.std(data['da_rrmse'])
        noda_rrmse_mean, noda_rrmse_std = np.mean(data['noda_rrmse']), np.std(data['noda_rrmse'])
        rrmse_improvement = ((noda_rrmse_mean - da_rrmse_mean) / noda_rrmse_mean * 100)
        print(f"RRMSE - 4D Var: {da_rrmse_mean:.6f}±{da_rrmse_std:.6f}, No DA: {noda_rrmse_mean:.6f}±{noda_rrmse_std:.6f}, Improvement: {rrmse_improvement:.2f}%")
        
        # SSIM
        da_ssim_mean, da_ssim_std = np.mean(data['da_ssim']), np.std(data['da_ssim'])
        noda_ssim_mean, noda_ssim_std = np.mean(data['noda_ssim']), np.std(data['noda_ssim'])
        ssim_improvement = ((da_ssim_mean - noda_ssim_mean) / noda_ssim_mean * 100)
        print(f"SSIM - 4D Var: {da_ssim_mean:.6f}±{da_ssim_std:.6f}, No DA: {noda_ssim_mean:.6f}±{noda_ssim_std:.6f}, Improvement: {ssim_improvement:.2f}%")
    
    print("\n" + "="*80)