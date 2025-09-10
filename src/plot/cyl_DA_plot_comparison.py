import torch
import torch.nn.functional as F
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

if __name__ == '__main__':
    lw = 3.0

    fig_save_path = '../../results/Comparison/figures/'
    
    model_paths = [
        '../../results/DMD/DA/cyl_comp_data.pkl',
        '../../results/CAE_DMD/DA/cyl_comp_data.pkl',
        '../../results/CAE_Koopman/DA/cyl_comp_data.pkl',
        '../../results/CAE_Linear/DA/cyl_comp_data.pkl',
        '../../results/CAE_Weaklinear/DA/cyl_comp_data.pkl',
        '../../results/CAE_MLP/DA/cyl_comp_data.pkl'
    ]

    model_names = ['DMD', 'DMD ROM', 'Koopman ROM', 'Linear ROM', 'Weaklinear ROM', 'MLP ROM']
    
    # Define colors and styles for each model (expanded for 6 models)
    colors = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#e6c229"]  # 6 colors
    line_styles = ['-', '-', '-', '-', '-', '-']  # 6 line styles
    markers = ['o', 's', '^', 'D', 'v', 'p']  # 6 markers
    
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
                'color': colors[i % len(colors)],  # Use modulo to handle more models
                'linestyle': line_styles[i % len(line_styles)],
                'marker': markers[i % len(markers)]
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
    time_steps_offset = time_steps + 700
    
    da_times = [800, 810, 820]
    
    # Plot 1: Comprehensive comparison for 4D Var performance (1x4 layout)
    fig, axes = plt.subplots(1, 4, figsize=(26, 7))  # Increased figure height for legend
    fig.suptitle('Cylinder Flow 4D-Var Performance', fontsize=28, fontweight='bold', y=0.98)
    
    # MSE comparison (4D Var only)
    for name, data in all_models_data.items():
        axes[0].plot(time_steps_offset, data['da_mse'], 
                    color=data['color'], linestyle=data['linestyle'], 
                    marker=data['marker'], label=name, linewidth=lw, markersize=2, alpha=0.9)
    
    for da_time in da_times:
        axes[0].axvline(x=da_time, color='black', linestyle='--', alpha=0.8, linewidth=1.5)
    axes[0].set_xlabel('Time Step', fontsize=16)
    axes[0].set_ylabel('MSE', fontsize=16)
    axes[0].set_title('MSE', fontsize=22, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(labelsize=14)
    
    # RRMSE comparison (4D Var only)
    for name, data in all_models_data.items():
        axes[1].plot(time_steps_offset, data['da_rrmse'], 
                    color=data['color'], linestyle=data['linestyle'], 
                    marker=data['marker'], label=name, linewidth=lw, markersize=2, alpha=0.9)
    
    for da_time in da_times:
        axes[1].axvline(x=da_time, color='black', linestyle='--', alpha=0.8, linewidth=1.5)
    axes[1].set_xlabel('Time Step', fontsize=16)
    axes[1].set_ylabel('RRMSE', fontsize=16)
    axes[1].set_title('RRMSE', fontsize=22, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(labelsize=14)
    
    # SSIM comparison (4D Var only)
    for name, data in all_models_data.items():
        axes[2].plot(time_steps_offset, data['da_ssim'], 
                    color=data['color'], linestyle=data['linestyle'], 
                    marker=data['marker'], label=name, linewidth=lw, markersize=2, alpha=0.9)
    
    for da_time in da_times:
        axes[2].axvline(x=da_time, color='black', linestyle='--', alpha=0.8, linewidth=1.5)
    axes[2].set_xlabel('Time Step', fontsize=16)
    axes[2].set_ylabel('SSIM', fontsize=16)
    axes[2].set_title('SSIM', fontsize=22, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].tick_params(labelsize=14)
    
    # Improvement ratio (4D Var / No DA)
    for name, data in all_models_data.items():
        mse_ratio = data['da_mse'] / data['noda_mse']
        axes[3].plot(time_steps_offset, mse_ratio, 
                    color=data['color'], linestyle=data['linestyle'], 
                    marker=data['marker'], label=f'{name}', linewidth=lw, markersize=2, alpha=0.9)
    
    for da_time in da_times:
        axes[3].axvline(x=da_time, color='black', linestyle='--', alpha=0.8, linewidth=1.5)
    axes[3].axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Equal Performance')
    axes[3].set_xlabel('Time Step', fontsize=16)
    axes[3].set_ylabel('Ratio (4D Var / No DA)', fontsize=16)
    axes[3].set_title('MSE Improvement Ratio', fontsize=22, fontweight='bold')
    axes[3].grid(True, alpha=0.3)
    axes[3].tick_params(labelsize=14)
    
    # Add unified legend below all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
               ncol=len(model_names), fontsize=24, frameon=True)
    
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for legend
    plt.savefig(f'{fig_save_path}cyl_4dvar_comparison.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Average performance bar chart (1x3 layout)
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))  # Increased figure size
    fig.suptitle('Average Performance Metrics Comparison', fontsize=22, fontweight='bold', y=0.98)
    
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
                           capsize=5, label='4D Var', alpha=0.8, color='#2E86AB')
        bars2 = axes[i].bar(x + width/2, noda_means, width, yerr=noda_stds, 
                           capsize=5, label='No DA', alpha=0.8, color='#F24236')
        
        axes[i].set_xlabel('Models', fontsize=16)
        axes[i].set_ylabel(f'{metric} Value', fontsize=16)
        axes[i].set_title(f'{metric} Comparison', fontsize=18, fontweight='bold')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(model_names_list, rotation=45, ha='right', fontsize=14)
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(labelsize=14)
    
    # Add unified legend for bar charts
    fig.legend(['4D Var', 'No DA'], loc='lower center', bbox_to_anchor=(0.5, -0.05), 
               ncol=2, fontsize=16, frameon=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for legend
    plt.savefig(f'{fig_save_path}cyl_average_performance.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    # Plot 3: Improvement percentage comparison (1x3 layout)
    # Focus on the first DA time frame (background field improvement)
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))  # Increased figure size
    fig.suptitle('4D VAR Background Field Improvement Over No DA (%)\n(First DA Time Frame Analysis)', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Find the index of first DA time (800 - 700 = 100th index)
    first_da_index = 800 - 700  # Convert to array index (assuming time_steps_offset starts from 700)
    
    for i, (metric, (da_key, noda_key)) in enumerate(zip(metrics, metric_keys)):
        model_names_list = list(all_models_data.keys())
        improvements = []
        
        for name in model_names_list:
            # Get values at the first DA time frame only
            da_value = all_models_data[name][da_key][first_da_index]
            noda_value = all_models_data[name][noda_key][first_da_index]
            
            if metric == 'SSIM':  # For SSIM, higher is better
                improvement = ((da_value - noda_value) / noda_value * 100)
            else:  # For MSE and RRMSE, lower is better
                improvement = ((noda_value - da_value) / noda_value * 100)
            
            improvements.append(improvement)
        
        bars = axes[i].bar(model_names_list, improvements, 
                          color=[all_models_data[name]['color'] for name in model_names_list],
                          alpha=0.4, edgecolor='black', linewidth=1)
        
        axes[i].set_xlabel('Models', fontsize=16)
        axes[i].set_ylabel('Improvement (%)', fontsize=16)
        axes[i].set_title(f'{metric} Background Field Improvement\n(t=800)', fontsize=18, fontweight='bold')
        axes[i].set_xticks(range(len(model_names_list)))
        axes[i].set_xticklabels(model_names_list, rotation=45, ha='right', fontsize=14)
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[i].tick_params(labelsize=14)
        
        # Add value labels on bars
        for bar, improvement in zip(bars, improvements):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -1.5),
                        f'{improvement:.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                        fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{fig_save_path}cyl_background_field_improvement.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    # Additional Plot: Average improvement across all DA windows
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))  # Increased figure size
    fig.suptitle('4D VAR Average Improvement Over No DA (%)\n(Entire Time Series)', 
                 fontsize=20, fontweight='bold', y=0.98)
    
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
                          alpha=0.4, edgecolor='black', linewidth=1)
        
        axes[i].set_xlabel('Models', fontsize=16)
        axes[i].set_ylabel('Improvement (%)', fontsize=16)
        axes[i].set_title(f'{metric} Average Improvement', fontsize=18, fontweight='bold')
        axes[i].set_xticks(range(len(model_names_list)))
        axes[i].set_xticklabels(model_names_list, rotation=45, ha='right', fontsize=14)
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[i].tick_params(labelsize=14)
        
        # Add value labels on bars
        for bar, improvement in zip(bars, improvements):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -1.5),
                        f'{improvement:.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                        fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{fig_save_path}cyl_average_improvement.png', dpi=100, bbox_inches='tight')
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
    
    # Enhanced summary: Background field improvement analysis
    print("\nBACKGROUND FIELD IMPROVEMENT ANALYSIS (First DA Time Frame - t=800):")
    print("=" * 70)
    
    first_da_index = 800 - 700  # Convert to array index
    
    for name, data in all_models_data.items():
        print(f"\n{name.upper()}:")
        print("-" * 40)
        
        # Background field values at t=800
        da_mse_bg = data['da_mse'][first_da_index]
        noda_mse_bg = data['noda_mse'][first_da_index]
        mse_bg_improvement = ((noda_mse_bg - da_mse_bg) / noda_mse_bg * 100)
        
        da_rrmse_bg = data['da_rrmse'][first_da_index]
        noda_rrmse_bg = data['noda_rrmse'][first_da_index]
        rrmse_bg_improvement = ((noda_rrmse_bg - da_rrmse_bg) / noda_rrmse_bg * 100)
        
        da_ssim_bg = data['da_ssim'][first_da_index]
        noda_ssim_bg = data['noda_ssim'][first_da_index]
        ssim_bg_improvement = ((da_ssim_bg - noda_ssim_bg) / noda_ssim_bg * 100)
        
        print(f"Background Field MSE - 4D Var: {da_mse_bg:.6f}, No DA: {noda_mse_bg:.6f}, Improvement: {mse_bg_improvement:.2f}%")
        print(f"Background Field RRMSE - 4D Var: {da_rrmse_bg:.6f}, No DA: {noda_rrmse_bg:.6f}, Improvement: {rrmse_bg_improvement:.2f}%")
        print(f"Background Field SSIM - 4D Var: {da_ssim_bg:.6f}, No DA: {noda_ssim_bg:.6f}, Improvement: {ssim_bg_improvement:.2f}%")
    
    # Best performing model for background field
    print("\nBEST BACKGROUND FIELD PERFORMANCE:")
    print("=" * 45)
    
    bg_performances = {}
    for name, data in all_models_data.items():
        bg_performances[name] = {
            'mse': data['da_mse'][first_da_index],
            'rrmse': data['da_rrmse'][first_da_index], 
            'ssim': data['da_ssim'][first_da_index]
        }
    
    best_bg_mse = min(bg_performances.keys(), key=lambda x: bg_performances[x]['mse'])
    best_bg_rrmse = min(bg_performances.keys(), key=lambda x: bg_performances[x]['rrmse'])
    best_bg_ssim = max(bg_performances.keys(), key=lambda x: bg_performances[x]['ssim'])
    
    print(f"Best Background MSE: {best_bg_mse} ({bg_performances[best_bg_mse]['mse']:.6f})")
    print(f"Best Background RRMSE: {best_bg_rrmse} ({bg_performances[best_bg_rrmse]['rrmse']:.6f})")
    print(f"Best Background SSIM: {best_bg_ssim} ({bg_performances[best_bg_ssim]['ssim']:.6f})")
    
    # Background field improvement ranking
    bg_improvements = {}
    for name, data in all_models_data.items():
        mse_improvement = ((data['noda_mse'][first_da_index] - data['da_mse'][first_da_index]) / data['noda_mse'][first_da_index] * 100)
        rrmse_improvement = ((data['noda_rrmse'][first_da_index] - data['da_rrmse'][first_da_index]) / data['noda_rrmse'][first_da_index] * 100)
        ssim_improvement = ((data['da_ssim'][first_da_index] - data['noda_ssim'][first_da_index]) / data['noda_ssim'][first_da_index] * 100)
        
        bg_improvements[name] = {
            'mse': mse_improvement,
            'rrmse': rrmse_improvement,
            'ssim': ssim_improvement
        }
    
    print(f"\nBest Background Field Improvement:")
    best_bg_mse_imp = max(bg_improvements.keys(), key=lambda x: bg_improvements[x]['mse'])
    best_bg_rrmse_imp = max(bg_improvements.keys(), key=lambda x: bg_improvements[x]['rrmse'])
    best_bg_ssim_imp = max(bg_improvements.keys(), key=lambda x: bg_improvements[x]['ssim'])
    
    print(f"MSE: {best_bg_mse_imp} ({bg_improvements[best_bg_mse_imp]['mse']:.2f}% improvement)")
    print(f"RRMSE: {best_bg_rrmse_imp} ({bg_improvements[best_bg_rrmse_imp]['rrmse']:.2f}% improvement)")
    print(f"SSIM: {best_bg_ssim_imp} ({bg_improvements[best_bg_ssim_imp]['ssim']:.2f}% improvement)")
    
    print("\n" + "="*80)