import torch
import torch.nn.functional as F
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import os

if __name__ == '__main__':
    lw = 1.5

    fig_save_path = '../../results/Comparison/figures/'
    os.makedirs(fig_save_path, exist_ok=True)
    
    model_paths = [
        '../../results/CAE_DMD/DA/era5_channel_wise_comp_data.pkl',
        '../../results/DMD/DA/era5_channel_wise_comp_data.pkl',
        '../../results/CAE_Koopman/DA/era5_channel_wise_comp_data.pkl',
        '../../results/CAE_Linear/DA/era5_channel_wise_comp_data.pkl',
        '../../results/CAE_Weaklinear/DA/era5_channel_wise_comp_data.pkl',
        '../../results/CAE_MLP/DA/era5_channel_wise_comp_data.pkl'
    ]

    model_names = ['DMD', 'DMD ROM', 'Koopman ROM', 'Linear ROM', 'Weaklinear ROM', 'MLP ROM']
    
    # Define colors and styles for each model
    colors = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f"]
    line_styles = ['-', '--', '-.', ':', '-', '--']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    channel_names = ['Geopotential', 'Temperature', 'Humidity', 'U_wind', 'V_wind']
    channel_labels = ['Geopotential', 'Temperature', 'Humidity', 'Wind (u)', 'Wind (v)']
    
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
            
            print(f"Loading {name}...")
            print(f"Available keys: {loaded_data.keys()}")
            
            # Extract and convert data for each channel
            model_data = {}
            for ch_idx, ch_name in enumerate(channel_names):
                ch_key = f'channel_{ch_idx}'  # channel_0, channel_1, etc.
                
                model_data[ch_name] = {
                    'da_mse': tensor_to_numpy(loaded_data['diffs_da_real_mse_channels'][ch_key]),
                    'noda_mse': tensor_to_numpy(loaded_data['diffs_noda_real_mse_channels'][ch_key]),
                    'da_rrmse': tensor_to_numpy(loaded_data['diffs_da_real_rrmse_channels'][ch_key]),
                    'noda_rrmse': tensor_to_numpy(loaded_data['diffs_noda_real_rrmse_channels'][ch_key]),
                    'da_ssim': np.array(loaded_data['diffs_da_real_ssim_channels'][ch_key]),
                    'noda_ssim': np.array(loaded_data['diffs_noda_real_ssim_channels'][ch_key])
                }
                
                print(f"  {ch_name}: DA MSE length = {len(model_data[ch_name]['da_mse'])}")
            
            all_models_data[name] = {
                'channels': model_data,
                'color': colors[i % len(colors)],
                'linestyle': line_styles[i % len(line_styles)],
                'marker': markers[i % len(markers)]
            }
            print(f"Successfully loaded {name}")
            
        except FileNotFoundError:
            print(f"Warning: Could not find file for {name} at {path}")
            continue
        except Exception as e:
            print(f"Error loading {name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_models_data:
        print("No data loaded successfully!")
        exit()
    
    # Get time steps (assuming all models have same length)
    first_model = next(iter(all_models_data.values()))
    first_channel = next(iter(first_model['channels'].values()))
    n_points = len(first_channel['da_mse'])
    time_steps = np.arange(n_points)
    time_steps_offset = time_steps + 1000  # ERA5 starts from 1000
    
    print(f"Total time steps: {n_points}")
    print(f"Time range: {time_steps_offset[0]} to {time_steps_offset[-1]}")
    
    # DA times for ERA5 (adjust according to your setup)
    da_times = [1010, 1020, 1030]
    da_times = [t for t in da_times if t <= time_steps_offset[-1]]
    
    print(f"DA times: {da_times}")
    
    # Process each channel separately
    for ch_idx, ch_name in enumerate(channel_names):
        ch_label = channel_labels[ch_idx]
        print(f"\nProcessing {ch_label} channel...")
        
        # Plot 1: Comprehensive comparison for 4D Var performance (1x4 layout)
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        fig.suptitle(f'ERA5 {ch_label} - 4D VAR Performance Comparison', fontsize=20, fontweight='bold', y=1.02)
        
        # MSE comparison (4D Var only)
        for name, data in all_models_data.items():
            axes[0].plot(time_steps_offset, data['channels'][ch_name]['da_mse'], 
                        color=data['color'], linestyle=data['linestyle'], 
                        marker=data['marker'], label=name, linewidth=lw, markersize=2, alpha=0.8)
        
        for da_time in da_times:
            axes[0].axvline(x=da_time, color='black', linestyle='--', alpha=0.8, linewidth=1.5)
        axes[0].set_xlabel('Time Step', fontsize=12)
        axes[0].set_ylabel('MSE', fontsize=12)
        axes[0].set_title('4D Var - MSE', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10, loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # RRMSE comparison (4D Var only)
        for name, data in all_models_data.items():
            axes[1].plot(time_steps_offset, data['channels'][ch_name]['da_rrmse'], 
                        color=data['color'], linestyle=data['linestyle'], 
                        marker=data['marker'], label=name, linewidth=lw, markersize=2, alpha=0.8)
        
        for da_time in da_times:
            axes[1].axvline(x=da_time, color='black', linestyle='--', alpha=0.8, linewidth=1.5)
        axes[1].set_xlabel('Time Step', fontsize=12)
        axes[1].set_ylabel('RRMSE', fontsize=12)
        axes[1].set_title('4D Var - RRMSE', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10, loc='best')
        axes[1].grid(True, alpha=0.3)
        
        # SSIM comparison (4D Var only)
        for name, data in all_models_data.items():
            axes[2].plot(time_steps_offset, data['channels'][ch_name]['da_ssim'], 
                        color=data['color'], linestyle=data['linestyle'], 
                        marker=data['marker'], label=name, linewidth=lw, markersize=2, alpha=0.8)
        
        for da_time in da_times:
            axes[2].axvline(x=da_time, color='black', linestyle='--', alpha=0.8, linewidth=1.5)
        axes[2].set_xlabel('Time Step', fontsize=12)
        axes[2].set_ylabel('SSIM', fontsize=12)
        axes[2].set_title('4D Var - SSIM', fontsize=14, fontweight='bold')
        axes[2].legend(fontsize=10, loc='best')
        axes[2].grid(True, alpha=0.3)
        
        # Improvement ratio (4D Var / No DA)
        for name, data in all_models_data.items():
            mse_ratio = data['channels'][ch_name]['da_mse'] / data['channels'][ch_name]['noda_mse']
            axes[3].plot(time_steps_offset, mse_ratio, 
                        color=data['color'], linestyle=data['linestyle'], 
                        marker=data['marker'], label=f'{name}', linewidth=lw, markersize=2, alpha=0.8)
        
        for da_time in da_times:
            axes[3].axvline(x=da_time, color='black', linestyle='--', alpha=0.8, linewidth=1.5)
        axes[3].axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Equal Performance')
        axes[3].set_xlabel('Time Step', fontsize=12)
        axes[3].set_ylabel('Ratio (4D Var / No DA)', fontsize=12)
        axes[3].set_title('MSE Improvement Ratio', fontsize=14, fontweight='bold')
        axes[3].legend(fontsize=10, loc='best')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{fig_save_path}era5_{ch_name}_4dvar_comparison.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Plot 2: Average performance bar chart (1x3 layout)
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        fig.suptitle(f'ERA5 {ch_label} - Average Performance Metrics Comparison', fontsize=18, fontweight='bold')
        
        metrics = ['MSE', 'RRMSE', 'SSIM']
        metric_keys = [('da_mse', 'noda_mse'), ('da_rrmse', 'noda_rrmse'), ('da_ssim', 'noda_ssim')]
        
        for i, (metric, (da_key, noda_key)) in enumerate(zip(metrics, metric_keys)):
            model_names_list = list(all_models_data.keys())
            da_means = [np.mean(all_models_data[name]['channels'][ch_name][da_key]) for name in model_names_list]
            noda_means = [np.mean(all_models_data[name]['channels'][ch_name][noda_key]) for name in model_names_list]
            da_stds = [np.std(all_models_data[name]['channels'][ch_name][da_key]) for name in model_names_list]
            noda_stds = [np.std(all_models_data[name]['channels'][ch_name][noda_key]) for name in model_names_list]
            
            x = np.arange(len(model_names_list))
            width = 0.35
            
            bars1 = axes[i].bar(x - width/2, da_means, width, yerr=da_stds, 
                               capsize=5, label='4D Var', alpha=0.8, color='#2E86AB')
            bars2 = axes[i].bar(x + width/2, noda_means, width, yerr=noda_stds, 
                               capsize=5, label='No DA', alpha=0.8, color='#F24236')
            
            axes[i].set_xlabel('Models', fontsize=12)
            axes[i].set_ylabel(f'{metric} Value', fontsize=12)
            axes[i].set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(model_names_list, rotation=45, ha='right', fontsize=10)
            axes[i].legend(fontsize=11)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{fig_save_path}era5_{ch_name}_average_performance.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Plot 3: Improvement percentage comparison (1x3 layout)
        if da_times:
            fig, axes = plt.subplots(1, 3, figsize=(20, 7))
            fig.suptitle(f'ERA5 {ch_label} - 4D VAR Background Field Improvement Over No DA (%)\n(First DA Time Frame Analysis)', fontsize=18, fontweight='bold')
            
            # Find the index of first DA time
            first_da_index = da_times[0] - 1000  # Convert to array index
            
            if first_da_index < n_points:
                for i, (metric, (da_key, noda_key)) in enumerate(zip(metrics, metric_keys)):
                    model_names_list = list(all_models_data.keys())
                    improvements = []
                    
                    for name in model_names_list:
                        da_value = all_models_data[name]['channels'][ch_name][da_key][first_da_index]
                        noda_value = all_models_data[name]['channels'][ch_name][noda_key][first_da_index]
                        
                        if metric == 'SSIM':  # For SSIM, higher is better
                            improvement = ((da_value - noda_value) / noda_value * 100)
                        else:  # For MSE and RRMSE, lower is better
                            improvement = ((noda_value - da_value) / noda_value * 100)
                        
                        improvements.append(improvement)
                    
                    bars = axes[i].bar(model_names_list, improvements, 
                                      color=[all_models_data[name]['color'] for name in model_names_list],
                                      alpha=0.4, edgecolor='black', linewidth=1)
                    
                    axes[i].set_xlabel('Models', fontsize=12)
                    axes[i].set_ylabel('Improvement (%)', fontsize=12)
                    axes[i].set_title(f'{metric} Background Field Improvement\n(t={da_times[0]})', fontsize=14, fontweight='bold')
                    axes[i].set_xticks(range(len(model_names_list)))
                    axes[i].set_xticklabels(model_names_list, rotation=45, ha='right', fontsize=10)
                    axes[i].grid(True, alpha=0.3)
                    axes[i].axhline(y=0, color='k', linestyle='-', alpha=0.3)
                    
                    # Add value labels on bars
                    for bar, improvement in zip(bars, improvements):
                        height = bar.get_height()
                        axes[i].text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -1.5),
                                    f'{improvement:.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                                    fontsize=10, fontweight='bold')
                
                plt.tight_layout()
                plt.savefig(f'{fig_save_path}era5_{ch_name}_background_field_improvement.png', dpi=300, bbox_inches='tight')
                plt.close(fig)
        
        # Plot 4: Average improvement across all DA windows
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        fig.suptitle(f'ERA5 {ch_label} - 4D VAR Average Improvement Over No DA (%)\n(Entire Time Series)', fontsize=18, fontweight='bold')
        
        for i, (metric, (da_key, noda_key)) in enumerate(zip(metrics, metric_keys)):
            model_names_list = list(all_models_data.keys())
            improvements = []
            
            for name in model_names_list:
                da_mean = np.mean(all_models_data[name]['channels'][ch_name][da_key])
                noda_mean = np.mean(all_models_data[name]['channels'][ch_name][noda_key])
                
                if metric == 'SSIM':  # For SSIM, higher is better
                    improvement = ((da_mean - noda_mean) / noda_mean * 100)
                else:  # For MSE and RRMSE, lower is better
                    improvement = ((noda_mean - da_mean) / noda_mean * 100)
                
                improvements.append(improvement)
            
            bars = axes[i].bar(model_names_list, improvements, 
                              color=[all_models_data[name]['color'] for name in model_names_list],
                              alpha=0.4, edgecolor='black', linewidth=1)
            
            axes[i].set_xlabel('Models', fontsize=12)
            axes[i].set_ylabel('Improvement (%)', fontsize=12)
            axes[i].set_title(f'{metric} Average Improvement', fontsize=14, fontweight='bold')
            axes[i].set_xticks(range(len(model_names_list)))
            axes[i].set_xticklabels(model_names_list, rotation=45, ha='right', fontsize=10)
            axes[i].grid(True, alpha=0.3)
            axes[i].axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            # Add value labels on bars
            for bar, improvement in zip(bars, improvements):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -1.5),
                            f'{improvement:.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                            fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{fig_save_path}era5_{ch_name}_average_improvement.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Completed analysis for {ch_label}")
    
    # Summary statistics for all channels
    print("="*100)
    print("ERA5 MULTI-CHANNEL MULTI-MODEL PERFORMANCE SUMMARY")
    print("="*100)
    
    for ch_idx, ch_name in enumerate(channel_names):
        ch_label = channel_labels[ch_idx]
        print(f"\n{'='*50}")
        print(f"CHANNEL: {ch_label.upper()}")
        print(f"{'='*50}")
        
        for name, data in all_models_data.items():
            print(f"\n{name.upper()}:")
            print("-" * 40)
            
            ch_data = data['channels'][ch_name]
            
            # MSE
            da_mse_mean, da_mse_std = np.mean(ch_data['da_mse']), np.std(ch_data['da_mse'])
            noda_mse_mean, noda_mse_std = np.mean(ch_data['noda_mse']), np.std(ch_data['noda_mse'])
            mse_improvement = ((noda_mse_mean - da_mse_mean) / noda_mse_mean * 100)
            print(f"MSE - 4D Var: {da_mse_mean:.6f}±{da_mse_std:.6f}, No DA: {noda_mse_mean:.6f}±{noda_mse_std:.6f}, Improvement: {mse_improvement:.2f}%")
            
            # RRMSE
            da_rrmse_mean, da_rrmse_std = np.mean(ch_data['da_rrmse']), np.std(ch_data['da_rrmse'])
            noda_rrmse_mean, noda_rrmse_std = np.mean(ch_data['noda_rrmse']), np.std(ch_data['noda_rrmse'])
            rrmse_improvement = ((noda_rrmse_mean - da_rrmse_mean) / noda_rrmse_mean * 100)
            print(f"RRMSE - 4D Var: {da_rrmse_mean:.6f}±{da_rrmse_std:.6f}, No DA: {noda_rrmse_mean:.6f}±{noda_rrmse_std:.6f}, Improvement: {rrmse_improvement:.2f}%")
            
            # SSIM
            da_ssim_mean, da_ssim_std = np.mean(ch_data['da_ssim']), np.std(ch_data['da_ssim'])
            noda_ssim_mean, noda_ssim_std = np.mean(ch_data['noda_ssim']), np.std(ch_data['noda_ssim'])
            ssim_improvement = ((da_ssim_mean - noda_ssim_mean) / noda_ssim_mean * 100)
            print(f"SSIM - 4D Var: {da_ssim_mean:.6f}±{da_ssim_std:.6f}, No DA: {noda_ssim_mean:.6f}±{noda_ssim_std:.6f}, Improvement: {ssim_improvement:.2f}%")
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)
    print(f"Generated plots for {len(channel_names)} channels and {len(all_models_data)} models")
    print(f"Total plots generated: {len(channel_names) * 4} plots")
    print(f"All plots saved to: {fig_save_path}")