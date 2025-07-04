import os
import sys
import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import time

# Add parent directories to path for imports
current_directory = os.getcwd()
upper_directory = os.path.abspath(os.path.join(current_directory, ".."))
upper_upper_directory = os.path.abspath(os.path.join(current_directory, "..", ".."))
sys.path.append(upper_directory)
sys.path.append(upper_upper_directory)

# Import models and trainer utilities
from cyl_model import CYL_forward_model, CylinderConfig
from cyl_trainer import DatasetCylinder, CylinderSequenceDataset, dict2namespace, set_seed


class CylinderModelEvaluator:
    """Class for evaluating trained Cylinder flow models"""
    
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        self.model = None
        self.base_dataset = None
        self.val_dataset = None
        
    def load_model(self, model_path):
        """Load the best trained model and DMD matrix"""
        print(f"[INFO] Loading model from {model_path}")
        
        # Create model
        model_config = CylinderConfig()
        model_config.input_channels = self.config.input_channels
        model_config.latent_dim = self.config.latent_dim
        model_config.seq_length = self.config.seq_length
        
        self.model = CYL_forward_model(model_config)
        
        # Load weights
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"[INFO] Model loaded successfully")
        
        # Load DMD matrix separately
        model_dir = os.path.dirname(model_path)
        dmd_filename = os.path.basename(model_path).replace('cyl_forward_model_', 'C_fwd_')
        dmd_path = os.path.join(model_dir, dmd_filename)
        
        if os.path.exists(dmd_path):
            self.model.C_fwd = torch.load(dmd_path, map_location=self.device)
            print(f"[INFO] DMD matrix loaded from {dmd_path}")
            print(f"[INFO] DMD matrix shape: {self.model.C_fwd.shape}")
        else:
            print(f"[WARNING] DMD matrix not found at {dmd_path}")
            print(f"[WARNING] Will use identity mapping for predictions")
            self.model.C_fwd = None
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[INFO] Model parameters: {total_params:,}")
        
    def load_data(self):
        """Load and prepare validation dataset"""
        print(f"[INFO] Loading validation dataset")
        
        # Create base dataset
        self.base_dataset = DatasetCylinder(
            data_path=self.config.data_path,
            normalize=getattr(self.config, 'normalize', True),
            train_ratio=getattr(self.config, 'train_ratio', 0.8),
            random_seed=getattr(self.config, 'random_seed', 42)
        )
        
        # Create validation sequence dataset
        self.val_dataset = CylinderSequenceDataset(
            self.base_dataset, 
            seq_length=self.config.seq_length, 
            is_training=False
        )
        
        print(f"[INFO] Validation dataset size: {len(self.val_dataset)}")
        print(f"[INFO] Data channels: {self.base_dataset.data.shape[2] if self.base_dataset.data.ndim == 5 else 1}")
        
    def denormalize_data(self, data):
        """Denormalize data using dataset statistics"""
        if self.base_dataset.mean is not None and self.base_dataset.std is not None:
            mean = torch.FloatTensor(self.base_dataset.mean).to(data.device)
            std = torch.FloatTensor(self.base_dataset.std).to(data.device)
            return data * std + mean
        return data
        
    def compute_velocity_magnitude(self, velocity_field):
        """
        Compute velocity magnitude from velocity components
        Args:
            velocity_field: torch.Tensor of shape [T, C, H, W] where C could be 1 or 2
        Returns:
            velocity_magnitude: torch.Tensor of shape [T, 1, H, W]
        """
        print(f"[DEBUG] Input velocity_field shape: {velocity_field.shape}")
        
        # Handle the case where velocity_field might have an extra dimension
        if velocity_field.dim() == 5:
            # Shape might be [1, T, C, H, W], squeeze the batch dimension
            velocity_field = velocity_field.squeeze(0)  # [T, C, H, W]
            print(f"[DEBUG] After squeezing batch dim: {velocity_field.shape}")
        
        # Now velocity_field should be [T, C, H, W]
        T, C, H, W = velocity_field.shape
        print(f"[DEBUG] T={T}, C={C}, H={H}, W={W}")
        
        if C == 1:
            # Single channel - already magnitude or single component
            result = velocity_field
        elif C == 2:
            # Two channels - compute magnitude from u and v components
            u = velocity_field[:, 0:1, :, :]  # [T, 1, H, W]
            v = velocity_field[:, 1:2, :, :]  # [T, 1, H, W]
            magnitude = torch.sqrt(u**2 + v**2)  # [T, 1, H, W]
            result = magnitude
        else:
            # More than 2 channels - take the first channel or compute norm
            print(f"[WARNING] Unexpected number of channels: {C}, using first channel")
            result = velocity_field[:, 0:1, :, :]
        
        print(f"[DEBUG] Output velocity_magnitude shape: {result.shape}")
        return result
        
    def compute_metrics(self, pred, gt):
        """Compute evaluation metrics between prediction and ground truth"""
        # Convert to numpy for metric computation
        pred_np = pred.detach().cpu().numpy()
        gt_np = gt.detach().cpu().numpy()
        
        print(f"[DEBUG] pred_np shape: {pred_np.shape}, gt_np shape: {gt_np.shape}")
        print(f"[DEBUG] pred_np range: [{pred_np.min():.6f}, {pred_np.max():.6f}]")
        print(f"[DEBUG] gt_np range: [{gt_np.min():.6f}, {gt_np.max():.6f}]")
        
        # Handle potential extra dimensions
        while pred_np.ndim > 4:
            # Remove leading singleton dimensions
            if pred_np.shape[0] == 1:
                pred_np = pred_np.squeeze(0)
                gt_np = gt_np.squeeze(0)
            else:
                break
        
        print(f"[DEBUG] After squeezing - pred_np shape: {pred_np.shape}, gt_np shape: {gt_np.shape}")
        
        # MSE
        mse = np.mean((pred_np - gt_np) ** 2)
        
        # MAE
        mae = np.mean(np.abs(pred_np - gt_np))
        
        # Relative L2 error
        gt_norm = np.linalg.norm(gt_np)
        if gt_norm > 1e-12:
            rel_l2 = np.linalg.norm(pred_np - gt_np) / gt_norm
        else:
            rel_l2 = 0.0
        
        # SSIM (computed for each frame and averaged)
        ssim_scores = []
        
        if pred_np.ndim == 4:  # [T, C, H, W]
            T, C, H, W = pred_np.shape
            print(f"[DEBUG] Processing {T} frames with {C} channels, size {H}x{W}")
            
            for t in range(T):
                for c in range(C):
                    pred_frame = pred_np[t, c]  # [H, W]
                    gt_frame = gt_np[t, c]      # [H, W]
                    
                    # Check if frames have meaningful variation
                    gt_range = gt_frame.max() - gt_frame.min()
                    pred_range = pred_frame.max() - pred_frame.min()
                    
                    if gt_range > 1e-8:  # Only compute SSIM if there's meaningful variation
                        try:
                            # Use the full data range for SSIM calculation
                            data_range = max(gt_range, pred_range, 1e-6)
                            
                            # Ensure window size is appropriate
                            win_size = min(7, min(H, W) // 4)
                            if win_size < 3:
                                win_size = 3
                            
                            ssim_val = ssim(
                                gt_frame, pred_frame, 
                                data_range=data_range,
                                win_size=win_size,
                                multichannel=False,
                                gaussian_weights=True,
                                sigma=1.5,
                                use_sample_covariance=False
                            )
                            
                            if not np.isnan(ssim_val) and not np.isinf(ssim_val):
                                ssim_scores.append(ssim_val)
                                if t == 0 and c == 0:  # Only print for first frame/channel to avoid spam
                                    print(f"[DEBUG] SSIM for frame {t}, channel {c}: {ssim_val:.6f}")
                            else:
                                print(f"[DEBUG] Invalid SSIM for frame {t}, channel {c}: {ssim_val}")
                                
                        except Exception as e:
                            print(f"[DEBUG] SSIM computation failed for frame {t}, channel {c}: {e}")
                    else:
                        if t == 0 and c == 0:  # Only print for first frame/channel
                            print(f"[DEBUG] Skipping SSIM for frame {t}, channel {c} due to low variation (range={gt_range:.8f})")
                        
        elif pred_np.ndim == 3:  # [C, H, W] or [T, H, W]
            if pred_np.shape[0] <= 3:  # Likely [C, H, W]
                C = pred_np.shape[0]
                for c in range(C):
                    pred_frame = pred_np[c]
                    gt_frame = gt_np[c]
                    
                    gt_range = gt_frame.max() - gt_frame.min()
                    
                    if gt_range > 1e-8:
                        try:
                            data_range = max(gt_range, pred_frame.max() - pred_frame.min(), 1e-6)
                            win_size = min(7, min(pred_frame.shape) // 4)
                            if win_size < 3:
                                win_size = 3
                                
                            ssim_val = ssim(
                                gt_frame, pred_frame, 
                                data_range=data_range,
                                win_size=win_size,
                                multichannel=False
                            )
                            
                            if not np.isnan(ssim_val) and not np.isinf(ssim_val):
                                ssim_scores.append(ssim_val)
                                
                        except Exception as e:
                            print(f"[DEBUG] SSIM computation failed for channel {c}: {e}")
            else:  # Likely [T, H, W]
                T = pred_np.shape[0]
                for t in range(T):
                    pred_frame = pred_np[t]
                    gt_frame = gt_np[t]
                    
                    gt_range = gt_frame.max() - gt_frame.min()
                    
                    if gt_range > 1e-8:
                        try:
                            data_range = max(gt_range, pred_frame.max() - pred_frame.min(), 1e-6)
                            win_size = min(7, min(pred_frame.shape) // 4)
                            if win_size < 3:
                                win_size = 3
                                
                            ssim_val = ssim(
                                gt_frame, pred_frame, 
                                data_range=data_range,
                                win_size=win_size,
                                multichannel=False
                            )
                            
                            if not np.isnan(ssim_val) and not np.isinf(ssim_val):
                                ssim_scores.append(ssim_val)
                                
                        except Exception as e:
                            print(f"[DEBUG] SSIM computation failed for frame {t}: {e}")
        
        ssim_mean = np.mean(ssim_scores) if ssim_scores else 0.0
        print(f"[DEBUG] SSIM scores count: {len(ssim_scores)}, mean: {ssim_mean:.6f}")
        
        return {
            'MSE': mse,
            'MAE': mae,
            'Relative_L2': rel_l2,
            'SSIM': ssim_mean
        }
    
    def reconstruction_prediction(self, input_seq):
        """Perform reconstruction prediction (identity mapping)"""
        with torch.no_grad():
            # Encode to latent space
            latent_seq = []
            for t in range(input_seq.shape[0]):
                z_t = self.model.phi_S(input_seq[t:t+1])  # [1, C, H, W] -> [1, latent_dim]
                latent_seq.append(z_t)
            
            latent_seq = torch.cat(latent_seq, dim=0)  # [T, latent_dim]
            
            # Decode back to original space
            recon_seq = []
            for t in range(latent_seq.shape[0]):
                x_recon = self.model.phi_inv_S(latent_seq[t:t+1])  # [1, latent_dim] -> [1, C, H, W]
                recon_seq.append(x_recon)
            
            recon_seq = torch.cat(recon_seq, dim=0)  # [T, C, H, W]
            
        return recon_seq
    
    def onestep_prediction(self, input_seq):
        """Perform one-step prediction using DMD"""
        with torch.no_grad():
            # Encode sequence to latent space
            latent_seq = []
            for t in range(input_seq.shape[0]):
                z_t = self.model.phi_S(input_seq[t:t+1])
                latent_seq.append(z_t)
            
            latent_seq = torch.cat(latent_seq, dim=0)  # [T, latent_dim]
            
            # Apply DMD for one-step prediction
            if hasattr(self.model, 'C_fwd') and self.model.C_fwd is not None:
                latent_pred = torch.matmul(latent_seq, self.model.C_fwd.T)  # [T, latent_dim]
            else:
                print("[WARNING] DMD matrix not found, using identity")
                latent_pred = latent_seq
            
            # Decode predictions
            pred_seq = []
            for t in range(latent_pred.shape[0]):
                x_pred = self.model.phi_inv_S(latent_pred[t:t+1])
                pred_seq.append(x_pred)
            
            pred_seq = torch.cat(pred_seq, dim=0)  # [T, C, H, W]
            
        return pred_seq
    
    def rollout_prediction(self, initial_state, num_steps):
        """Perform rollout prediction (autoregressive)"""
        self.model.eval()  # 确保模型在评估模式
        
        with torch.no_grad():
            predictions = []
            current_state = initial_state.clone()  # 避免修改原始数据
            device = next(self.model.parameters()).device
            current_state = current_state.to(device)
            
            for step in range(num_steps):
                # Encode current state
                z_current = self.model.phi_S(current_state.unsqueeze(0))
                
                # Apply DMD
                if hasattr(self.model, 'C_fwd') and self.model.C_fwd is not None:
                    C_fwd = self.model.C_fwd.to(device).to(z_current.dtype)
                    z_next = torch.matmul(z_current, C_fwd)  # 移除.T
                else:
                    print("[WARNING] DMD matrix not found, using identity")
                    z_next = z_current
                
                # Decode next state
                next_state = self.model.phi_inv_S(z_next)
                next_state = next_state.squeeze(0)
                
                predictions.append(next_state)
                current_state = next_state
            
            # Stack predictions: [T, C, H, W]
            rollout_seq = torch.stack(predictions, dim=0)
            
        return rollout_seq
    
    def evaluate_sample(self, sample_idx=5):
        """Evaluate model on a specific validation sample"""
        print(f"\n[INFO] Evaluating sample {sample_idx}")
        
        if sample_idx >= len(self.val_dataset):
            raise ValueError(f"Sample index {sample_idx} exceeds dataset size {len(self.val_dataset)}")
        
        # Get sample data
        state_seq, state_next_seq = self.val_dataset[sample_idx]
        state_seq = state_seq.to(self.device)  # [T, C, H, W]
        state_next_seq = state_next_seq.to(self.device)  # [T, C, H, W]
        
        print(f"[INFO] Input sequence shape: {state_seq.shape}")
        print(f"[INFO] Target sequence shape: {state_next_seq.shape}")
        print(f"[INFO] Number of channels: {state_seq.shape[1]}")
        
        # Store original normalized data for reference
        input_normalized = state_seq.clone()
        target_normalized = state_next_seq.clone()
        
        # Denormalize for evaluation
        input_denorm = self.denormalize_data(state_seq)
        target_denorm = self.denormalize_data(state_next_seq)
        
        results = {}
        
        # 1. Reconstruction prediction
        print("[INFO] Performing reconstruction prediction...")
        recon_pred_norm = self.reconstruction_prediction(input_normalized)
        recon_pred_denorm = self.denormalize_data(recon_pred_norm)
        
        # Compute velocity magnitude for evaluation
        print("[INFO] Computing velocity magnitude for reconstruction...")
        input_magnitude = self.compute_velocity_magnitude(input_denorm)
        recon_magnitude = self.compute_velocity_magnitude(recon_pred_denorm)
        
        recon_metrics = self.compute_metrics(recon_magnitude, input_magnitude)
        results['reconstruction'] = {
            'prediction_normalized': recon_pred_norm,
            'prediction_denormalized': recon_pred_denorm,
            'prediction_magnitude': recon_magnitude,
            'ground_truth_denormalized': input_denorm,
            'ground_truth_magnitude': input_magnitude,
            'metrics': recon_metrics
        }
        
        # 2. One-step prediction
        print("[INFO] Performing one-step prediction...")
        onestep_pred_norm = self.onestep_prediction(input_normalized)
        onestep_pred_denorm = self.denormalize_data(onestep_pred_norm)
        
        # Compute velocity magnitude for evaluation
        target_magnitude = self.compute_velocity_magnitude(target_denorm)
        onestep_magnitude = self.compute_velocity_magnitude(onestep_pred_denorm)
        
        onestep_metrics = self.compute_metrics(onestep_magnitude, target_magnitude)
        results['onestep'] = {
            'prediction_normalized': onestep_pred_norm,
            'prediction_denormalized': onestep_pred_denorm,
            'prediction_magnitude': onestep_magnitude,
            'ground_truth_denormalized': target_denorm,
            'ground_truth_magnitude': target_magnitude,
            'metrics': onestep_metrics
        }
        
        # 3. Rollout prediction
        print("[INFO] Performing rollout prediction...")
        initial_state = input_normalized[0]  # First frame
        rollout_pred_norm = self.rollout_prediction(initial_state, len(state_seq))
        rollout_pred_denorm = self.denormalize_data(rollout_pred_norm)
        
        # For rollout, compare with the target sequence (all frames)
        rollout_target_denorm = target_denorm
        rollout_magnitude = self.compute_velocity_magnitude(rollout_pred_denorm)
        rollout_target_magnitude = self.compute_velocity_magnitude(rollout_target_denorm)
        
        rollout_metrics = self.compute_metrics(rollout_magnitude, rollout_target_magnitude)
        results['rollout'] = {
            'prediction_normalized': rollout_pred_norm,
            'prediction_denormalized': rollout_pred_denorm,
            'prediction_magnitude': rollout_magnitude,
            'ground_truth_denormalized': rollout_target_denorm,
            'ground_truth_magnitude': rollout_target_magnitude,
            'metrics': rollout_metrics
        }
        
        return results
    
    def print_metrics(self, results):
        """Print evaluation metrics for all prediction methods"""
        print("\n" + "="*80)
        print("EVALUATION METRICS (Based on Velocity Magnitude)")
        print("="*80)
        
        methods = ['reconstruction', 'onestep', 'rollout']
        metrics = ['MSE', 'MAE', 'Relative_L2', 'SSIM']
        
        # Print header
        print(f"{'Method':<15} {'MSE':<12} {'MAE':<12} {'Rel_L2':<12} {'SSIM':<12}")
        print("-" * 80)
        
        # Print metrics for each method
        for method in methods:
            if method in results:
                m = results[method]['metrics']
                print(f"{method:<15} {m['MSE']:<12.6f} {m['MAE']:<12.6f} {m['Relative_L2']:<12.6f} {m['SSIM']:<12.6f}")
        
        print("="*80)
    
    def visualize_results(self, results, save_path, sample_idx=5):
        """Create comprehensive visualization of results"""
        print(f"\n[INFO] Creating visualizations...")
        
        methods = ['reconstruction', 'onestep', 'rollout']
        n_methods = len(methods)
        
        # Select time steps to visualize (first, middle, last)
        seq_length = results['reconstruction']['prediction_magnitude'].shape[0]
        time_indices = [0, seq_length // 2, seq_length - 1]
        n_times = len(time_indices)
        
        # Create figure
        fig, axes = plt.subplots(n_methods + 1, n_times, figsize=(4*n_times, 4*(n_methods + 1)))
        if n_times == 1:
            axes = axes.reshape(-1, 1)
        
        # Color maps and ranges for visualization
        def get_vmin_vmax(data):
            return data.min().item(), data.max().item()
        
        # Plot ground truth magnitude
        gt_data = results['reconstruction']['ground_truth_magnitude']
        print(f"[DEBUG] Visualization gt_data shape: {gt_data.shape}")
        
        for j, t_idx in enumerate(time_indices):
            # gt_data should be [T, 1, H, W] for magnitude
            # So we use [t_idx, 0] to get [H, W]
            if gt_data.dim() == 4:
                selected_data = gt_data[t_idx, 0]  # [H, W]
            elif gt_data.dim() == 5:
                # Handle case where shape is [1, T, 1, H, W] or similar
                if gt_data.shape[0] == 1:
                    selected_data = gt_data[0, t_idx, 0]  # [H, W]
                else:
                    selected_data = gt_data[t_idx, 0, 0]  # [H, W]
            else:
                selected_data = gt_data[t_idx]  # Fallback
            
            print(f"[DEBUG] Visualization selected_data shape for t={t_idx}: {selected_data.shape}")
            
            vmin, vmax = get_vmin_vmax(selected_data)
            im = axes[0, j].imshow(selected_data.cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
            axes[0, j].set_title(f'Ground Truth\nVelocity Magnitude\nt={t_idx}', fontsize=12)
            axes[0, j].axis('off')
            plt.colorbar(im, ax=axes[0, j], fraction=0.046, pad=0.04)
        
        # Plot predictions for each method
        for i, method in enumerate(methods):
            pred_data = results[method]['prediction_magnitude']
            gt_data = results[method]['ground_truth_magnitude']
            
            for j, t_idx in enumerate(time_indices):
                # Handle different dimension cases consistently
                def get_2d_slice(data, t_idx):
                    if data.dim() == 4:
                        return data[t_idx, 0]  # [H, W]
                    elif data.dim() == 5:
                        if data.shape[0] == 1:
                            return data[0, t_idx, 0]  # [H, W]
                        else:
                            return data[t_idx, 0, 0]  # [H, W]
                    else:
                        return data[t_idx]  # Fallback
                
                gt_selected = get_2d_slice(gt_data, t_idx)
                pred_selected = get_2d_slice(pred_data, t_idx)
                
                # Use same color scale as ground truth
                vmin, vmax = get_vmin_vmax(gt_selected)
                
                im = axes[i+1, j].imshow(pred_selected.cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
                
                # Add metrics to title
                metrics = results[method]['metrics']
                title = f'{method.capitalize()} Prediction\nVelocity Magnitude\nt={t_idx}\nMSE={metrics["MSE"]:.4f}, SSIM={metrics["SSIM"]:.3f}'
                axes[i+1, j].set_title(title, fontsize=10)
                axes[i+1, j].axis('off')
                plt.colorbar(im, ax=axes[i+1, j], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # Save figure
        os.makedirs(save_path, exist_ok=True)
        fig_path = os.path.join(save_path, f'cylinder_predictions_sample_{sample_idx}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Visualization saved to {fig_path}")
        
        # Create error visualization
        self.visualize_errors(results, save_path, sample_idx, time_indices)
        
        # Create component visualization if multi-channel
        sample_data = results['reconstruction']['prediction_denormalized']
        if sample_data.shape[1] == 2:
            self.visualize_velocity_components(results, save_path, sample_idx, time_indices)
    
    def visualize_errors(self, results, save_path, sample_idx, time_indices):
        """Create error visualization"""
        methods = ['reconstruction', 'onestep', 'rollout']
        n_methods = len(methods)
        n_times = len(time_indices)
        
        fig, axes = plt.subplots(n_methods, n_times, figsize=(4*n_times, 4*n_methods))
        if n_times == 1:
            axes = axes.reshape(-1, 1)
        if n_methods == 1:
            axes = axes.reshape(1, -1)
        
        for i, method in enumerate(methods):
            pred_data = results[method]['prediction_magnitude']
            gt_data = results[method]['ground_truth_magnitude']
            
            for j, t_idx in enumerate(time_indices):
                # Handle different dimension cases consistently
                def get_2d_slice(data, t_idx):
                    if data.dim() == 4:
                        return data[t_idx, 0]  # [H, W]
                    elif data.dim() == 5:
                        if data.shape[0] == 1:
                            return data[0, t_idx, 0]  # [H, W]
                        else:
                            return data[t_idx, 0, 0]  # [H, W]
                    else:
                        return data[t_idx]  # Fallback
                
                gt_selected = get_2d_slice(gt_data, t_idx)
                pred_selected = get_2d_slice(pred_data, t_idx)
                
                # Compute absolute error
                error = torch.abs(pred_selected - gt_selected)
                
                im = axes[i, j].imshow(error.cpu().numpy(), cmap='hot')
                axes[i, j].set_title(f'{method.capitalize()} Error\nVelocity Magnitude\nt={t_idx}', fontsize=12)
                axes[i, j].axis('off')
                plt.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # Save error figure
        error_fig_path = os.path.join(save_path, f'cylinder_errors_sample_{sample_idx}.png')
        plt.savefig(error_fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Error visualization saved to {error_fig_path}")
        
        # Create time series metrics plot
        self.plot_metrics_comparison(results, save_path, sample_idx)
    
    def visualize_velocity_components(self, results, save_path, sample_idx, time_indices):
        """Visualize individual velocity components (u, v) if available"""
        print(f"[INFO] Creating velocity component visualizations...")
        
        methods = ['reconstruction', 'onestep', 'rollout']
        n_methods = len(methods)
        n_times = len(time_indices)
        
        # Create figure for u-component
        fig_u, axes_u = plt.subplots(n_methods + 1, n_times, figsize=(4*n_times, 4*(n_methods + 1)))
        fig_v, axes_v = plt.subplots(n_methods + 1, n_times, figsize=(4*n_times, 4*(n_methods + 1)))
        
        if n_times == 1:
            axes_u = axes_u.reshape(-1, 1)
            axes_v = axes_v.reshape(-1, 1)
        
        # Plot ground truth components
        gt_data = results['reconstruction']['ground_truth_denormalized']
        for j, t_idx in enumerate(time_indices):
            # U component - shape is [T, C, H, W], so [t_idx, 0] gives [H, W]
            vmin_u, vmax_u = gt_data[t_idx, 0].min().item(), gt_data[t_idx, 0].max().item()
            im_u = axes_u[0, j].imshow(gt_data[t_idx, 0].cpu().numpy(), cmap='RdBu_r', vmin=vmin_u, vmax=vmax_u)
            axes_u[0, j].set_title(f'Ground Truth\nU-component\nt={t_idx}', fontsize=12)
            axes_u[0, j].axis('off')
            plt.colorbar(im_u, ax=axes_u[0, j], fraction=0.046, pad=0.04)
            
            # V component - shape is [T, C, H, W], so [t_idx, 1] gives [H, W]
            vmin_v, vmax_v = gt_data[t_idx, 1].min().item(), gt_data[t_idx, 1].max().item()
            im_v = axes_v[0, j].imshow(gt_data[t_idx, 1].cpu().numpy(), cmap='RdBu_r', vmin=vmin_v, vmax=vmax_v)
            axes_v[0, j].set_title(f'Ground Truth\nV-component\nt={t_idx}', fontsize=12)
            axes_v[0, j].axis('off')
            plt.colorbar(im_v, ax=axes_v[0, j], fraction=0.046, pad=0.04)
        
        # Plot predictions for each method
        for i, method in enumerate(methods):
            pred_data = results[method]['prediction_denormalized']
            gt_data = results[method]['ground_truth_denormalized']
            
            for j, t_idx in enumerate(time_indices):
                # U component
                vmin_u, vmax_u = gt_data[t_idx, 0].min().item(), gt_data[t_idx, 0].max().item()
                im_u = axes_u[i+1, j].imshow(pred_data[t_idx, 0].cpu().numpy(), cmap='RdBu_r', vmin=vmin_u, vmax=vmax_u)
                axes_u[i+1, j].set_title(f'{method.capitalize()}\nU-component\nt={t_idx}', fontsize=10)
                axes_u[i+1, j].axis('off')
                plt.colorbar(im_u, ax=axes_u[i+1, j], fraction=0.046, pad=0.04)
                
                # V component
                vmin_v, vmax_v = gt_data[t_idx, 1].min().item(), gt_data[t_idx, 1].max().item()
                im_v = axes_v[i+1, j].imshow(pred_data[t_idx, 1].cpu().numpy(), cmap='RdBu_r', vmin=vmin_v, vmax=vmax_v)
                axes_v[i+1, j].set_title(f'{method.capitalize()}\nV-component\nt={t_idx}', fontsize=10)
                axes_v[i+1, j].axis('off')
                plt.colorbar(im_v, ax=axes_v[i+1, j], fraction=0.046, pad=0.04)
        
        # Save component figures
        plt.figure(fig_u.number)
        plt.tight_layout()
        u_fig_path = os.path.join(save_path, f'cylinder_u_component_sample_{sample_idx}.png')
        plt.savefig(u_fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(fig_v.number)
        plt.tight_layout()
        v_fig_path = os.path.join(save_path, f'cylinder_v_component_sample_{sample_idx}.png')
        plt.savefig(v_fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Velocity component visualizations saved:")
        print(f"       U-component: {u_fig_path}")
        print(f"       V-component: {v_fig_path}")
    
    def plot_metrics_comparison(self, results, save_path, sample_idx):
        """Plot metrics comparison across methods"""
        methods = ['reconstruction', 'onestep', 'rollout']
        metrics = ['MSE', 'MAE', 'Relative_L2', 'SSIM']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        method_colors = {'reconstruction': 'blue', 'onestep': 'red', 'rollout': 'green'}
        
        for i, metric in enumerate(metrics):
            values = [results[method]['metrics'][metric] for method in methods]
            colors = [method_colors[method] for method in methods]
            
            bars = axes[i].bar(methods, values, color=colors, alpha=0.7)
            axes[i].set_title(f'{metric} Comparison\n(Velocity Magnitude)', fontsize=14)
            axes[i].set_ylabel(metric)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.4f}' if metric != 'SSIM' else f'{value:.3f}',
                           ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save metrics comparison
        metrics_fig_path = os.path.join(save_path, f'cylinder_metrics_comparison_sample_{sample_idx}.png')
        plt.savefig(metrics_fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Metrics comparison saved to {metrics_fig_path}")


def main():
    """Main evaluation function"""
    # Load configuration
    config_path = 'cyl_config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config = dict2namespace(config)
    
    # Set seed for reproducibility
    set_seed(getattr(config, 'random_seed', 42))
    
    # Device setup
    if not getattr(config, 'use_cpu', False) and torch.cuda.is_available():
        device = f"cuda:{config.gpu_id}"
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(config.gpu_id)}")
    else:
        device = "cpu"
        print(f"[INFO] Using CPU")
    
    # Create evaluator
    evaluator = CylinderModelEvaluator(config, device)
    
    # Load trained model
    model_path = os.path.join(config.model_save_path, 'cyl_forward_model_best.pt')
    evaluator.load_model(model_path)
    
    # Load validation data
    evaluator.load_data()
    
    # Evaluate specific sample
    sample_idx = 5
    print(f"\n[INFO] Starting evaluation of sample {sample_idx}")
    
    start_time = time.time()
    results = evaluator.evaluate_sample(sample_idx)
    eval_time = time.time() - start_time
    
    print(f"[INFO] Evaluation completed in {eval_time:.2f} seconds")
    
    # Print metrics
    evaluator.print_metrics(results)
    
    # Create visualizations
    save_path = os.path.join(config.model_save_path, 'evaluation_results')
    evaluator.visualize_results(results, save_path, sample_idx)
    
    print(f"\n[INFO] Evaluation completed successfully!")
    print(f"[INFO] Results saved to {save_path}")
    print(f"[INFO] Metrics are computed based on velocity magnitude")
    if evaluator.base_dataset.data.ndim == 5 and evaluator.base_dataset.data.shape[2] == 2:
        print(f"[INFO] Individual velocity component visualizations also created")


if __name__ == "__main__":
    main()