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
from kol_model import KOL_forward_model, KolmogorovConfig
from kol_trainer import DatasetKol, KolmogorovSequenceDataset, dict2namespace, set_seed


class ModelEvaluator:
    """Class for evaluating trained Kolmogorov flow models"""
    
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
        model_config = KolmogorovConfig()
        model_config.input_channels = self.config.input_channels
        model_config.latent_dim = self.config.latent_dim
        model_config.seq_length = self.config.seq_length
        
        self.model = KOL_forward_model(model_config)
        
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
        dmd_filename = os.path.basename(model_path).replace('kol_forward_model_', 'C_fwd_')
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
        self.base_dataset = DatasetKol(
            data_path=self.config.data_path,
            normalize=getattr(self.config, 'normalize', True),
            train_ratio=getattr(self.config, 'train_ratio', 0.8),
            random_seed=getattr(self.config, 'random_seed', 42)
        )
        
        # Create validation sequence dataset
        self.val_dataset = KolmogorovSequenceDataset(
            self.base_dataset, 
            seq_length=self.config.seq_length, 
            is_training=False
        )
        
        print(f"[INFO] Validation dataset size: {len(self.val_dataset)}")
        
    def denormalize_data(self, data):
        """Denormalize data using dataset statistics"""
        if self.base_dataset.mean is not None and self.base_dataset.std is not None:
            mean = torch.FloatTensor(self.base_dataset.mean).to(data.device)
            std = torch.FloatTensor(self.base_dataset.std).to(data.device)
            return data * std + mean
        return data
        
    def compute_metrics(self, pred, gt):
        """Compute evaluation metrics between prediction and ground truth"""
        # Convert to numpy for metric computation
        pred_np = pred.detach().cpu().numpy()
        gt_np = gt.detach().cpu().numpy()
        
        # MSE
        mse = np.mean((pred_np - gt_np) ** 2)
        
        # MAE
        mae = np.mean(np.abs(pred_np - gt_np))
        
        # Relative L2 error
        rel_l2 = np.linalg.norm(pred_np - gt_np) / np.linalg.norm(gt_np)
        
        # SSIM (computed for each frame and averaged)
        ssim_scores = []
        if pred_np.ndim == 4:  # [T, C, H, W]
            for t in range(pred_np.shape[0]):
                for c in range(pred_np.shape[1]):
                    ssim_val = ssim(gt_np[t, c], pred_np[t, c], data_range=gt_np[t, c].max() - gt_np[t, c].min())
                    ssim_scores.append(ssim_val)
        elif pred_np.ndim == 3:  # [C, H, W]
            for c in range(pred_np.shape[0]):
                ssim_val = ssim(gt_np[c], pred_np[c], data_range=gt_np[c].max() - gt_np[c].min())
                ssim_scores.append(ssim_val)
        
        ssim_mean = np.mean(ssim_scores) if ssim_scores else 0.0
        
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
        with torch.no_grad():
            predictions = []
            current_state = initial_state
            
            for step in range(num_steps):
                # Encode current state
                z_current = self.model.phi_S(current_state.unsqueeze(0))  # [1, C, H, W] -> [1, latent_dim]
                
                # Apply DMD
                if hasattr(self.model, 'C_fwd') and self.model.C_fwd is not None:
                    z_next = torch.matmul(z_current, self.model.C_fwd.T)
                else:
                    print("[WARNING] DMD matrix not found, using identity")
                    z_next = z_current
                
                # Decode next state
                next_state = self.model.phi_inv_S(z_next)  # [1, latent_dim] -> [1, C, H, W]
                next_state = next_state.squeeze(0)  # [C, H, W]
                
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
        
        recon_metrics = self.compute_metrics(recon_pred_denorm, input_denorm)
        results['reconstruction'] = {
            'prediction_normalized': recon_pred_norm,
            'prediction_denormalized': recon_pred_denorm,
            'ground_truth_denormalized': input_denorm,
            'metrics': recon_metrics
        }
        
        # 2. One-step prediction
        print("[INFO] Performing one-step prediction...")
        onestep_pred_norm = self.onestep_prediction(input_normalized)
        onestep_pred_denorm = self.denormalize_data(onestep_pred_norm)
        
        onestep_metrics = self.compute_metrics(onestep_pred_denorm, target_denorm)
        results['onestep'] = {
            'prediction_normalized': onestep_pred_norm,
            'prediction_denormalized': onestep_pred_denorm,
            'ground_truth_denormalized': target_denorm,
            'metrics': onestep_metrics
        }
        
        # 3. Rollout prediction
        print("[INFO] Performing rollout prediction...")
        initial_state = input_normalized[0]  # First frame
        rollout_pred_norm = self.rollout_prediction(initial_state, len(state_seq))
        rollout_pred_denorm = self.denormalize_data(rollout_pred_norm)
        
        # For rollout, compare with the target sequence (all frames)
        rollout_target_denorm = target_denorm
        rollout_metrics = self.compute_metrics(rollout_pred_denorm, rollout_target_denorm)
        results['rollout'] = {
            'prediction_normalized': rollout_pred_norm,
            'prediction_denormalized': rollout_pred_denorm,
            'ground_truth_denormalized': rollout_target_denorm,
            'metrics': rollout_metrics
        }
        
        return results
    
    def print_metrics(self, results):
        """Print evaluation metrics for all prediction methods"""
        print("\n" + "="*80)
        print("EVALUATION METRICS")
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
        seq_length = results['reconstruction']['prediction_denormalized'].shape[0]
        time_indices = [0, seq_length // 2, seq_length - 1]
        n_times = len(time_indices)
        
        # Create figure
        fig, axes = plt.subplots(n_methods + 1, n_times, figsize=(4*n_times, 4*(n_methods + 1)))
        if n_times == 1:
            axes = axes.reshape(-1, 1)
        
        # Color maps and ranges for visualization
        def get_vmin_vmax(data):
            return data.min().item(), data.max().item()
        
        # Plot ground truth
        gt_data = results['reconstruction']['ground_truth_denormalized']
        for j, t_idx in enumerate(time_indices):
            vmin, vmax = get_vmin_vmax(gt_data[t_idx, 0])
            im = axes[0, j].imshow(gt_data[t_idx, 0].cpu().numpy(), cmap='RdBu_r', vmin=vmin, vmax=vmax)
            axes[0, j].set_title(f'Ground Truth\nt={t_idx}', fontsize=12)
            axes[0, j].axis('off')
            plt.colorbar(im, ax=axes[0, j], fraction=0.046, pad=0.04)
        
        # Plot predictions for each method
        for i, method in enumerate(methods):
            pred_data = results[method]['prediction_denormalized']
            gt_data = results[method]['ground_truth_denormalized']
            
            for j, t_idx in enumerate(time_indices):
                # Use same color scale as ground truth
                vmin, vmax = get_vmin_vmax(gt_data[t_idx, 0])
                
                im = axes[i+1, j].imshow(pred_data[t_idx, 0].cpu().numpy(), cmap='RdBu_r', vmin=vmin, vmax=vmax)
                
                # Add metrics to title
                metrics = results[method]['metrics']
                title = f'{method.capitalize()} Prediction\nt={t_idx}\nMSE={metrics["MSE"]:.4f}, SSIM={metrics["SSIM"]:.3f}'
                axes[i+1, j].set_title(title, fontsize=10)
                axes[i+1, j].axis('off')
                plt.colorbar(im, ax=axes[i+1, j], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # Save figure
        os.makedirs(save_path, exist_ok=True)
        fig_path = os.path.join(save_path, f'kolmogorov_predictions_sample_{sample_idx}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Visualization saved to {fig_path}")
        
        # Create error visualization
        self.visualize_errors(results, save_path, sample_idx, time_indices)
    
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
            pred_data = results[method]['prediction_denormalized']
            gt_data = results[method]['ground_truth_denormalized']
            
            for j, t_idx in enumerate(time_indices):
                # Compute absolute error
                error = torch.abs(pred_data[t_idx, 0] - gt_data[t_idx, 0])
                
                im = axes[i, j].imshow(error.cpu().numpy(), cmap='hot')
                axes[i, j].set_title(f'{method.capitalize()} Error\nt={t_idx}', fontsize=12)
                axes[i, j].axis('off')
                plt.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # Save error figure
        error_fig_path = os.path.join(save_path, f'kolmogorov_errors_sample_{sample_idx}.png')
        plt.savefig(error_fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Error visualization saved to {error_fig_path}")
        
        # Create time series metrics plot
        self.plot_metrics_comparison(results, save_path, sample_idx)
    
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
            axes[i].set_title(f'{metric} Comparison', fontsize=14)
            axes[i].set_ylabel(metric)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.4f}' if metric != 'SSIM' else f'{value:.3f}',
                           ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save metrics comparison
        metrics_fig_path = os.path.join(save_path, f'kolmogorov_metrics_comparison_sample_{sample_idx}.png')
        plt.savefig(metrics_fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Metrics comparison saved to {metrics_fig_path}")


def main():
    """Main evaluation function"""
    # Load configuration
    config_path = 'kol_config.yaml'
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
    evaluator = ModelEvaluator(config, device)
    
    # Load trained model
    model_path = os.path.join(config.model_save_path, 'kol_forward_model_best.pt')
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


if __name__ == "__main__":
    main()