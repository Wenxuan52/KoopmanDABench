import os
import sys
import yaml
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

# Add project root to path
current_file_path = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(current_file_path, '..', '..', '..'))
sys.path.insert(0, project_root)

from src.models.DMD_4dvar.dmd4dvar import DMD4DVAR
from src.models.DMD_4dvar.utils import DMD4DVARDataLoader, add_observation_noise, visualize_observation_pattern
from src.models.DMD.dmd import DMD


class DMD4DVARTrainer:
    """Trainer for DMD+4DVAR model"""
    
    def __init__(self, config_path: str):
        """
        Initialize trainer with configuration file
        
        Args:
            config_path: Path to configuration YAML file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = torch.device(self.config['device'] if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create results directory
        self.results_dir = os.path.join(
            self.config['results']['save_dir'],
            self.config['experiment_name'],
            datetime.now().strftime('%Y%m%d_%H%M%S')
        )
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Save configuration
        with open(os.path.join(self.results_dir, 'config.yaml'), 'w') as f:
            yaml.dump(self.config, f)
        
        # Initialize components
        self._setup_data()
        self._setup_model()
        
    def _setup_data(self):
        """Setup data loaders"""
        data_config = self.config['data']
        
        self.dataloader = DMD4DVARDataLoader(
            dataset_name=data_config['dataset_name'],
            data_path=data_config['data_path'],
            obs_mode=data_config['observation']['mode'],
            obs_ratio=data_config['observation']['ratio'],
            obs_pattern=data_config['observation'].get('pattern', None),
            normalize=data_config['normalize'],
            train_ratio=data_config['train_ratio'],
            random_seed=self.config['seed'],
            sequence_length=data_config['sequence_length'],
            obs_frequency=data_config['observation']['frequency'],
            **data_config.get('dataset_kwargs', {})
        )
        
        # Get PyTorch dataloaders
        self.train_loader, self.val_loader = self.dataloader.get_dataloaders(
            batch_size=data_config['batch_size'],
            num_workers=data_config['num_workers']
        )
        
        # Print data statistics
        print("\nData Statistics:")
        stats = self.dataloader.get_data_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
    def _setup_model(self):
        """Setup DMD+4DVAR model"""
        model_config = self.config['model']
        
        # Load encoder/decoder if specified
        encoder, decoder = None, None
        if model_config.get('encoder_path') and model_config.get('decoder_path'):
            print(f"\nLoading encoder from: {model_config['encoder_path']}")
            print(f"Loading decoder from: {model_config['decoder_path']}")
            # TODO: Load actual encoder/decoder models
            # encoder = torch.load(model_config['encoder_path'])
            # decoder = torch.load(model_config['decoder_path'])
            
        # Initialize DMD+4DVAR model
        self.model = DMD4DVAR(
            encoder=encoder,
            decoder=decoder,
            dmd_params=model_config['dmd_params'],
            device=self.device
        )
        
        # Set error covariances
        self._setup_error_covariances()
        
    def _setup_error_covariances(self):
        """Setup error covariance matrices B and R"""
        assim_config = self.config['assimilation']
        
        # Get dimensions based on whether we're in latent space
        if self.config['model'].get('use_latent_space', False) and self.config['model'].get('latent_dim'):
            # Use latent dimension for B
            n_features = self.config['model']['latent_dim']
        else:
            # Use physical dimension
            if hasattr(self.dataloader.train_dataset, 'data_shape'):
                n_features = np.prod(self.dataloader.train_dataset.data_shape)
            else:
                # Fallback: get from first batch
                for batch in self.train_loader:
                    n_features = batch['sequence_flat'].shape[-1]
                    break
        
        # Background error covariance B
        if assim_config['B_type'] == 'identity':
            B = np.eye(n_features) * assim_config['B_scale']
        elif assim_config['B_type'] == 'diagonal':
            B = np.diag(np.random.rand(n_features) * assim_config['B_scale'])
        else:
            raise ValueError(f"Unknown B_type: {assim_config['B_type']}")
        
        # Observation error covariance R
        # Get number of observations
        H_example = list(self.dataloader.train_dataset.get_observation_operators().values())[0]
        n_obs = H_example.shape[0]
        
        if assim_config['R_type'] == 'identity':
            R = np.eye(n_obs) * assim_config['R_scale']
        elif assim_config['R_type'] == 'diagonal':
            R = np.diag(np.random.rand(n_obs) * assim_config['R_scale'])
        else:
            raise ValueError(f"Unknown R_type: {assim_config['R_type']}")
        
        self.model.set_error_covariances(B, R)
        print(f"\nError covariances set: B shape {B.shape}, R shape {R.shape}")
        
    def train_dmd(self):
        """Train DMD model on training data"""
        dmd_config = self.config['model']['dmd_training']
        
        if dmd_config['load_pretrained']:
            # Load pre-trained DMD model
            print(f"\nLoading pre-trained DMD from: {dmd_config['dmd_path']}")
            
            # Determine latent dimension
            if self.config['model'].get('use_latent_space', False):
                latent_dim = self.config['model'].get('latent_dim')
                if latent_dim is None:
                    raise ValueError("latent_dim must be specified when loading DMD trained in latent space")
            else:
                latent_dim = None
                
            self.model.load_dmd(dmd_config['dmd_path'], latent_dim=latent_dim)
            
        else:
            # Train new DMD model
            print("\nTraining DMD model...")
            
            # Prepare training data
            X_train_list = []
            Y_train_list = []
            
            # Collect all training sequences
            for batch in tqdm(self.train_loader, desc="Preparing DMD training data"):
                seq_flat = batch['sequence_flat']  # [batch, seq_len, n_features]
                
                # Create time-delay embedded data
                X_batch = seq_flat[:, :-1, :].reshape(-1, seq_flat.shape[-1])  # States at t
                Y_batch = seq_flat[:, 1:, :].reshape(-1, seq_flat.shape[-1])   # States at t+1
                
                X_train_list.append(X_batch)
                Y_train_list.append(Y_batch)
            
            # Concatenate all data
            X_train = np.vstack(X_train_list)
            Y_train = np.vstack(Y_train_list)
            
            print(f"DMD training data shape: X={X_train.shape}, Y={Y_train.shape}")
            
            # Train DMD
            use_latent = self.config['model'].get('use_latent_space', False)
            self.model.train_dmd(X_train, Y_train, use_latent=use_latent)
            
            # Save trained DMD
            dmd_save_path = os.path.join(self.results_dir, 'dmd_model.npz')
            self.model.dmd.save(dmd_save_path)
            print(f"DMD model saved to: {dmd_save_path}")
    
    def generate_dmd_background(self, x0: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Generate DMD background trajectory from initial condition
        
        Args:
            x0: Initial condition (n_features,)
            n_steps: Number of time steps to predict
            
        Returns:
            Background trajectory (n_features, n_steps)
        """
        return self.model.dmd.predict(x0=x0, n_steps=n_steps)
            
    def assimilate(self, val_indices: List[int]):
        """
        Perform data assimilation on specific validation samples
        
        Args:
            val_indices: Specific validation indices to assimilate
        """
        assim_config = self.config['assimilation']
        
        if not val_indices:
            raise ValueError("val_indices must be provided and non-empty")
        
        print(f"\nPerforming data assimilation on {len(val_indices)} validation samples...")
        
        # Results storage
        results = {
            'indices': val_indices,
            'background_trajectories': [],     # DMD forecast trajectories
            'analysis_trajectories': [],       # Full assimilated trajectories  
            'observations': [],
            'true_sequences': [],
            'costs': [],
            'errors': []
        }
        
        # Process each validation sample
        for idx in tqdm(val_indices, desc="Assimilating"):
            # Validate index
            if idx >= len(self.val_loader.dataset):
                raise IndexError(f"Index {idx} exceeds validation dataset size {len(self.val_loader.dataset)}")
            
            # Get validation sample
            sample = self.val_loader.dataset[idx]
            
            # Extract data
            x0 = sample['initial_condition']
            observations = sample['observations']
            sequence_true = sample['sequence_flat']
            
            print(f"\nProcessing sample {idx}:")
            print(f"  Initial condition shape: {x0.shape}")
            print(f"  True sequence shape: {sequence_true.shape}")
            print(f"  Number of observations: {len(observations)}")
            
            # Get observation operators
            H_operators = self.val_loader.dataset.get_observation_operators()
            
            # Add observation noise if specified
            if assim_config.get('add_noise', False):
                observations = add_observation_noise(
                    observations,
                    noise_level=assim_config['noise_level'],
                    noise_type=assim_config.get('noise_type', 'gaussian')
                )
            
            # Generate DMD background trajectory
            n_steps = sequence_true.shape[0]
            background_trajectory = self.generate_dmd_background(x0, n_steps)  # (n_features, n_steps)
            
            print(f"  Generated background trajectory shape: {background_trajectory.shape}")
            
            # Perform data assimilation
            assim_result = self.model.assimilate(
                x0=x0,
                observations=observations,
                H_operators=H_operators,
                method=assim_config['method'],
                window_size=assim_config['window_size'],
                max_iter=assim_config.get('max_iter', 100),
                tol=assim_config.get('tol', 1e-6)
            )
            
            # Extract analysis trajectory
            if 'trajectory' in assim_result:
                # Full trajectory available
                analysis_trajectory = assim_result['trajectory']  # (n_features, n_steps)
                print(f"  Analysis trajectory shape: {analysis_trajectory.shape}")
            else:
                # Only final analysis state available, need to generate trajectory
                analysis_state = assim_result['analysis']  # (n_features,)
                print(f"  Analysis state shape: {analysis_state.shape}")
                
                # Generate trajectory from analysis state
                analysis_trajectory = self.generate_dmd_background(analysis_state, n_steps)
                print(f"  Generated analysis trajectory shape: {analysis_trajectory.shape}")
            
            # Store results
            results['background_trajectories'].append(background_trajectory)
            results['analysis_trajectories'].append(analysis_trajectory) 
            results['observations'].append(observations)
            results['true_sequences'].append(sequence_true)
            results['costs'].append(assim_result['cost'])
            
            # Compute errors if true sequence available
            if sequence_true is not None:
                # Compute errors for both background and analysis
                forecast_steps = min(background_trajectory.shape[1], sequence_true.shape[0])
                
                # Background errors
                bg_errors = self.model.dmd.compute_error(
                    sequence_true[:forecast_steps].T,
                    background_trajectory[:, :forecast_steps]
                )
                
                # Analysis errors
                forecast_steps_analysis = min(analysis_trajectory.shape[1], sequence_true.shape[0])
                analysis_errors = self.model.dmd.compute_error(
                    sequence_true[:forecast_steps_analysis].T,
                    analysis_trajectory[:, :forecast_steps_analysis]
                )
                
                # Store both error sets
                sample_errors = {
                    'background': bg_errors,
                    'analysis': analysis_errors
                }
                results['errors'].append(sample_errors)
                
                print(f"  Background MSE: {bg_errors.get('mse', 'N/A'):.6f}")
                print(f"  Analysis MSE: {analysis_errors.get('mse', 'N/A'):.6f}")
        
        # Save results
        self._save_results(results)
        
        # Generate visualizations
        if self.config['results']['visualize']:
            self._visualize_results(results)
        
        return results
    
    def _save_results(self, results: Dict[str, Any]):
        """Save assimilation results"""
        # Save as pickle for full data
        results_path = os.path.join(self.results_dir, 'assimilation_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nResults saved to: {results_path}")
        
        # Save summary statistics as JSON
        summary = {
            'num_samples': len(results['indices']),
            'sample_indices': results['indices'],
            'mean_cost': float(np.mean(results['costs'])),
            'std_cost': float(np.std(results['costs'])),
        }
        
        if results['errors']:
            # Average errors across samples
            for error_type in ['background', 'analysis']:
                if error_type in results['errors'][0]:
                    for key in results['errors'][0][error_type].keys():
                        values = [err[error_type][key] for err in results['errors']]
                        summary[f'mean_{error_type}_{key}'] = float(np.mean(values))
                        summary[f'std_{error_type}_{key}'] = float(np.std(values))
        
        summary_path = os.path.join(self.results_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\nAssimilation Summary:")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
                
        # Print data shapes for verification
        print(f"\nData Shapes Verification:")
        for i, idx in enumerate(results['indices']):
            print(f"  Sample {idx}:")
            print(f"    Background trajectory: {results['background_trajectories'][i].shape}")
            print(f"    Analysis trajectory: {results['analysis_trajectories'][i].shape}")
            print(f"    True sequence: {results['true_sequences'][i].shape}")
    
    def _visualize_results(self, results: Dict[str, Any]):
        """Generate visualizations of assimilation results"""
        viz_dir = os.path.join(self.results_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Visualize observation pattern
        if self.dataloader.obs_mode == 'fixed':
            H = self.dataloader.val_dataset.H_operators
            data_shape = self.dataloader.val_dataset.data_shape
            if len(data_shape) == 2:
                obs_pattern_path = os.path.join(viz_dir, 'observation_pattern.png')
                visualize_observation_pattern(H, data_shape, save_path=obs_pattern_path)
        
        # Plot sample results
        n_samples_to_plot = min(5, len(results['indices']))
        
        for i in range(n_samples_to_plot):
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Get data
            true_seq = results['true_sequences'][i]
            background_traj = results['background_trajectories'][i]  # (n_features, n_steps)
            analysis_traj = results['analysis_trajectories'][i]      # (n_features, n_steps)
            
            # Plot first component time series
            ax = axes[0, 0]
            time_steps = range(true_seq.shape[0])
            ax.plot(time_steps, true_seq[:, 0], 'k-', label='True', linewidth=2)
            
            # Plot background and analysis trajectories
            forecast_steps = min(len(time_steps), background_traj.shape[1])
            ax.plot(range(forecast_steps), background_traj[0, :forecast_steps], 'b--', 
                   label='Background (DMD)', linewidth=2)
            
            analysis_steps = min(len(time_steps), analysis_traj.shape[1])
            ax.plot(range(analysis_steps), analysis_traj[0, :analysis_steps], 'r-', 
                   label='Analysis (4DVAR)', linewidth=2)
            
            # Mark observation times
            obs_times = list(results['observations'][i].keys())
            for t in obs_times:
                ax.axvline(x=t, color='g', linestyle=':', alpha=0.3)
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Component 1')
            ax.set_title('Time Series Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot error evolution
            ax = axes[0, 1]
            if i < len(results['errors']):
                # Background error
                bg_error_norm = np.linalg.norm(
                    true_seq[:forecast_steps, :] - background_traj[:, :forecast_steps].T, axis=1)
                ax.semilogy(range(forecast_steps), bg_error_norm, 'b--', 
                           label='Background Error', linewidth=2)
                
                # Analysis error
                analysis_error_norm = np.linalg.norm(
                    true_seq[:analysis_steps, :] - analysis_traj[:, :analysis_steps].T, axis=1)
                ax.semilogy(range(analysis_steps), analysis_error_norm, 'r-', 
                           label='Analysis Error', linewidth=2)
                
                ax.set_xlabel('Time')
                ax.set_ylabel('L2 Error')
                ax.set_title('Error Evolution')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Plot cost (if available)
            ax = axes[1, 0]
            if isinstance(results['costs'][i], dict) and 'cost_history' in results['costs'][i]:
                cost_history = results['costs'][i]['cost_history']
                ax.semilogy(cost_history, 'b-', linewidth=2)
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Cost')
                ax.set_title('Optimization Cost History')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'Final Cost: {results["costs"][i]:.6f}', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title('Final Cost')
            
            # Plot phase space (first two components)
            ax = axes[1, 1]
            if true_seq.shape[1] >= 2:
                ax.plot(true_seq[:, 0], true_seq[:, 1], 'k-', 
                       label='True', linewidth=2, alpha=0.7)
                
                # Background trajectory
                if background_traj.shape[0] >= 2:
                    ax.plot(background_traj[0, :forecast_steps], background_traj[1, :forecast_steps], 
                           'b--', label='Background', linewidth=2)
                
                # Analysis trajectory
                if analysis_traj.shape[0] >= 2:
                    ax.plot(analysis_traj[0, :analysis_steps], analysis_traj[1, :analysis_steps], 
                           'r-', label='Analysis', linewidth=2)
                
                # Initial condition
                ax.scatter(true_seq[0, 0], true_seq[0, 1], 
                          color='g', s=100, marker='o', label='Initial', zorder=5)
                
                ax.set_xlabel('Component 1')
                ax.set_ylabel('Component 2')
                ax.set_title('Phase Space')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.suptitle(f'Assimilation Results - Sample {results["indices"][i]}', fontsize=14)
            plt.tight_layout()
            
            # Save figure
            fig_path = os.path.join(viz_dir, f'sample_{results["indices"][i]}_results.png')
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"\nVisualizations saved to: {viz_dir}")
    
    def run(self, val_indices: List[int]):
        """
        Run complete DMD+4DVAR workflow
        
        Args:
            val_indices: Specific validation indices to assimilate (required)
        """
        if not val_indices:
            raise ValueError("val_indices must be provided and non-empty")
        
        print(f"\n{'='*60}")
        print(f"Running DMD+4DVAR Experiment: {self.config['experiment_name']}")
        print(f"Target samples: {val_indices}")
        print(f"{'='*60}")
        
        # Step 1: Train or load DMD
        self.train_dmd()
        
        # Step 2: Perform data assimilation on specified samples
        results = self.assimilate(val_indices)
        
        print(f"\n{'='*60}")
        print(f"Experiment completed. Results saved to: {self.results_dir}")
        print(f"{'='*60}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='DMD+4DVAR Training and Assimilation')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--val_indices', type=int, nargs='+', required=True,
                       help='Specific validation indices to assimilate (required)')
    
    args = parser.parse_args()
    
    # Create trainer and run
    trainer = DMD4DVARTrainer(args.config)
    trainer.run(val_indices=args.val_indices)


if __name__ == "__main__":
    main()