import os
import sys
import argparse
import yaml
import json
import shutil
from datetime import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import model-specific trainers
from models.CAE_MLP.trainer import train_caemlp_from_config


def create_experiment_folder(base_path: str, model_name: str, dataset_name: str) -> str:
    """Create a unique experiment folder with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{dataset_name}_{model_name}_{timestamp}"
    exp_path = os.path.join(base_path, exp_name)
    os.makedirs(exp_path, exist_ok=True)
    return exp_path


def save_config(config_path: str, save_path: str, overrides: dict):
    """Save the configuration file with overrides to experiment folder"""
    # Load original config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply overrides
    for key, value in overrides.items():
        if '.' in key:
            # Handle nested keys
            keys = key.split('.')
            d = config
            for k in keys[:-1]:
                if k not in d:
                    d[k] = {}
                d = d[k]
            d[keys[-1]] = value
        else:
            config[key] = value
    
    # Save to experiment folder
    config_save_path = os.path.join(save_path, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return config


def plot_losses(train_losses: dict, val_losses: dict, save_path: str, train_mode: str = 'jointly'):
    """Plot and save training losses"""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    if train_mode == 'jointly':
        # Single plot for joint training
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Training Progress - Joint Mode', fontsize=16)
        
        # Total loss
        ax = axes[0, 0]
        ax.plot(train_losses['total'], label='Train', linewidth=2)
        ax.plot(val_losses['total'], label='Validation', linewidth=2)
        ax.set_title('Total Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Reconstruction loss
        ax = axes[0, 1]
        ax.plot(train_losses['reconstruction'], label='Train', linewidth=2)
        ax.plot(val_losses['reconstruction'], label='Validation', linewidth=2)
        ax.set_title('Reconstruction Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Prediction loss
        ax = axes[1, 0]
        ax.plot(train_losses['prediction'], label='Train', linewidth=2)
        ax.plot(val_losses['prediction'], label='Validation', linewidth=2)
        ax.set_title('Prediction Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Latent regularization
        ax = axes[1, 1]
        ax.plot(train_losses['latent_reg'], label='Train', linewidth=2)
        ax.plot(val_losses['latent_reg'], label='Validation', linewidth=2)
        ax.set_title('Latent Regularization')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    elif train_mode == 'separately':
        # Plot for separate training stages
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Training Progress - Separate Mode', fontsize=16)
        
        # Stage 1: Reconstruction
        ax = axes[0]
        ax.plot(train_losses['stage1']['reconstruction'], label='Train', linewidth=2)
        ax.plot(val_losses['stage1']['reconstruction'], label='Validation', linewidth=2)
        ax.set_title('Stage 1: CAE Reconstruction')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Stage 2: Prediction
        ax = axes[1]
        ax.plot(train_losses['stage2']['prediction'], label='Train', linewidth=2)
        ax.plot(val_losses['stage2']['prediction'], label='Validation', linewidth=2)
        ax.set_title('Stage 2: Linear Predictor')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    elif train_mode == 'multiply':
        # Plot for multiply training stages
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Training Progress - Multiply Mode', fontsize=16)
        
        # Stage 1: Reconstruction
        ax = axes[0, 0]
        ax.plot(train_losses['stage1']['reconstruction'], label='Train', linewidth=2)
        ax.plot(val_losses['stage1']['reconstruction'], label='Validation', linewidth=2)
        ax.set_title('Stage 1: CAE Reconstruction')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Stage 2: Prediction
        ax = axes[0, 1]
        ax.plot(train_losses['stage2']['prediction'], label='Train', linewidth=2)
        ax.plot(val_losses['stage2']['prediction'], label='Validation', linewidth=2)
        ax.set_title('Stage 2: Linear Predictor')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Stage 3: Total loss
        ax = axes[0, 2]
        ax.plot(train_losses['stage3']['total'], label='Train', linewidth=2)
        ax.plot(val_losses['stage3']['total'], label='Validation', linewidth=2)
        ax.set_title('Stage 3: Total Loss (Fine-tuning)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Stage 3: Individual losses
        ax = axes[1, 0]
        ax.plot(train_losses['stage3']['reconstruction'], label='Train', linewidth=2)
        ax.plot(val_losses['stage3']['reconstruction'], label='Validation', linewidth=2)
        ax.set_title('Stage 3: Reconstruction Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        ax.plot(train_losses['stage3']['prediction'], label='Train', linewidth=2)
        ax.plot(val_losses['stage3']['prediction'], label='Validation', linewidth=2)
        ax.set_title('Stage 3: Prediction Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 2]
        ax.plot(train_losses['stage3']['latent_reg'], label='Train', linewidth=2)
        ax.plot(val_losses['stage3']['latent_reg'], label='Validation', linewidth=2)
        ax.set_title('Stage 3: Latent Regularization')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_path, 'training_curves.pdf'), bbox_inches='tight')
    plt.close()


def save_loss_data(train_losses: dict, val_losses: dict, save_path: str):
    """Save loss data as numpy arrays"""
    loss_dir = os.path.join(save_path, 'losses')
    os.makedirs(loss_dir, exist_ok=True)
    
    # Save as numpy arrays
    np.save(os.path.join(loss_dir, 'train_losses.npy'), train_losses)
    np.save(os.path.join(loss_dir, 'val_losses.npy'), val_losses)
    
    # Also save as JSON for easy reading
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    train_losses_json = convert_to_serializable(train_losses)
    val_losses_json = convert_to_serializable(val_losses)
    
    with open(os.path.join(loss_dir, 'losses.json'), 'w') as f:
        json.dump({
            'train_losses': train_losses_json,
            'val_losses': val_losses_json
        }, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description='Train dynamical system models')
    parser.add_argument('--model', type=str, required=True,
                      choices=['CAE_LinearMLP', 'CAE_WeakLinearMLP', 'DMD', 'KoopmanAE'],
                      help='Model type to train')
    parser.add_argument('--dataset', type=str, required=True,
                      choices=['kolmogorov', 'cylinder', 'chap'],
                      help='Dataset to use')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Automatically select device
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("CUDA not available, using CPU")

    # Continue with training setup...
    print(f"Running on device: {device}")
    
    # Load base configuration
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
    
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Create experiment folder
    base_save_path = base_config.get('training', {}).get('save_path', './results/checkpoints')
    exp_path = create_experiment_folder(base_save_path, args.model, args.dataset)
    
    print("="*70)
    print(f"Training {args.model} on {args.dataset} dataset")
    print(f"Configuration: {args.config}")
    print(f"Device: {device}")
    print(f"Experiment path: {exp_path}")
    print("="*70)
    
    # Prepare configuration overrides
    overrides = {
        'dataset.name': args.dataset,
        'model.type': args.model,
        'training.device': device,
        'training.save_path': exp_path,
        'dataset.random_seed': args.seed
    }
    
    # Dataset-specific configurations
    if args.dataset == 'cylinder':
        # Update cylinder-specific settings if needed
        if 'target_resolution' not in base_config.get('dataset', {}):
            overrides['dataset.target_resolution'] = [64, 64]
    elif args.dataset == 'chap':
        # Ensure CHAP has required parameters
        if 'chemical' not in base_config.get('dataset', {}):
            overrides['dataset.chemical'] = 'Cl'
        if 'target_shape' not in base_config.get('dataset', {}):
            overrides['dataset.target_shape'] = [128, 128]
    
    # Save configuration to experiment folder
    final_config = save_config(args.config, exp_path, overrides)
    
    # Train model based on type
    if args.model in ['CAE_LinearMLP', 'CAE_WeakLinearMLP']:
        # Train CAE-MLP models
        try:
            trainer, train_losses, val_losses = train_caemlp_from_config(
                args.config,
                **overrides
            )
            
            # Get training mode from config
            train_mode = final_config.get('training', {}).get('train_mode', 'jointly')
            
            # Plot and save losses
            plot_losses(train_losses, val_losses, exp_path, train_mode)
            save_loss_data(train_losses, val_losses, exp_path)
            
            # Copy best and latest checkpoints to organized structure
            checkpoint_files = [f for f in os.listdir(exp_path) if f.endswith('.pth')]
            
            # Organize checkpoints
            checkpoints_dir = os.path.join(exp_path, 'checkpoints')
            os.makedirs(checkpoints_dir, exist_ok=True)
            
            for ckpt_file in checkpoint_files:
                src = os.path.join(exp_path, ckpt_file)
                dst = os.path.join(checkpoints_dir, ckpt_file)
                shutil.move(src, dst)
            
            # Create summary
            summary = {
                'model': args.model,
                'dataset': args.dataset,
                'train_mode': train_mode,
                'device': device,
                'seed': args.seed,
                'experiment_path': exp_path,
                'config_file': args.config,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Add final performance metrics
            if train_mode == 'jointly':
                summary['best_val_loss'] = float(trainer.best_val_loss)
                summary['best_epoch'] = int(trainer.best_epoch)
                summary['final_train_loss'] = float(train_losses['total'][-1])
                summary['final_val_loss'] = float(val_losses['total'][-1])
            
            # Save summary
            with open(os.path.join(exp_path, 'summary.json'), 'w') as f:
                json.dump(summary, f, indent=4)
            
            print("\n" + "="*70)
            print("Training completed successfully!")
            print(f"Results saved to: {exp_path}")
            print("="*70)
            
            # Print final metrics
            if train_mode == 'jointly':
                print(f"\nBest validation loss: {trainer.best_val_loss:.6f} at epoch {trainer.best_epoch}")
                print(f"Final training loss: {train_losses['total'][-1]:.6f}")
                print(f"Final validation loss: {val_losses['total'][-1]:.6f}")
            else:
                print(f"\nTraining completed with {train_mode} mode")
                print("Check training_curves.png for detailed loss progression")
            
        except Exception as e:
            print(f"\nError during training: {e}")
            import traceback
            traceback.print_exc()
            
            # Save error log
            with open(os.path.join(exp_path, 'error.log'), 'w') as f:
                f.write(f"Error: {str(e)}\n\n")
                f.write(traceback.format_exc())
            
            raise
    
    elif args.model == 'DMD':
        # TODO: Import and train DMD model
        print("DMD model training not yet implemented")
        raise NotImplementedError("DMD model training will be implemented separately")
    
    elif args.model == 'KoopmanAE':
        # TODO: Import and train KoopmanAE model
        print("KoopmanAE model training not yet implemented")
        raise NotImplementedError("KoopmanAE model training will be implemented separately")
    
    print("\nTraining pipeline completed!")


if __name__ == "__main__":
    main()