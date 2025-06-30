import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import json
import yaml
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import model and utils
import sys

current_file_path = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(current_file_path, '..', '..', '..'))
sys.path.insert(0, project_root)

from src.models.CAE_MLP.caemlp import CAE_DMD
from src.models.CAE_MLP.utils import CAEMLPLoss, reshape_data_for_cae, create_caemlp_dataloaders


class CAEDMDTrainer:
    """Specialized trainer for CAE-DMD models"""
    
    def __init__(self, 
                 model: CAE_DMD,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 epochs: int = 100,
                 lr: float = 0.001,
                 optimizer_config: Dict[str, Any] = None,
                 scheduler_config: Optional[Dict[str, Any]] = None,
                 loss_config: Optional[Dict[str, Any]] = None,
                 device: str = 'cuda',
                 save_path: str = './checkpoints',
                 verbose: bool = True,
                 config: Optional[Dict[str, Any]] = None,
                 dmd_min_samples: int = 1000,
                 dmd_buffer_collection_epochs: int = 20,
                 dmd_retrain_interval: int = 10):
        """
        Initialize CAE-DMD trainer
        
        Args:
            model: CAE_DMD model instance
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Total number of training epochs
            lr: Learning rate for CAE training
            optimizer_config: Optimizer configuration
            scheduler_config: Learning rate scheduler configuration
            loss_config: Loss function configuration
            device: Device to train on
            save_path: Path to save checkpoints
            verbose: Whether to print training progress
            config: Full configuration dictionary
            dmd_min_samples: Minimum samples required to fit DMD
            dmd_buffer_collection_epochs: Epochs for collecting latent data
            dmd_retrain_interval: Epochs between DMD retraining
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.save_path = save_path
        self.verbose = verbose
        self.config = config or {}
        
        # DMD-specific parameters
        self.dmd_min_samples = dmd_min_samples
        self.dmd_buffer_collection_epochs = dmd_buffer_collection_epochs
        self.dmd_retrain_interval = dmd_retrain_interval
        
        # Store optimizer config
        self.optimizer_config = optimizer_config or {'type': 'Adam', 'params': {}}
        self.scheduler_config = scheduler_config
        
        # Setup loss function
        loss_config = loss_config or {}
        self.loss_fn = CAEMLPLoss(**loss_config)
        
        # Get dataset info from config
        self.dataset_name = config.get('dataset', {}).get('name', 'unknown')
        self.target_shape = config.get('dataset', {}).get('target_shape', None)
        
        # Initialize training history
        self._init_training_history()
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # Create save directory
        os.makedirs(self.save_path, exist_ok=True)
        
        # Training phase tracking
        self.current_phase = 'cae_training'  # 'cae_training', 'data_collection', 'joint_training'
        self.dmd_fitted_epoch = None
        self.last_dmd_retrain_epoch = 0
    
    def _init_training_history(self):
        """Initialize training history"""
        self.train_losses = {
            'cae_reconstruction': [],
            'total_after_dmd': [],
            'reconstruction_after_dmd': [],
            'prediction_after_dmd': []
        }
        self.val_losses = {
            'cae_reconstruction': [],
            'total_after_dmd': [],
            'reconstruction_after_dmd': [],
            'prediction_after_dmd': []
        }
        self.dmd_stats = {
            'buffer_size': [],
            'fitting_epochs': [],
            'reconstruction_errors': []
        }
    
    def _create_optimizer(self, parameters, lr=None) -> optim.Optimizer:
        """Create optimizer for specific parameters"""
        opt_type = self.optimizer_config.get('type', 'Adam')
        opt_params = self.optimizer_config.get('params', {}).copy()
        opt_params['lr'] = lr or self.lr
        
        if opt_type == 'Adam':
            return optim.Adam(parameters, **opt_params)
        elif opt_type == 'AdamW':
            return optim.AdamW(parameters, **opt_params)
        elif opt_type == 'SGD':
            return optim.SGD(parameters, **opt_params)
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")
    
    def _create_scheduler(self, optimizer, epochs=None) -> _LRScheduler:
        """Create learning rate scheduler"""
        if not self.scheduler_config:
            return None
            
        sched_type = self.scheduler_config.get('type', 'StepLR')
        sched_params = self.scheduler_config.get('params', {}).copy()
        
        if sched_type == 'StepLR':
            return optim.lr_scheduler.StepLR(optimizer, **sched_params)
        elif sched_type == 'ExponentialLR':
            return optim.lr_scheduler.ExponentialLR(optimizer, **sched_params)
        elif sched_type == 'CosineAnnealingLR':
            if epochs:
                sched_params['T_max'] = epochs
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, **sched_params)
        elif sched_type == 'ReduceLROnPlateau':
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **sched_params)
        else:
            raise ValueError(f"Unknown scheduler type: {sched_type}")
    
    def _train_cae_only(self, epoch: int, optimizer: optim.Optimizer) -> float:
        """Train CAE reconstruction only"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} - CAE Training', 
                   disable=not self.verbose)
        
        for x_t, _ in pbar:
            x_t = x_t.to(self.device)
            x_t = reshape_data_for_cae(x_t, self.dataset_name, self.target_shape)
            
            optimizer.zero_grad()
            
            # CAE forward pass only
            x_recon, _ = self.model.cae(x_t)
            loss = self.loss_fn.reconstruction_loss(x_t, x_recon)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.cae.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if pbar.n % 10 == 0:
                pbar.set_postfix({'recon_loss': f"{loss.item():.4f}"})
        
        return total_loss / len(self.train_loader)
    
    def _collect_latent_data(self, epoch: int):
        """Collect latent data for DMD training"""
        self.model.eval()
        collected_pairs = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} - Collecting Latent Data', 
                   disable=not self.verbose)
        
        with torch.no_grad():
            for x_t, x_next in pbar:
                x_t = x_t.to(self.device)
                x_next = x_next.to(self.device)
                x_t = reshape_data_for_cae(x_t, self.dataset_name, self.target_shape)
                x_next = reshape_data_for_cae(x_next, self.dataset_name, self.target_shape)
                
                # Collect latent data pairs
                self.model.collect_latent_data(x_t, x_next)
                collected_pairs += x_t.shape[0]
                
                pbar.set_postfix({
                    'buffer_size': len(self.model.latent_buffer),
                    'collected': collected_pairs
                })
        
        buffer_size = len(self.model.latent_buffer)
        if self.verbose:
            print(f"Collected {collected_pairs} pairs, total buffer size: {buffer_size}")
        
        return buffer_size
    
    def _train_with_dmd(self, epoch: int, optimizer: optim.Optimizer) -> Dict[str, float]:
        """Train with DMD predictions"""
        self.model.train()
        epoch_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'prediction': 0.0
        }
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} - Joint Training', 
                   disable=not self.verbose)
        
        for x_t, x_next in pbar:
            x_t = x_t.to(self.device)
            x_next = x_next.to(self.device)
            x_t = reshape_data_for_cae(x_t, self.dataset_name, self.target_shape)
            x_next = reshape_data_for_cae(x_next, self.dataset_name, self.target_shape)
            
            optimizer.zero_grad()
            
            # Forward pass with DMD prediction
            x_t_recon, x_next_pred, z_t, z_next = self.model(x_t)
            
            if x_next_pred is not None:
                # Compute losses
                recon_loss = self.loss_fn.reconstruction_loss(x_t, x_t_recon)
                pred_loss = self.loss_fn.prediction_loss(x_next, x_next_pred)
                total_loss = recon_loss + pred_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.cae.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Update epoch losses
                epoch_losses['total'] += total_loss.item()
                epoch_losses['reconstruction'] += recon_loss.item()
                epoch_losses['prediction'] += pred_loss.item()
                
                # Continue collecting data for DMD retraining
                with torch.no_grad():
                    self.model.collect_latent_data(x_t, x_next)
                
                pbar.set_postfix({
                    'total': f"{total_loss.item():.4f}",
                    'recon': f"{recon_loss.item():.4f}",
                    'pred': f"{pred_loss.item():.4f}"
                })
            else:
                # DMD not fitted yet, only train reconstruction
                recon_loss = self.loss_fn.reconstruction_loss(x_t, x_t_recon)
                recon_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.cae.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_losses['reconstruction'] += recon_loss.item()
                
                # Collect data for DMD
                with torch.no_grad():
                    self.model.collect_latent_data(x_t, x_next)
                
                pbar.set_postfix({'recon': f"{recon_loss.item():.4f}"})
        
        # Average losses
        n_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        
        return epoch_losses
    
    def _validate_cae_only(self) -> float:
        """Validate CAE reconstruction only"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for x_t, _ in self.val_loader:
                x_t = x_t.to(self.device)
                x_t = reshape_data_for_cae(x_t, self.dataset_name, self.target_shape)
                
                x_recon, _ = self.model.cae(x_t)
                loss = self.loss_fn.reconstruction_loss(x_t, x_recon)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def _validate_with_dmd(self) -> Dict[str, float]:
        """Validate with DMD predictions"""
        self.model.eval()
        val_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'prediction': 0.0
        }
        
        with torch.no_grad():
            for x_t, x_next in self.val_loader:
                x_t = x_t.to(self.device)
                x_next = x_next.to(self.device)
                x_t = reshape_data_for_cae(x_t, self.dataset_name, self.target_shape)
                x_next = reshape_data_for_cae(x_next, self.dataset_name, self.target_shape)
                
                x_t_recon, x_next_pred, z_t, z_next = self.model(x_t)
                
                recon_loss = self.loss_fn.reconstruction_loss(x_t, x_t_recon)
                val_losses['reconstruction'] += recon_loss.item()
                
                if x_next_pred is not None:
                    pred_loss = self.loss_fn.prediction_loss(x_next, x_next_pred)
                    val_losses['prediction'] += pred_loss.item()
                    val_losses['total'] += (recon_loss + pred_loss).item()
                else:
                    val_losses['total'] += recon_loss.item()
        
        # Average losses
        n_batches = len(self.val_loader)
        for key in val_losses:
            val_losses[key] /= n_batches
        
        return val_losses
    
    def _try_fit_dmd(self, epoch: int) -> bool:
        """Try to fit DMD model"""
        buffer_size = len(self.model.latent_buffer)
        
        if buffer_size >= self.dmd_min_samples:
            if self.verbose:
                print(f"\nAttempting to fit DMD at epoch {epoch} with {buffer_size} samples...")
            
            success = self.model.fit_dmd(min_samples=self.dmd_min_samples)
            
            if success:
                self.dmd_fitted_epoch = epoch
                self.last_dmd_retrain_epoch = epoch
                self.current_phase = 'joint_training'
                
                # Get DMD reconstruction error
                dmd_errors = self.model.get_dmd_reconstruction_error()
                if dmd_errors:
                    self.dmd_stats['reconstruction_errors'].append(dmd_errors)
                    if self.verbose:
                        print(f"DMD reconstruction errors: {dmd_errors}")
                
                # Save DMD model
                dmd_path = os.path.join(self.save_path, f'dmd_epoch_{epoch}.npz')
                self.model.save_dmd(dmd_path)
                
                return True
        
        return False
    
    def _should_retrain_dmd(self, epoch: int) -> bool:
        """Check if DMD should be retrained"""
        return (self.model.dmd_fitted and 
                epoch - self.last_dmd_retrain_epoch >= self.dmd_retrain_interval and
                len(self.model.latent_buffer) >= self.dmd_min_samples)
    
    def _save_checkpoint(self, epoch: int, train_losses: Dict, val_losses: Dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'dmd_stats': self.dmd_stats,
            'current_phase': self.current_phase,
            'dmd_fitted_epoch': self.dmd_fitted_epoch,
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(self.save_path, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.save_path, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
            if self.verbose:
                print(f"Saved best model with validation loss: {self.best_val_loss:.6f}")
    
    def train(self) -> Tuple[Dict[str, List], Dict[str, List]]:
        """
        Train the CAE-DMD model
        
        Training process:
        1. Train CAE for reconstruction
        2. Collect latent data while continuing CAE training
        3. Fit DMD when enough data is collected
        4. Joint training with DMD predictions
        5. Periodically retrain DMD with new data
        
        Returns:
            train_losses: Dictionary of training losses
            val_losses: Dictionary of validation losses
        """
        if self.verbose:
            print(f"Starting CAE-DMD training")
            print(f"Dataset: {self.dataset_name}")
            print(f"Device: {self.device}")
            print(f"Total epochs: {self.epochs}")
            print(f"DMD min samples: {self.dmd_min_samples}")
            print(f"Save path: {self.save_path}")
            print("-" * 50)
        
        # Create optimizer and scheduler for CAE
        optimizer = self._create_optimizer(self.model.cae.parameters())
        scheduler = self._create_scheduler(optimizer, self.epochs)
        
        for epoch in range(1, self.epochs + 1):
            
            # Phase 1: CAE training only (before data collection)
            if self.current_phase == 'cae_training':
                train_loss = self._train_cae_only(epoch, optimizer)
                val_loss = self._validate_cae_only()
                
                self.train_losses['cae_reconstruction'].append(train_loss)
                self.val_losses['cae_reconstruction'].append(val_loss)
                
                if self.verbose:
                    print(f"Epoch {epoch} - CAE Only | Train: {train_loss:.6f}, Val: {val_loss:.6f}")
                
                # Start data collection phase
                if epoch >= self.dmd_buffer_collection_epochs:
                    self.current_phase = 'data_collection'
                    if self.verbose:
                        print("Starting latent data collection phase...")
            
            # Phase 2: Data collection while training
            elif self.current_phase == 'data_collection':
                # Continue CAE training while collecting data
                train_losses = self._train_with_dmd(epoch, optimizer)
                val_losses = self._validate_with_dmd()
                
                # Record losses
                self.train_losses['cae_reconstruction'].append(train_losses['reconstruction'])
                self.val_losses['cae_reconstruction'].append(val_losses['reconstruction'])
                
                buffer_size = len(self.model.latent_buffer)
                self.dmd_stats['buffer_size'].append(buffer_size)
                
                if self.verbose:
                    print(f"Epoch {epoch} - Data Collection | Train: {train_losses['reconstruction']:.6f}, "
                          f"Val: {val_losses['reconstruction']:.6f}, Buffer: {buffer_size}")
                
                # Try to fit DMD
                if self._try_fit_dmd(epoch):
                    if self.verbose:
                        print(f"DMD fitted successfully at epoch {epoch}!")
                        print("Switching to joint training phase...")
            
            # Phase 3: Joint training with DMD
            elif self.current_phase == 'joint_training':
                # Check if DMD needs retraining
                if self._should_retrain_dmd(epoch):
                    if self.verbose:
                        print(f"Retraining DMD at epoch {epoch}...")
                    self.model.fit_dmd(min_samples=self.dmd_min_samples)
                    self.last_dmd_retrain_epoch = epoch
                    
                    # Save retrained DMD
                    dmd_path = os.path.join(self.save_path, f'dmd_retrain_epoch_{epoch}.npz')
                    self.model.save_dmd(dmd_path)
                
                # Train with DMD predictions
                train_losses = self._train_with_dmd(epoch, optimizer)
                val_losses = self._validate_with_dmd()
                
                # Record losses
                self.train_losses['total_after_dmd'].append(train_losses['total'])
                self.train_losses['reconstruction_after_dmd'].append(train_losses['reconstruction'])
                self.train_losses['prediction_after_dmd'].append(train_losses['prediction'])
                
                self.val_losses['total_after_dmd'].append(val_losses['total'])
                self.val_losses['reconstruction_after_dmd'].append(val_losses['reconstruction'])
                self.val_losses['prediction_after_dmd'].append(val_losses['prediction'])
                
                # Track best model
                current_val_loss = val_losses['total']
                is_best = current_val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = current_val_loss
                    self.best_epoch = epoch
                
                if self.verbose:
                    print(f"Epoch {epoch} - Joint Training | Train: {train_losses['total']:.6f}, "
                          f"Val: {val_losses['total']:.6f} | "
                          f"Recon: {train_losses['reconstruction']:.6f}, "
                          f"Pred: {train_losses['prediction']:.6f}")
                    if is_best:
                        print(f"New best model! Val loss: {self.best_val_loss:.6f}")
                
                # Save checkpoint
                self._save_checkpoint(epoch, train_losses, val_losses, is_best)
            
            # Learning rate scheduling
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if self.current_phase == 'joint_training':
                        scheduler.step(val_losses['total'])
                    else:
                        scheduler.step(val_loss if 'val_loss' in locals() else val_losses['reconstruction'])
                else:
                    scheduler.step()
        
        # Save final training history
        self._save_training_history()
        
        if self.verbose:
            print("\n" + "="*50)
            print("CAE-DMD training completed!")
            if self.dmd_fitted_epoch:
                print(f"DMD fitted at epoch {self.dmd_fitted_epoch}")
                print(f"Best model at epoch {self.best_epoch} with val loss {self.best_val_loss:.6f}")
            else:
                print("DMD was not fitted - insufficient data collected")
            print(f"Final buffer size: {len(self.model.latent_buffer)}")
        
        return self.train_losses, self.val_losses
    
    def _save_training_history(self):
        """Save training history to JSON"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'dmd_stats': self.dmd_stats,
            'training_phases': {
                'current_phase': self.current_phase,
                'dmd_fitted_epoch': self.dmd_fitted_epoch,
                'dmd_min_samples': self.dmd_min_samples,
                'buffer_collection_epochs': self.dmd_buffer_collection_epochs,
                'retrain_interval': self.dmd_retrain_interval
            },
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
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
        
        history = convert_to_serializable(history)
        
        history_path = os.path.join(self.save_path, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)


def train_cae_dmd_from_config(config_path: str, **kwargs):
    """
    Train CAE-DMD model from configuration file
    
    Args:
        config_path: Path to configuration YAML file
        **kwargs: Override configuration parameters
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with kwargs
    for key, value in kwargs.items():
        if '.' in key:
            # Handle nested keys like 'dataset.batch_size'
            keys = key.split('.')
            d = config
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
        else:
            config[key] = value
    
    # Create data loaders
    train_loader, val_loader, data_info = create_caemlp_dataloaders(config['dataset'])
    
    # Update config with data info
    config['data_info'] = data_info
    
    # Create model
    model_config = config['model']
    
    # Get input shape and channels from data info
    input_channels = data_info.get('num_channels', 1)
    if 'spatial_resolution' in data_info:
        input_shape = data_info['spatial_resolution']
    else:
        input_shape = config['dataset'].get('target_shape', (64, 64))
    
    # Create CAE_DMD model
    model = CAE_DMD(
        input_channels=input_channels,
        input_shape=input_shape,
        **model_config['params']
    )
    
    # Create trainer
    trainer_config = config['training']
    trainer = CAEDMDTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=trainer_config['epochs'],
        lr=trainer_config['learning_rate'],
        optimizer_config=trainer_config['optimizer'],
        scheduler_config=trainer_config.get('scheduler', None),
        loss_config=trainer_config.get('loss', None),
        device=trainer_config['device'],
        save_path=trainer_config['save_path'],
        verbose=trainer_config.get('verbose', True),
        config=config,
        dmd_min_samples=trainer_config.get('dmd_min_samples', 1000),
        dmd_buffer_collection_epochs=trainer_config.get('dmd_buffer_collection_epochs', 20),
        dmd_retrain_interval=trainer_config.get('dmd_retrain_interval', 10)
    )
    
    # Train model
    train_losses, val_losses = trainer.train()
    
    return trainer, train_losses, val_losses


if __name__ == "__main__":
    # Debug example
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CAE-DMD model')
    parser.add_argument('--config', type=str, default='configs/CAE_DMD.yaml',
                      help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default='kolmogorov',
                      choices=['kolmogorov', 'cylinder', 'chap'],
                      help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size')
    parser.add_argument('--dmd_min_samples', type=int, default=500,
                      help='Minimum samples for DMD fitting')
    
    args = parser.parse_args()
    
    # Override config for debugging
    overrides = {
        'dataset.name': args.dataset,
        'dataset.data_path': 'data',
        'dataset.batch_size': args.batch_size,
        'model.type': 'CAE_DMD',
        'training.epochs': args.epochs,
        'training.dmd_min_samples': args.dmd_min_samples,
        'training.dmd_buffer_collection_epochs': max(args.epochs // 4, 5),
        'training.dmd_retrain_interval': max(args.epochs // 10, 5),
        'training.save_path': f'results/checkpoints/CAE_DMD/{args.dataset}_debug'
    }
    
    if args.dataset == 'chap':
        overrides['dataset.chemical'] = 'Cl'
        overrides['dataset.target_shape'] = (64, 64)
    
    print(f"Training CAE_DMD on {args.dataset} dataset")
    print(f"Total epochs: {args.epochs}")
    print(f"DMD min samples: {args.dmd_min_samples}")
    
    try:
        trainer, train_losses, val_losses = train_cae_dmd_from_config(
            args.config,
            **overrides
        )
        
        # Print final results
        print("\n" + "="*50)
        print("Training Results Summary:")
        print("="*50)
        
        if trainer.dmd_fitted_epoch:
            print(f"✓ DMD successfully fitted at epoch {trainer.dmd_fitted_epoch}")
            print(f"✓ Best model at epoch {trainer.best_epoch}")
            print(f"✓ Best validation loss: {trainer.best_val_loss:.6f}")
            print(f"✓ Final buffer size: {len(trainer.model.latent_buffer)}")
            
            # Show phase breakdown
            print(f"\nTraining Phases:")
            print(f"  CAE Training: Epochs 1-{trainer.dmd_buffer_collection_epochs}")
            print(f"  Data Collection: Epochs {trainer.dmd_buffer_collection_epochs+1}-{trainer.dmd_fitted_epoch}")
            print(f"  Joint Training: Epochs {trainer.dmd_fitted_epoch+1}-{args.epochs}")
            
            # Show final losses
            if train_losses['total_after_dmd']:
                print(f"\nFinal Joint Training Losses:")
                print(f"  Total: {train_losses['total_after_dmd'][-1]:.6f}")
                print(f"  Reconstruction: {train_losses['reconstruction_after_dmd'][-1]:.6f}")
                print(f"  Prediction: {train_losses['prediction_after_dmd'][-1]:.6f}")
            
            # DMD statistics
            if trainer.dmd_stats['reconstruction_errors']:
                latest_dmd_error = trainer.dmd_stats['reconstruction_errors'][-1]
                print(f"\nDMD Reconstruction Quality:")
                for metric, value in latest_dmd_error.items():
                    print(f"  {metric}: {value:.6f}")
        else:
            print("⚠ Warning: DMD was not fitted during training")
            print(f"  Buffer size: {len(trainer.model.latent_buffer)}")
            print(f"  Required: {trainer.dmd_min_samples}")
            print("  Consider:")
            print("    - Increasing number of epochs")
            print("    - Reducing dmd_min_samples")
            print("    - Checking data loader size")
        
        # Model saving info
        print(f"\nModel saved to: {trainer.save_path}")
        print(f"  - Latest checkpoint: latest_checkpoint.pth")
        if trainer.best_epoch > 0:
            print(f"  - Best checkpoint: best_checkpoint.pth")
        if trainer.dmd_fitted_epoch:
            print(f"  - DMD models: dmd_epoch_*.npz")
        print(f"  - Training history: training_history.json")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to provide helpful debugging info
        print(f"\nDebugging Information:")
        print(f"  Config file: {args.config}")
        print(f"  Dataset: {args.dataset}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Epochs: {args.epochs}")
        print(f"  DMD min samples: {args.dmd_min_samples}")
        
        # Suggest solutions
        print(f"\nPossible solutions:")
        print(f"  1. Check if config file exists and is valid")
        print(f"  2. Verify dataset path in config")
        print(f"  3. Ensure sufficient GPU memory for batch size")
        print(f"  4. Check data format compatibility")
        raise