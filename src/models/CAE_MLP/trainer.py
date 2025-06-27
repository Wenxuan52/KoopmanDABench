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
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import model and utils
import sys

current_file_path = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(current_file_path, '..', '..', '..'))
sys.path.insert(0, project_root)

from src.models.CAE_MLP.caemlp import CAE_LinearMLP, CAE_WeakLinearMLP
from src.models.CAE_MLP.utils import CAEMLPLoss, reshape_data_for_cae, create_caemlp_dataloaders


class CAEMLPTrainer:
    """Trainer for CAE-MLP models"""
    
    def __init__(self, 
                 model: Union[CAE_LinearMLP, CAE_WeakLinearMLP],
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 train_mode: str = 'jointly',
                 epochs: int = 100,
                 lr: float = 0.001,
                 optimizer_config: Dict[str, Any] = None,
                 scheduler_config: Optional[Dict[str, Any]] = None,
                 loss_config: Optional[Dict[str, Any]] = None,
                 device: str = 'cuda',
                 save_path: str = './checkpoints',
                 verbose: bool = True,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize trainer
        
        Args:
            model: CAE-MLP model instance
            train_loader: Training data loader
            val_loader: Validation data loader
            train_mode: Training mode ('separately', 'jointly', 'multiply')
            epochs: Number of training epochs (or epochs per stage for multi-stage training)
            lr: Learning rate
            optimizer_config: Optimizer configuration
            scheduler_config: Learning rate scheduler configuration
            loss_config: Loss function configuration
            device: Device to train on
            save_path: Path to save checkpoints
            verbose: Whether to print training progress
            config: Full configuration dictionary
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_mode = train_mode
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.save_path = save_path
        self.verbose = verbose
        self.config = config or {}
        
        # Store optimizer config for creating stage-specific optimizers
        self.optimizer_config = optimizer_config or {'type': 'Adam', 'params': {}}
        self.scheduler_config = scheduler_config
        
        # Setup loss function
        loss_config = loss_config or {}
        self.loss_fn = CAEMLPLoss(**loss_config)
        
        # Check if this is weak linear model
        self.is_weak_linear = isinstance(model, CAE_WeakLinearMLP)
        
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
        
        # Initialize linear predictor with identity matrix
        self._init_linear_predictor()
    
    def _init_training_history(self):
        """Initialize training history based on training mode"""
        if self.train_mode == 'separately':
            # Stage 1: CAE reconstruction
            self.train_losses_stage1 = {'reconstruction': []}
            self.val_losses_stage1 = {'reconstruction': []}
            
            # Stage 2: Linear predictor
            self.train_losses_stage2 = {'prediction': []}
            self.val_losses_stage2 = {'prediction': []}
            
        elif self.train_mode == 'multiply':
            # Stage 1: CAE reconstruction
            self.train_losses_stage1 = {'reconstruction': []}
            self.val_losses_stage1 = {'reconstruction': []}
            
            # Stage 2: Linear predictor
            self.train_losses_stage2 = {'prediction': []}
            self.val_losses_stage2 = {'prediction': []}
            
            # Stage 3: Joint fine-tuning
            self.train_losses_stage3 = {
                'total': [],
                'reconstruction': [],
                'prediction': [],
                'latent_reg': []
            }
            self.val_losses_stage3 = {
                'total': [],
                'reconstruction': [],
                'prediction': [],
                'latent_reg': []
            }
            if self.is_weak_linear:
                self.train_losses_stage3['linear_constraint'] = []
                self.val_losses_stage3['linear_constraint'] = []
        
        else:  # jointly
            self.train_losses = {
                'total': [],
                'reconstruction': [],
                'prediction': [],
                'latent_reg': []
            }
            self.val_losses = {
                'total': [],
                'reconstruction': [],
                'prediction': [],
                'latent_reg': []
            }
            if self.is_weak_linear:
                self.train_losses['linear_constraint'] = []
                self.val_losses['linear_constraint'] = []
    
    def _init_linear_predictor(self):
        """Initialize linear predictor with identity matrix"""
        if hasattr(self.model, 'dynamics') and hasattr(self.model.dynamics, 'linear'):
            # For CAE_LinearMLP
            latent_dim = self.model.dynamics.linear.weight.shape[0]
            with torch.no_grad():
                self.model.dynamics.linear.weight.copy_(torch.eye(latent_dim))
                self.model.dynamics.linear.bias.zero_()
        elif hasattr(self.model, 'dynamics') and hasattr(self.model.dynamics, 'linear_approx'):
            # For CAE_WeakLinearMLP
            latent_dim = self.model.dynamics.linear_approx.weight.shape[0]
            with torch.no_grad():
                self.model.dynamics.linear_approx.weight.copy_(torch.eye(latent_dim))
                self.model.dynamics.linear_approx.bias.zero_()
    
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
    
    def _train_cae_reconstruction(self, epochs: int):
        """Stage 1: Train CAE for reconstruction only"""
        if self.verbose:
            print("\n" + "="*50)
            print("Stage 1: Training CAE for reconstruction")
            print("="*50)
        
        # Create optimizer for CAE only
        optimizer = self._create_optimizer(self.model.cae.parameters())
        scheduler = self._create_scheduler(optimizer, epochs)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(1, epochs + 1):
            # Training
            self.model.train()
            epoch_loss = 0.0
            
            pbar = tqdm(self.train_loader, desc=f'Stage 1 - Epoch {epoch}/{epochs}', 
                       disable=not self.verbose)
            
            for x_t, _ in pbar:
                x_t = x_t.to(self.device)
                x_t = reshape_data_for_cae(x_t, self.dataset_name, self.target_shape)
                
                optimizer.zero_grad()
                
                # CAE forward pass
                x_recon, _ = self.model.cae(x_t)
                loss = self.loss_fn.reconstruction_loss(x_t, x_recon)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.cae.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                
                if pbar.n % 10 == 0:
                    pbar.set_postfix({'recon_loss': f"{loss.item():.4f}"})
            
            # Validation
            val_loss = self._validate_reconstruction()
            
            # Record losses
            train_loss = epoch_loss / len(self.train_loader)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Learning rate scheduling
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            if self.verbose:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # Store losses
        self.train_losses_stage1['reconstruction'] = train_losses
        self.val_losses_stage1['reconstruction'] = val_losses
    
    def _train_linear_predictor(self, epochs: int):
        """Stage 2: Train linear predictor with fixed encoder"""
        if self.verbose:
            print("\n" + "="*50)
            print("Stage 2: Training linear predictor with fixed encoder")
            print("="*50)
        
        # Freeze CAE
        for param in self.model.cae.parameters():
            param.requires_grad = False
        
        # Create optimizer for dynamics only
        optimizer = self._create_optimizer(self.model.dynamics.parameters())
        scheduler = self._create_scheduler(optimizer, epochs)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(1, epochs + 1):
            # Training
            self.model.eval()  # Keep CAE in eval mode
            self.model.dynamics.train()  # Only dynamics in train mode
            
            epoch_loss = 0.0
            
            pbar = tqdm(self.train_loader, desc=f'Stage 2 - Epoch {epoch}/{epochs}', 
                       disable=not self.verbose)
            
            for x_t, x_next in pbar:
                x_t = x_t.to(self.device)
                x_next = x_next.to(self.device)
                x_t = reshape_data_for_cae(x_t, self.dataset_name, self.target_shape)
                x_next = reshape_data_for_cae(x_next, self.dataset_name, self.target_shape)
                
                optimizer.zero_grad()
                
                # Encode with no gradients
                with torch.no_grad():
                    z_t = self.model.cae.encode(x_t)
                    z_next_true = self.model.cae.encode(x_next)
                
                # Predict next latent state
                z_next_pred = self.model.dynamics(z_t)
                
                # Latent prediction loss
                loss = self.loss_fn.mse_loss(z_next_pred, z_next_true)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.dynamics.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                
                if pbar.n % 10 == 0:
                    pbar.set_postfix({'pred_loss': f"{loss.item():.4f}"})
            
            # Validation
            val_loss = self._validate_predictor()
            
            # Record losses
            train_loss = epoch_loss / len(self.train_loader)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Learning rate scheduling
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            if self.verbose:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # Store losses
        self.train_losses_stage2['prediction'] = train_losses
        self.val_losses_stage2['prediction'] = val_losses
        
        # Unfreeze CAE for potential stage 3
        for param in self.model.cae.parameters():
            param.requires_grad = True
    
    def _train_jointly(self, epochs: int):
        """Train CAE and predictor jointly"""
        if self.verbose:
            print("\n" + "="*50)
            print("Training CAE and predictor jointly")
            print("="*50)
        
        # Create optimizer for all parameters
        optimizer = self._create_optimizer(self.model.parameters())
        scheduler = self._create_scheduler(optimizer, epochs)
        
        for epoch in range(1, epochs + 1):
            # Train
            train_epoch_losses = self._train_epoch_jointly(optimizer)
            
            # Validate
            val_epoch_losses = self._validate_jointly()
            
            # Update history
            for key in train_epoch_losses:
                self.train_losses[key].append(train_epoch_losses[key])
            for key in val_epoch_losses:
                self.val_losses[key].append(val_epoch_losses[key])
            
            # Check if best model
            current_val_loss = val_epoch_losses['total']
            is_best = current_val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = current_val_loss
                self.best_epoch = epoch
            
            # Save checkpoint
            self._save_checkpoint(epoch, is_best, stage='joint')
            
            # Learning rate scheduling
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(current_val_loss)
                else:
                    scheduler.step()
            
            # Print progress
            if self.verbose:
                self._print_epoch_progress(epoch, epochs, train_epoch_losses, 
                                         val_epoch_losses, optimizer)
    
    def _train_epoch_jointly(self, optimizer) -> Dict[str, float]:
        """Train for one epoch (joint mode)"""
        self.model.train()
        epoch_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'prediction': 0.0,
            'latent_reg': 0.0
        }
        if self.is_weak_linear:
            epoch_losses['linear_constraint'] = 0.0
        
        pbar = tqdm(self.train_loader, desc='Training', disable=not self.verbose)
        
        for batch_idx, (x_t, x_next) in enumerate(pbar):
            x_t = x_t.to(self.device)
            x_next = x_next.to(self.device)
            x_t = reshape_data_for_cae(x_t, self.dataset_name, self.target_shape)
            x_next = reshape_data_for_cae(x_next, self.dataset_name, self.target_shape)
            
            optimizer.zero_grad()
            
            # Forward pass
            if self.is_weak_linear:
                x_t_recon, x_next_pred, z_t, z_next, z_next_linear = self.model(x_t)
                losses = self.loss_fn.compute_cae_weaklinear_loss(
                    x_t, x_next, x_t_recon, x_next_pred, z_t, z_next, z_next_linear
                )
            else:
                x_t_recon, x_next_pred, z_t, z_next = self.model(x_t)
                losses = self.loss_fn.compute_cae_linear_loss(
                    x_t, x_next, x_t_recon, x_next_pred, z_t, z_next
                )
            
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update epoch losses
            for key, value in losses.items():
                epoch_losses[key] += value.item()
            
            # Update progress bar
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'loss': f"{losses['total'].item():.4f}",
                    'recon': f"{losses['reconstruction'].item():.4f}",
                    'pred': f"{losses['prediction'].item():.4f}"
                })
        
        # Average losses
        n_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        
        return epoch_losses
    
    def _validate_reconstruction(self) -> float:
        """Validate CAE reconstruction"""
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
    
    def _validate_predictor(self) -> float:
        """Validate linear predictor"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for x_t, x_next in self.val_loader:
                x_t = x_t.to(self.device)
                x_next = x_next.to(self.device)
                x_t = reshape_data_for_cae(x_t, self.dataset_name, self.target_shape)
                x_next = reshape_data_for_cae(x_next, self.dataset_name, self.target_shape)
                
                z_t = self.model.cae.encode(x_t)
                z_next_true = self.model.cae.encode(x_next)
                z_next_pred = self.model.dynamics(z_t)
                
                loss = self.loss_fn.mse_loss(z_next_pred, z_next_true)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def _validate_jointly(self) -> Dict[str, float]:
        """Validate model (joint mode)"""
        self.model.eval()
        val_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'prediction': 0.0,
            'latent_reg': 0.0
        }
        if self.is_weak_linear:
            val_losses['linear_constraint'] = 0.0
        
        with torch.no_grad():
            for x_t, x_next in self.val_loader:
                x_t = x_t.to(self.device)
                x_next = x_next.to(self.device)
                x_t = reshape_data_for_cae(x_t, self.dataset_name, self.target_shape)
                x_next = reshape_data_for_cae(x_next, self.dataset_name, self.target_shape)
                
                if self.is_weak_linear:
                    x_t_recon, x_next_pred, z_t, z_next, z_next_linear = self.model(x_t)
                    losses = self.loss_fn.compute_cae_weaklinear_loss(
                        x_t, x_next, x_t_recon, x_next_pred, z_t, z_next, z_next_linear
                    )
                else:
                    x_t_recon, x_next_pred, z_t, z_next = self.model(x_t)
                    losses = self.loss_fn.compute_cae_linear_loss(
                        x_t, x_next, x_t_recon, x_next_pred, z_t, z_next
                    )
                
                for key, value in losses.items():
                    val_losses[key] += value.item()
        
        # Average losses
        n_batches = len(self.val_loader)
        for key in val_losses:
            val_losses[key] /= n_batches
        
        return val_losses
    
    def _print_epoch_progress(self, epoch: int, total_epochs: int, 
                            train_losses: Dict[str, float], 
                            val_losses: Dict[str, float],
                            optimizer: optim.Optimizer):
        """Print epoch progress"""
        print(f"\nEpoch [{epoch}/{total_epochs}]")
        print(f"Train Loss: {train_losses['total']:.6f} | "
              f"Val Loss: {val_losses['total']:.6f} | "
              f"Best Val Loss: {self.best_val_loss:.6f} (Epoch {self.best_epoch})")
        print(f"Train - Recon: {train_losses['reconstruction']:.6f}, "
              f"Pred: {train_losses['prediction']:.6f}, "
              f"Latent: {train_losses['latent_reg']:.6f}")
        if self.is_weak_linear and 'linear_constraint' in train_losses:
            print(f"Train - Linear Constraint: {train_losses['linear_constraint']:.6f}")
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6f}")
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False, stage: str = ''):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'train_mode': self.train_mode,
            'stage': stage
        }
        
        # Add stage-specific losses
        if self.train_mode == 'separately':
            checkpoint['train_losses_stage1'] = self.train_losses_stage1
            checkpoint['val_losses_stage1'] = self.val_losses_stage1
            if hasattr(self, 'train_losses_stage2'):
                checkpoint['train_losses_stage2'] = self.train_losses_stage2
                checkpoint['val_losses_stage2'] = self.val_losses_stage2
        elif self.train_mode == 'multiply':
            checkpoint['train_losses_stage1'] = self.train_losses_stage1
            checkpoint['val_losses_stage1'] = self.val_losses_stage1
            if hasattr(self, 'train_losses_stage2'):
                checkpoint['train_losses_stage2'] = self.train_losses_stage2
                checkpoint['val_losses_stage2'] = self.val_losses_stage2
            if hasattr(self, 'train_losses_stage3'):
                checkpoint['train_losses_stage3'] = self.train_losses_stage3
                checkpoint['val_losses_stage3'] = self.val_losses_stage3
        else:  # jointly
            checkpoint['train_losses'] = self.train_losses
            checkpoint['val_losses'] = self.val_losses
            checkpoint['best_val_loss'] = self.best_val_loss
            checkpoint['best_epoch'] = self.best_epoch
        
        # Save latest checkpoint
        latest_path = os.path.join(self.save_path, f'latest_checkpoint_{stage}.pth')
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.save_path, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
            if self.verbose:
                print(f"Saved best model with validation loss: {self.best_val_loss:.6f}")
    
    def train(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Train the model according to specified training mode
        
        Returns:
            train_losses: Dictionary of training losses
            val_losses: Dictionary of validation losses
        """
        if self.verbose:
            print(f"Starting training with mode: {self.train_mode}")
            print(f"Model type: {type(self.model).__name__}")
            print(f"Dataset: {self.dataset_name}")
            print(f"Device: {self.device}")
            print(f"Save path: {self.save_path}")
            print("-" * 50)
        
        if self.train_mode == 'separately':
            # Stage 1: Train CAE
            stage1_epochs = self.config.get('training', {}).get('stage1_epochs', self.epochs)
            self._train_cae_reconstruction(stage1_epochs)
            
            # Stage 2: Train predictor
            stage2_epochs = self.config.get('training', {}).get('stage2_epochs', self.epochs)
            self._train_linear_predictor(stage2_epochs)
            
            # Save final checkpoint
            self._save_checkpoint(stage1_epochs + stage2_epochs, stage='final')
            
            # Combine losses for return
            train_losses = {
                'stage1': self.train_losses_stage1,
                'stage2': self.train_losses_stage2
            }
            val_losses = {
                'stage1': self.val_losses_stage1,
                'stage2': self.val_losses_stage2
            }
            
        elif self.train_mode == 'multiply':
            # Stage 1: Train CAE
            stage1_epochs = self.config.get('training', {}).get('stage1_epochs', self.epochs)
            self._train_cae_reconstruction(stage1_epochs)
            
            # Stage 2: Train predictor
            stage2_epochs = self.config.get('training', {}).get('stage2_epochs', self.epochs)
            self._train_linear_predictor(stage2_epochs)
            
            # Stage 3: Joint fine-tuning
            stage3_epochs = self.config.get('training', {}).get('stage3_epochs', self.epochs)
            if self.verbose:
                print("\n" + "="*50)
                print("Stage 3: Joint fine-tuning")
                print("="*50)
            
            # Reset best tracking for stage 3
            self.best_val_loss = float('inf')
            self.best_epoch = 0
            
            # Use the jointly training method for stage 3
            self.train_losses = self.train_losses_stage3
            self.val_losses = self.val_losses_stage3
            self._train_jointly(stage3_epochs)
            
            # Save final checkpoint
            total_epochs = stage1_epochs + stage2_epochs + stage3_epochs
            self._save_checkpoint(total_epochs, stage='final')
            
            # Combine losses for return
            train_losses = {
                'stage1': self.train_losses_stage1,
                'stage2': self.train_losses_stage2,
                'stage3': self.train_losses_stage3
            }
            val_losses = {
                'stage1': self.val_losses_stage1,
                'stage2': self.val_losses_stage2,
                'stage3': self.val_losses_stage3
            }
            
        else:  # jointly
            self._train_jointly(self.epochs)
            train_losses = self.train_losses
            val_losses = self.val_losses
        
        # Save final results
        self._save_training_history(train_losses, val_losses)
        
        if self.verbose:
            print("\n" + "="*50)
            print(f"Training completed!")
            if self.train_mode == 'jointly':
                print(f"Best model at epoch {self.best_epoch} "
                      f"with validation loss {self.best_val_loss:.6f}")
        
        return train_losses, val_losses
    
    def _save_training_history(self, train_losses: Dict, val_losses: Dict):
        """Save training history to JSON"""
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_mode': self.train_mode,
            'config': self.config
        }
        
        if self.train_mode == 'jointly':
            history['best_epoch'] = self.best_epoch
            history['best_val_loss'] = self.best_val_loss
        
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


def train_caemlp_from_config(config_path: str, **kwargs):
    """
    Train CAE-MLP model from configuration file
    
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
    model_type = model_config['type']
    
    # Get input shape and channels from data info
    input_channels = data_info.get('num_channels', 1)
    if 'spatial_resolution' in data_info:
        input_shape = data_info['spatial_resolution']
    else:
        # Infer from batch shape
        batch_shape = data_info['batch_shape']
        if model_type == 'CAE_LinearMLP':
            # For datasets without explicit spatial resolution
            input_shape = config['dataset'].get('target_shape', (64, 64))
        else:
            input_shape = config['dataset'].get('target_shape', (64, 64))
    
    # Create model
    if model_type == 'CAE_LinearMLP':
        model = CAE_LinearMLP(
            input_channels=input_channels,
            input_shape=input_shape,
            **model_config['params']
        )
    elif model_type == 'CAE_WeakLinearMLP':
        model = CAE_WeakLinearMLP(
            input_channels=input_channels,
            input_shape=input_shape,
            **model_config['params']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create trainer
    trainer = CAEMLPTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_mode=config['training']['train_mode'],
        epochs=config['training']['epochs'],
        lr=config['training']['learning_rate'],
        optimizer_config=config['training']['optimizer'],
        scheduler_config=config['training'].get('scheduler', None),
        loss_config=config['training'].get('loss', None),
        device=config['training']['device'],
        save_path=config['training']['save_path'],
        verbose=config['training'].get('verbose', True),
        config=config
    )
    
    # Train model
    train_losses, val_losses = trainer.train()
    
    return trainer, train_losses, val_losses


if __name__ == "__main__":
    # Debug example
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CAE-MLP model')
    parser.add_argument('--config', type=str, default='configs/CAE_MLP_.yaml',
                      help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default='kolmogorov',
                      choices=['kolmogorov', 'cylinder', 'chap'],
                      help='Dataset to use')
    parser.add_argument('--model', type=str, default='CAE_LinearMLP',
                      choices=['CAE_LinearMLP', 'CAE_WeakLinearMLP'],
                      help='Model type')
    parser.add_argument('--train_mode', type=str, default='jointly',
                      choices=['separately', 'jointly', 'multiply'],
                      help='Training mode')
    parser.add_argument('--epochs', type=int, default=10,
                      help='Number of epochs for debugging')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size')
    
    args = parser.parse_args()
    
    # Override config for debugging
    overrides = {
        'dataset.name': args.dataset,
        'dataset.data_path': 'data',
        'dataset.batch_size': args.batch_size,
        'model.type': args.model,
        'training.train_mode': args.train_mode,
        'training.epochs': args.epochs,
        'training.save_path': f'results/checkpoints/CAE_MLP/{args.dataset}_{args.model}_{args.train_mode}_debug'
    }
    
    # For multi-stage training, set stage-specific epochs
    if args.train_mode in ['separately', 'multiply']:
        overrides['training.stage1_epochs'] = max(args.epochs // 3, 1)
        overrides['training.stage2_epochs'] = max(args.epochs // 3, 1)
        if args.train_mode == 'multiply':
            overrides['training.stage3_epochs'] = max(args.epochs // 3, 1)
    
    # Special settings for different datasets
    # if args.dataset == 'cylinder':
    #     overrides['dataset.target_resolution'] = (32, 128)
    if args.dataset == 'chap':
        overrides['dataset.chemical'] = 'Cl'
        overrides['dataset.target_shape'] = (64, 64)  # Simplified for debugging
    
    print(f"Training {args.model} on {args.dataset} dataset")
    print(f"Training mode: {args.train_mode}")
    print(f"Total epochs: {args.epochs}")
    
    try:
        trainer, train_losses, val_losses = train_caemlp_from_config(
            args.config,
            **overrides
        )
        
        # Print final results
        print("\nFinal Results:")
        if args.train_mode == 'jointly':
            print(f"Best validation loss: {trainer.best_val_loss:.6f} at epoch {trainer.best_epoch}")
            print(f"Final train loss: {train_losses['total'][-1]:.6f}")
            print(f"Final val loss: {val_losses['total'][-1]:.6f}")
        else:
            print("Training completed successfully with multi-stage approach")
            if args.train_mode == 'separately':
                print(f"Stage 1 - Final recon loss: {val_losses['stage1']['reconstruction'][-1]:.6f}")
                print(f"Stage 2 - Final pred loss: {val_losses['stage2']['prediction'][-1]:.6f}")
            elif args.train_mode == 'multiply':
                print(f"Stage 1 - Final recon loss: {val_losses['stage1']['reconstruction'][-1]:.6f}")
                print(f"Stage 2 - Final pred loss: {val_losses['stage2']['prediction'][-1]:.6f}")
                print(f"Stage 3 - Final total loss: {val_losses['stage3']['total'][-1]:.6f}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()