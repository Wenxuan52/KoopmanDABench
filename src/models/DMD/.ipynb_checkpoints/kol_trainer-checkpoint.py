import os
import sys
import torch
import numpy as np
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import random
from torch.utils.data import Dataset, DataLoader

# Add parent directories to path for imports
current_directory = os.getcwd()
upper_directory = os.path.abspath(os.path.join(current_directory, ".."))
upper_upper_directory = os.path.abspath(os.path.join(current_directory, "..", ".."))
sys.path.append(upper_directory)
sys.path.append(upper_upper_directory)

# Import models
from kol_model import KOL_forward_model, KolmogorovConfig

def set_seed(seed: int = 42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Set random seed to {seed}")

class SimpleNamespace:
    """Simple utility to convert dict to namespace"""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def dict2namespace(config):
    """Convert nested dictionary to namespace"""
    namespace = SimpleNamespace()
    for key, value in config.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict2namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace

class BaseDataset(Dataset):
    """Base dataset class for DMD models"""
    
    def __init__(self, data_path: str, normalize: bool = True, 
                 train_ratio: float = 0.8, random_seed: int = 42):
        self.data_path = data_path
        self.normalize = normalize
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        
        # To be filled by child classes
        self.data = None
        self.train_data = None
        self.val_data = None
        self.mean = None
        self.std = None
        
        # Load data
        self._load_data()
        
        # Split train/val
        self._split_data()
        
        # Normalize if needed
        if self.normalize:
            self._normalize_data()
    
    def _load_data(self):
        """To be implemented by child classes"""
        raise NotImplementedError
    
    def _split_data(self):
        """Split data into train/val sets randomly"""
        n_samples = len(self.data)
        train_size = int(n_samples * self.train_ratio)
        
        # Set random seed for reproducible splits
        np.random.seed(self.random_seed)
        
        # Create random indices
        indices = np.random.permutation(n_samples)
        print(indices)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Split data randomly
        self.train_data = self.data[train_indices]
        self.val_data = self.data[val_indices]
        
        print(f"[INFO] Data split: {len(self.train_data)} train, {len(self.val_data)} validation (random split)")
    
    def _normalize_data(self):
        """Normalize data using mean and std from training set"""
        # Calculate mean and std from training data
        if self.data.ndim == 4:  # [samples, time, height, width]
            self.mean = np.mean(self.train_data, axis=(0, 1, 2, 3), keepdims=True)
            self.std = np.std(self.train_data, axis=(0, 1, 2, 3), keepdims=True)
        elif self.data.ndim == 5:  # [samples, time, channels, height, width]
            self.mean = np.mean(self.train_data, axis=(0, 1, 3, 4), keepdims=True)
            self.std = np.std(self.train_data, axis=(0, 1, 3, 4), keepdims=True)
        
        # Avoid division by zero
        self.std[self.std < 1e-8] = 1.0
        
        # Apply normalization
        if self.mean is not None and self.std is not None:
            self.train_data = (self.train_data - self.mean) / self.std
            self.val_data = (self.val_data - self.mean) / self.std
            print(f"[INFO] Data normalized - Mean: {self.mean.flatten()[:3]}..., Std: {self.std.flatten()[:3]}...")
    
    def set_normalization_params(self, mean: np.ndarray, std: np.ndarray):
        """Set normalization parameters (for validation set)"""
        self.mean = mean
        self.std = std
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample

class DatasetKol(BaseDataset):
    """Dataset for Kolmogorov flow data"""
    
    def _load_data(self):
        """Load Kolmogorov dataset"""
        file_path = os.path.join(self.data_path, "RE1000", "kf_2d_re1000_64_120seed.npy")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Kolmogorov dataset not found at {file_path}")
        
        # Load data: [120, 320, 64, 64]
        data = np.load(file_path)
        print(f"[INFO] Loaded Kolmogorov data with shape: {data.shape}")
        
        self.data = data

class KolmogorovSequenceDataset(Dataset):
    """Dataset that creates sequences for DMD training from BaseDataset"""
    
    def __init__(self, base_dataset, seq_length: int = 10, is_training: bool = True):
        self.base_dataset = base_dataset
        self.seq_length = seq_length
        self.is_training = is_training
        
        # Use training or validation data
        if is_training:
            self.data = base_dataset.train_data
            self.split_name = "train"
        else:
            self.data = base_dataset.val_data
            self.split_name = "val"
        
        # Calculate valid sequence starts
        self.n_samples, self.total_time, self.height, self.width = self.data.shape
        self.valid_starts = self.total_time - seq_length
        
        print(f"[INFO] {self.split_name} sequences: {self.n_samples} samples × {self.valid_starts} time starts = {len(self)} total")
    
    def __len__(self):
        return self.n_samples * self.valid_starts
    
    def __getitem__(self, idx):
        sample_idx = idx // self.valid_starts
        time_start = idx % self.valid_starts
        
        # Load sequences: [T, H, W]
        state_seq = self.data[sample_idx, time_start:time_start+self.seq_length]
        state_next_seq = self.data[sample_idx, time_start+1:time_start+self.seq_length+1]
        
        # Convert to tensor and add channel dimension
        state_seq = torch.FloatTensor(state_seq).unsqueeze(1)      # [T, H, W] -> [T, 1, H, W]
        state_next_seq = torch.FloatTensor(state_next_seq).unsqueeze(1)  # [T, H, W] -> [T, 1, H, W]
        
        return state_seq, state_next_seq

def create_optimized_dataloader(dataset, config, is_training=True):
    """Create optimized dataloader for training or validation"""
    dataloader_kwargs = {
        'batch_size': config.batch_size,
        'shuffle': is_training,  # Only shuffle training data
        'num_workers': config.num_workers,
        'pin_memory': getattr(config, 'pin_memory', True),
        'drop_last': is_training,  # Only drop last for training
    }
    
    # Add advanced optimizations if available
    if hasattr(config, 'prefetch_factor') and config.num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = config.prefetch_factor
    
    if hasattr(config, 'persistent_workers') and config.num_workers > 0:
        dataloader_kwargs['persistent_workers'] = config.persistent_workers
    
    return DataLoader(dataset, **dataloader_kwargs)

def validate_model(model, val_loader, config, device, epoch):
    """Validate the model on validation set"""
    model.eval()
    val_fwd_loss = 0.0
    val_id_loss = 0.0
    
    with torch.no_grad():
        val_progress = tqdm(val_loader, desc=f'Validation {epoch}', leave=False)
        for state_seq, state_next_seq in val_progress:
            state_seq = state_seq.to(device, non_blocking=True)
            state_next_seq = state_next_seq.to(device, non_blocking=True)
            
            # Compute validation loss
            loss_fwd, loss_identity, _ = model.compute_loss(state_seq, state_next_seq)
            
            val_fwd_loss += loss_fwd.item()
            val_id_loss += loss_identity.item()
            
            val_progress.set_postfix({
                'Val_Fwd': f'{loss_fwd.item():.4f}',
                'Val_ID': f'{loss_identity.item():.4f}'
            })
    
    avg_val_fwd = val_fwd_loss / len(val_loader)
    avg_val_id = val_id_loss / len(val_loader)
    
    model.train()  # Return to training mode
    
    return avg_val_fwd, avg_val_id

def train_forward_model(model, train_loader, val_loader, config, device='cuda'):
    """Enhanced training function with validation"""
    print(f"[INFO] Training forward model on {device}")
    
    model.to(device)
    model.train()
    
    # Optimizer with better settings
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=getattr(config, 'decay_step', 20), 
        gamma=getattr(config, 'decay_rate', 0.8)
    )
    
    # Track losses
    history = {
        'train_forward': [], 'train_identity': [],
        'val_forward': [], 'val_identity': [],
        'epochs': [], 'times': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = getattr(config, 'patience', 10)
    
    print("[INFO] Starting training with validation...")
    
    for epoch in range(config.num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_fwd_loss = 0.0
        train_id_loss = 0.0
        
        train_progress = tqdm(train_loader, desc=f'Train Epoch {epoch+1}/{config.num_epochs}')
        
        for batch_idx, (state_seq, state_next_seq) in enumerate(train_progress):
            state_seq = state_seq.to(device, non_blocking=True)
            state_next_seq = state_next_seq.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Compute DMD loss
            loss_fwd, loss_identity, C_fwd = model.compute_loss(state_seq, state_next_seq)
            
            # Total loss
            total_loss = loss_fwd + config.lambda_identity * loss_identity
            
            total_loss.backward()
            
            # Gradient clipping if specified
            if hasattr(config, 'gradient_clip'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            
            optimizer.step()
            
            train_fwd_loss += loss_fwd.item()
            train_id_loss += loss_identity.item()
            
            # Update progress bar
            train_progress.set_postfix({
                'Fwd': f'{loss_fwd.item():.4f}',
                'ID': f'{loss_identity.item():.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Calculate training averages
        avg_train_fwd = train_fwd_loss / len(train_loader)
        avg_train_id = train_id_loss / len(train_loader)
        
        # Validation phase
        avg_val_fwd, avg_val_id = validate_model(model, val_loader, config, device, epoch+1)
        
        # Learning rate scheduling
        scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        
        # Record history
        history['train_forward'].append(avg_train_fwd)
        history['train_identity'].append(avg_train_id)
        history['val_forward'].append(avg_val_fwd)
        history['val_identity'].append(avg_val_id)
        history['epochs'].append(epoch + 1)
        history['times'].append(epoch_time)
        
        # Print epoch summary
        print(f'Epoch {epoch+1:3d}: '
              f'Train[Fwd={avg_train_fwd:.6f}, ID={avg_train_id:.6f}] '
              f'Val[Fwd={avg_val_fwd:.6f}, ID={avg_val_id:.6f}] '
              f'Time={epoch_time:.1f}s')
        
        # Early stopping check
        current_val_loss = avg_val_fwd + avg_val_id
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            patience_counter = 0
            # Save best model
            save_model(model, config.model_save_path, 'best')
            print(f'[INFO] New best model saved (val_loss={current_val_loss:.6f})')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'[INFO] Early stopping triggered after {patience} epochs without improvement')
                break
        
        # Save checkpoint
        if hasattr(config, 'save_every_n_epochs') and (epoch + 1) % config.save_every_n_epochs == 0:
            save_model(model, config.model_save_path, epoch + 1)
    
    # Final save
    save_model(model, config.model_save_path, 'final')
    
    # Plot training history
    plot_training_history(history, config.model_save_path)
    
    print(f"[INFO] Training completed! Best validation loss: {best_val_loss:.6f}")
    print(f"[INFO] Average epoch time: {np.mean(history['times']):.1f}s")
    
    return history

def save_model(model, save_path, epoch):
    """Save model weights and DMD matrix"""
    os.makedirs(save_path, exist_ok=True)
    
    # Save full model
    model_path = os.path.join(save_path, f'kol_forward_model_{epoch}.pt')
    torch.save(model.state_dict(), model_path)
    
    # Save DMD matrix separately
    if model.C_fwd is not None:
        dmd_path = os.path.join(save_path, f'C_fwd_{epoch}.pt')
        torch.save(model.C_fwd, dmd_path)
    
    print(f'[INFO] Saved model to {model_path}')

def plot_training_history(history, save_path):
    """Plot comprehensive training history"""
    plt.figure(figsize=(20, 6))
    
    # Forward loss
    plt.subplot(1, 4, 1)
    plt.plot(history['epochs'], history['train_forward'], 'b-', label='Train Forward', linewidth=2)
    plt.plot(history['epochs'], history['val_forward'], 'r-', label='Val Forward', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Forward Loss')
    plt.title('Forward (Prediction) Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Identity loss
    plt.subplot(1, 4, 2)
    plt.plot(history['epochs'], history['train_identity'], 'b-', label='Train Identity', linewidth=2)
    plt.plot(history['epochs'], history['val_identity'], 'r-', label='Val Identity', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Identity Loss')
    plt.title('Identity (Reconstruction) Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Combined loss
    plt.subplot(1, 4, 3)
    train_total = np.array(history['train_forward']) + np.array(history['train_identity'])
    val_total = np.array(history['val_forward']) + np.array(history['val_identity'])
    plt.plot(history['epochs'], train_total, 'b-', label='Train Total', linewidth=2)
    plt.plot(history['epochs'], val_total, 'r-', label='Val Total', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Combined Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Training time
    plt.subplot(1, 4, 4)
    plt.plot(history['epochs'], history['times'], 'g-', linewidth=2)
    plt.axhline(y=np.mean(history['times']), color='red', linestyle='--', 
                label=f'Avg: {np.mean(history["times"]):.1f}s')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('Epoch Training Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_path, 'training_history.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'[INFO] Saved training history to {plot_path}')

def main():
    """Main training function"""
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
        print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(config.gpu_id).total_memory / 1024**3:.1f}GB")
    else:
        device = "cpu"
        print(f"[INFO] Using CPU")
    
    # Create base dataset with train/val split
    base_dataset = DatasetKol(
        data_path=config.data_path,
        normalize=getattr(config, 'normalize', True),
        train_ratio=getattr(config, 'train_ratio', 0.8),
        random_seed=getattr(config, 'random_seed', 42)
    )
    
    # Create sequence datasets
    train_dataset = KolmogorovSequenceDataset(
        base_dataset, 
        seq_length=config.seq_length, 
        is_training=True
    )
    
    val_dataset = KolmogorovSequenceDataset(
        base_dataset, 
        seq_length=config.seq_length, 
        is_training=False
    )
    
    # Create data loaders
    train_loader = create_optimized_dataloader(train_dataset, config, is_training=True)
    val_loader = create_optimized_dataloader(val_dataset, config, is_training=False)
    
    # Create model
    model_config = KolmogorovConfig()
    model_config.input_channels = config.input_channels
    model_config.latent_dim = config.latent_dim
    model_config.seq_length = config.seq_length
    
    model = KOL_forward_model(model_config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total parameters: {total_params:,}")
    print(f"[INFO] Trainable parameters: {trainable_params:,}")
    
    # Test model architecture
    print("\n[INFO] Testing model architecture...")
    sample_data = torch.randn(2, 1, 64, 64)
    model.verify_model(sample_data)
    
    # Train model with validation
    print(f"\n[INFO] Starting training with validation...")
    history = train_forward_model(model, train_loader, val_loader, config, device)
    
    print(f"\n[INFO] Training completed! Models saved to {config.model_save_path}")

if __name__ == "__main__":
    main()