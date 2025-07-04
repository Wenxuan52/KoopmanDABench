import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import yaml
from typing import Optional

# Add paths for imports
current_directory = os.getcwd()
upper_directory = os.path.abspath(os.path.join(current_directory, ".."))
upper_upper_directory = os.path.abspath(os.path.join(current_directory, "..", ".."))
sys.path.append(upper_directory)
sys.path.append(upper_upper_directory)

from cylinder_model import CYLINDER_C_FORWARD, CYLINDER_C_INVERSE
from utils import dict2namespace, count_parameters


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
        print(f"[INFO] Random indices: {indices}")
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


class CylinderDynamicsDataset(Dataset):
    """Memory-efficient Dataset for Cylinder flow dynamics data"""
    
    def __init__(self, data_path: str, seq_length: int = 10, normalize: bool = True, 
                 train_ratio: float = 0.8, random_seed: int = 42, subsample_ratio: float = 0.1):
        self.data_path = data_path
        self.seq_length = seq_length
        self.normalize = normalize
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        self.subsample_ratio = subsample_ratio
        
        # Load and process data
        self._load_data()
        self._create_indices()
        
        if self.normalize:
            self._compute_normalization_stats()
    
    def _load_data(self):
        """Load raw data without creating all sequences in memory"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Cylinder dataset not found at {self.data_path}")
        
        # Load data: [49, 1000, 2, 64, 64]
        self.raw_data = np.load(self.data_path)
        print(f"[INFO] Loaded Cylinder data with shape: {self.raw_data.shape}")
        
        if len(self.raw_data.shape) != 5:
            raise ValueError(f"Expected 5D data [samples, time, channels, height, width], got {self.raw_data.shape}")
        
        self.n_samples, self.n_time, self.n_channels, self.height, self.width = self.raw_data.shape
    
    def _create_indices(self):
        """Create indices for valid sequences with subsampling"""
        valid_indices = []
        
        for sample_idx in range(self.n_samples):
            for t in range(0, self.n_time - self.seq_length, max(1, int(1/self.subsample_ratio))):
                valid_indices.append((sample_idx, t))
        
        # Random split
        np.random.seed(self.random_seed)
        np.random.shuffle(valid_indices)
        
        train_size = int(len(valid_indices) * self.train_ratio)
        self.train_indices = valid_indices[:train_size]
        self.val_indices = valid_indices[train_size:]
        
        # Use only training indices for now
        self.indices = self.train_indices
        
        print(f"[INFO] Created {len(self.train_indices)} training sequences")
        print(f"[INFO] Created {len(self.val_indices)} validation sequences")
        print(f"[INFO] Subsampling ratio: {self.subsample_ratio}")
    
    def _compute_normalization_stats(self):
        """Compute normalization statistics from a subset of training data"""
        if not self.normalize:
            return
            
        # Sample a subset of training data for computing stats
        sample_size = min(1000, len(self.train_indices))
        sample_indices = np.random.choice(len(self.train_indices), sample_size, replace=False)
        
        sample_data = []
        for idx in sample_indices:
            sample_idx, t = self.train_indices[idx]
            seq_current = self.raw_data[sample_idx, t:t+self.seq_length]
            sample_data.append(seq_current)
        
        sample_data = np.array(sample_data)  # [sample_size, seq_length, channels, height, width]
        
        # Compute mean and std across all dimensions except channels
        # Result shape: [channels]
        self.mean = np.mean(sample_data, axis=(0, 1, 3, 4))  # [channels]
        self.std = np.std(sample_data, axis=(0, 1, 3, 4))    # [channels]
        
        # Reshape for broadcasting: [channels] -> [channels, 1, 1]
        self.mean = self.mean.reshape(-1, 1, 1)
        self.std = self.std.reshape(-1, 1, 1)
        
        # Avoid division by zero
        self.std[self.std < 1e-8] = 1.0
        
        print(f"[INFO] Normalization computed from {sample_size} samples")
        print(f"[INFO] Mean shape: {self.mean.shape}, Std shape: {self.std.shape}")
        print(f"[INFO] Mean: {self.mean.flatten()}, Std: {self.std.flatten()}")
    
    def set_split(self, split='train'):
        """Set which split to use: 'train' or 'val'"""
        if split == 'train':
            self.indices = self.train_indices
        elif split == 'val':
            self.indices = self.val_indices
        else:
            raise ValueError("Split must be 'train' or 'val'")
        print(f"[INFO] Dataset split set to: {split} ({len(self.indices)} samples)")

    def __len__(self):
        """Return the number of samples in the current split"""
        return len(self.indices)
    
    def __getitem__(self, idx):
        sample_idx, t = self.indices[idx]
        
        # Extract sequences on-the-fly
        seq_current = self.raw_data[sample_idx, t:t+self.seq_length].copy()  # [seq_length, channels, height, width]
        seq_next = self.raw_data[sample_idx, t+1:t+self.seq_length+1].copy()  # [seq_length, channels, height, width]
        
        # Apply normalization if needed
        if self.normalize and hasattr(self, 'mean'):
            seq_current = (seq_current - self.mean) / self.std
            seq_next = (seq_next - self.mean) / self.std
        
        return torch.tensor(seq_current, dtype=torch.float32), torch.tensor(seq_next, dtype=torch.float32)
    
    def set_split(self, split='train'):
        """Set which split to use: 'train' or 'val'"""
        if split == 'train':
            self.indices = self.train_indices
        elif split == 'val':
            self.indices = self.val_indices
        else:
            raise ValueError("Split must be 'train' or 'val'")
        print(f"[INFO] DA Dataset split set to: {split} ({len(self.indices)} samples)")


class CylinderDADataset(Dataset):
    """Memory-efficient Dataset for Cylinder flow data assimilation"""
    
    def __init__(self, data_path: str, history_len: int = 10, obs_noise_std: float = 0.01,
                 normalize: bool = True, train_ratio: float = 0.8, random_seed: int = 42, 
                 subsample_ratio: float = 0.1):
        self.data_path = data_path
        self.history_len = history_len
        self.obs_noise_std = obs_noise_std
        self.normalize = normalize
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        self.subsample_ratio = subsample_ratio
        
        # Load and process data
        self._load_data()
        self._create_indices()
        
        if self.normalize:
            self._compute_normalization_stats()
    
    def _load_data(self):
        """Load raw data without creating all pairs in memory"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Cylinder dataset not found at {self.data_path}")
        
        # Load data: [49, 1000, 2, 64, 64]
        self.raw_data = np.load(self.data_path)
        print(f"[INFO] Loaded Cylinder data for DA with shape: {self.raw_data.shape}")
        
        if len(self.raw_data.shape) != 5:
            raise ValueError(f"Expected 5D data [samples, time, channels, height, width], got {self.raw_data.shape}")
        
        self.n_samples, self.n_time, self.n_channels, self.height, self.width = self.raw_data.shape
    
    def _create_indices(self):
        """Create indices for valid DA pairs with subsampling"""
        valid_indices = []
        
        for sample_idx in range(self.n_samples):
            for t in range(self.history_len, self.n_time, max(1, int(1/self.subsample_ratio))):
                valid_indices.append((sample_idx, t))
        
        # Random split
        np.random.seed(self.random_seed)
        np.random.shuffle(valid_indices)
        
        train_size = int(len(valid_indices) * self.train_ratio)
        self.train_indices = valid_indices[:train_size]
        self.val_indices = valid_indices[train_size:]
        
        # Use only training indices for now
        self.indices = self.train_indices
        
        print(f"[INFO] Created {len(self.train_indices)} training DA pairs")
        print(f"[INFO] Created {len(self.val_indices)} validation DA pairs")
        print(f"[INFO] Subsampling ratio: {self.subsample_ratio}")
    
    def _compute_normalization_stats(self):
        """Compute normalization statistics from a subset of training data"""
        if not self.normalize:
            return
            
        # Sample a subset of training data for computing stats
        sample_size = min(500, len(self.train_indices))
        sample_indices = np.random.choice(len(self.train_indices), sample_size, replace=False)
        
        sample_data = []
        for idx in sample_indices:
            sample_idx, t = self.train_indices[idx]
            hist_obs = self.raw_data[sample_idx, t-self.history_len:t]
            sample_data.append(hist_obs)
        
        sample_data = np.array(sample_data)  # [sample_size, history_len, channels, height, width]
        
        # Compute mean and std
        self.mean = np.mean(sample_data, axis=(0, 1, 3, 4), keepdims=True)  # [1, 1, channels, 1, 1]
        self.std = np.std(sample_data, axis=(0, 1, 3, 4), keepdims=True)
        
        # Avoid division by zero
        self.std[self.std < 1e-8] = 1.0
        
        print(f"[INFO] DA normalization computed from {sample_size} samples")
        print(f"[INFO] Mean: {self.mean.flatten()}, Std: {self.std.flatten()}")
    
    def __len__(self):
        """Return the number of samples in the current split"""
        return len(self.indices)
    
    def __getitem__(self, idx):
        sample_idx, t = self.indices[idx]
        
        # Extract data on-the-fly
        hist_obs = self.raw_data[sample_idx, t-self.history_len:t].copy()  # [history_len, channels, height, width]
        true_state = self.raw_data[sample_idx, t].copy()  # [channels, height, width]
        
        # Add observation noise
        noise = np.random.normal(0, self.obs_noise_std, hist_obs.shape)
        hist_obs_noisy = hist_obs + noise
        
        # Apply normalization if needed
        if self.normalize and hasattr(self, 'mean'):
            hist_obs = (hist_obs - self.mean) / self.std
            hist_obs_noisy = (hist_obs_noisy - self.mean) / self.std
            true_state = (true_state - self.mean[0, 0]) / self.std[0, 0]  # Remove time dimension
        
        return (torch.tensor(hist_obs_noisy, dtype=torch.float32), 
                torch.tensor(true_state, dtype=torch.float32),
                torch.tensor(hist_obs, dtype=torch.float32))
    
    def __len__(self):
        return len(self.indices)


def evaluate_forward_model(forward_model, val_dataset, device='cpu', weight_matrix=None):
    """Evaluate forward model on validation set"""
    forward_model.eval()
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    total_loss = 0.0
    total_fwd_loss = 0.0
    total_id_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in val_dataloader:
            pre_sequences, post_sequences = [data.to(device) for data in batch_data]
            
            if weight_matrix is not None:
                B = batch_data[0].shape[0]
                C = batch_data[0].shape[2]
                weight_matrix_batch = torch.tensor(weight_matrix, dtype=torch.float32).repeat(B, C, 1, 1).to(device)
                loss_fwd, loss_id, _ = forward_model.compute_loss(pre_sequences, post_sequences, weight_matrix_batch)
            else:
                loss_fwd, loss_id, _ = forward_model.compute_loss(pre_sequences, post_sequences)
            
            total_fwd_loss += loss_fwd.item()
            total_id_loss += loss_id.item()
            total_loss += (loss_fwd + 0.3 * loss_id).item()  # Using same lamb=0.3
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_fwd_loss = total_fwd_loss / num_batches
    avg_id_loss = total_id_loss / num_batches
    
    forward_model.train()
    return avg_loss, avg_fwd_loss, avg_id_loss


def train_forward_model(forward_model, 
                       train_dataset,
                       model_save_folder: str,
                       learning_rate: float = 1e-3,
                       lamb: float = 0.3,
                       weight_decay: float = 0,
                       batch_size: int = 64,
                       num_epochs: int = 20,
                       gradclip: float = 1,
                       device: str = 'cpu',
                       weight_matrix=None):
    
    print(f"[INFO] Training forward model for {num_epochs} epochs")
    if not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)
    
    # Set to training split first
    train_dataset.set_split('train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    num_samples = len(train_dataset.train_indices)
    
    train_loss = {'forward': [[] for _ in range(num_epochs)], 
                  'id': [[] for _ in range(num_epochs)],
                  'total': [[] for _ in range(num_epochs)]}
    
    val_loss = {'forward': [], 'id': [], 'total': []}
    
    trainable_parameters = filter(lambda p: p.requires_grad, forward_model.parameters())
    count_parameters(forward_model)
    
    optimizer = Adam(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)
    forward_model.train()
    forward_model.to(device)
    
    best_val_loss = np.inf
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase - ensure we're using training split
        train_dataset.set_split('train')
        
        all_loss = []
        C_fwd = torch.zeros((forward_model.hidden_dim, forward_model.hidden_dim), dtype=torch.float32).to(device)
        
        for batch_idx, batch_data in tqdm(enumerate(train_dataloader), 
                                desc=f'Training Epochs with lr={optimizer.param_groups[0]["lr"]:.6f}:', 
                                total=len(train_dataloader)):
            
            B = batch_data[0].shape[0]
            pre_sequences, post_sequences = [data.to(device) for data in batch_data]
            
            # Debug: print tensor shapes for first batch
            if epoch == 0 and batch_idx == 0:
                print(f"[DEBUG] pre_sequences shape: {pre_sequences.shape}")
                print(f"[DEBUG] post_sequences shape: {post_sequences.shape}")
                print(f"[DEBUG] Expected format: [batch_size, seq_length, channels, height, width]")
            
            if weight_matrix is not None:
                C = batch_data[0].shape[2]
                weight_matrix_batch = torch.tensor(weight_matrix, dtype=torch.float32).repeat(B, C, 1, 1).to(device)
                loss_fwd, loss_id, tmp_C_fwd = forward_model.compute_loss(pre_sequences, post_sequences, weight_matrix_batch)
            else:
                loss_fwd, loss_id, tmp_C_fwd = forward_model.compute_loss(pre_sequences, post_sequences)
            
            loss = loss_fwd + lamb * loss_id
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(forward_model.parameters(), gradclip)
            optimizer.step()
            
            C_fwd += tmp_C_fwd.detach() * (B / num_samples)
            train_loss['forward'][epoch].append(loss_fwd.item())
            train_loss['id'][epoch].append(loss_id.item())
            train_loss['total'][epoch].append(loss.item())
            all_loss.append(loss.item())
        
        # Calculate average training losses
        train_loss['forward'][epoch] = np.mean(train_loss['forward'][epoch])
        train_loss['id'][epoch] = np.mean(train_loss['id'][epoch])
        train_loss['total'][epoch] = np.mean(train_loss['total'][epoch])
        
        # Validation phase
        train_dataset.set_split('val')
        val_loss_epoch, val_fwd_loss_epoch, val_id_loss_epoch = evaluate_forward_model(
            forward_model, train_dataset, device, weight_matrix)
        
        val_loss['total'].append(val_loss_epoch)
        val_loss['forward'].append(val_fwd_loss_epoch)
        val_loss['id'].append(val_id_loss_epoch)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train - Total: {train_loss["total"][epoch]:.4f}, Forward: {train_loss["forward"][epoch]:.4f}, ID: {train_loss["id"][epoch]:.4f}')
        print(f'Val   - Total: {val_loss_epoch:.4f}, Forward: {val_fwd_loss_epoch:.4f}, ID: {val_id_loss_epoch:.4f}')
        
        # Save model if validation loss improved
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            patience_counter = 0
            print(f"[INFO] New best validation loss: {val_loss_epoch:.4f} - Saving model")
            forward_model.save_model(model_save_folder)
            try:
                forward_model.save_C_fwd(model_save_folder, C_fwd)
            except:
                forward_model.save_C_forward(model_save_folder, C_fwd)
            forward_model.to(device)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[INFO] Early stopping triggered after {patience} epochs without improvement")
                break
        
        print('')
    
    print(f"[INFO] Training completed. Best validation loss: {best_val_loss:.4f}")
    return train_loss, val_loss


def main():
    # Device setup
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using {device} device")
    if torch.cuda.is_available():
        print(f"[INFO] {torch.cuda.get_device_properties(0)}")
    
    # Paths
    data_path = "../../../data/cylinder/cylinder_data.npy"
    model_save_folder = "model_weights"
    
    # Training configuration
    config = {
        'seq_length': 12,
        'history_len': 10,
        'num_epochs': 60,
        'decay_step': 15,
        'batch_size': 128,
        'learning_rate': 1e-3,
        'decay_rate': 0.8,
        'lamb': 1.0,
        'nu': 0.1,
        'subsample_ratio': 1.0, # Use only 5% of data for faster training
    }
    
    print("[INFO] Starting Cylinder Flow Model Training")
    print(f"[INFO] Configuration: {config}")
    
    # ========================================
    # Train Forward Model
    # ========================================
    print("\n" + "="*50)
    print("TRAINING FORWARD MODEL")
    print("="*50)
    
    # Load dynamics dataset
    dynamics_dataset = CylinderDynamicsDataset(data_path=data_path, 
                                              seq_length=config['seq_length'],
                                              normalize=True,
                                              subsample_ratio=config['subsample_ratio'])
    
    # Create forward model
    forward_model = CYLINDER_C_FORWARD()
    
    # Train forward model with learning rate decay
    for i in range(config['num_epochs'] // config['decay_step']):
        batch_size = config['batch_size']
        num_epochs = config['decay_step']
        learning_rate = config['learning_rate'] * (config['decay_rate'] ** i)
        
        print(f"\n[INFO] Training forward model stage {i+1}")
        print(f"[INFO] Epochs: {num_epochs}, Batch size: {batch_size}, Learning rate: {learning_rate}")
        
        loss_info = train_forward_model(forward_model, 
                                       dynamics_dataset,
                                       batch_size=batch_size,
                                       num_epochs=num_epochs,
                                       learning_rate=learning_rate,
                                       model_save_folder=model_save_folder,
                                       device=device,
                                       lamb=config['lamb'])
    
    # Load best forward model
    forward_model.load_state_dict(torch.load(os.path.join(model_save_folder, 'forward_model.pt')))
    try:
        forward_model.C_forward = torch.load(os.path.join(model_save_folder, 'C_forward.pt'))
    except:
        forward_model.C_forward = torch.load(os.path.join(model_save_folder, 'C_fwd.pt'))
    
    forward_model.to(device)
    
    # Compute statistics for the forward model
    single_step_dataset = CylinderDynamicsDataset(data_path=data_path, 
                                                 seq_length=1,
                                                 normalize=True,
                                                 subsample_ratio=0.01)  # Even smaller subset for stats
    forward_model.compute_z_b(single_step_dataset, device=device, save_path=model_save_folder)
    del dynamics_dataset, single_step_dataset

if __name__ == "__main__":
    main()