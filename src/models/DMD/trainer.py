import os
import sys
import torch
import numpy as np
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directories to path for imports
current_directory = os.getcwd()
upper_directory = os.path.abspath(os.path.join(current_directory, ".."))
upper_upper_directory = os.path.abspath(os.path.join(current_directory, "..", ".."))
sys.path.append(upper_directory)
sys.path.append(upper_upper_directory)

# Import models
from kol_model import KOL_forward_model, KolmogorovConfig

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

class KolmogorovDataset:
    """Simple dataset class for Kolmogorov flow data"""
    def __init__(self, data_path, seq_length=5):
        """
        Args:
            data_path: path to .npy file with shape [N_samples, T, H, W] = [120, 320, 64, 64]
            seq_length: length of sequences for training
        """
        print(f"[INFO] Loading data from {data_path}")
        self.data = np.load(data_path, mmap_mode='r')  # [120, 320, 64, 64]
        print(f"[INFO] Original data shape: {self.data.shape}")
        
        # Unpack 4D shape correctly
        self.n_samples, self.total_time, self.height, self.width = self.data.shape
        self.channels = 1  # Single channel (will be added via unsqueeze)
        
        self.seq_length = seq_length
        
        # Create sequence pairs for DMD training
        self.valid_starts = self.total_time - seq_length
        print(f"[INFO] Samples: {self.n_samples}, Time steps: {self.total_time}, Valid starts: {self.valid_starts}")
        print(f"[INFO] Spatial resolution: {self.height}x{self.width}")
        print(f"[INFO] Final data format will be: [B, T, 1, {self.height}, {self.width}]")
        
    def __len__(self):
        return self.n_samples * self.valid_starts
    
    def __getitem__(self, idx):
        sample_idx = idx // self.valid_starts
        time_start = idx % self.valid_starts
        
        # Load sequences: [T, H, W]
        state_seq = self.data[sample_idx, time_start:time_start+self.seq_length]
        state_next_seq = self.data[sample_idx, time_start+1:time_start+self.seq_length+1]
        
        # Convert to tensor and add channel dimension using unsqueeze
        state_seq = torch.FloatTensor(state_seq).unsqueeze(1)      # [T, H, W] -> [T, 1, H, W]
        state_next_seq = torch.FloatTensor(state_next_seq).unsqueeze(1)  # [T, H, W] -> [T, 1, H, W]
        
        return state_seq, state_next_seq

def train_forward_model(model, dataset, config, device='cpu'):
    """Train the forward model using DMD"""
    print(f"[INFO] Training forward model on {device}")
    
    model.to(device)
    model.train()
    
    # Create data loader with CPU-friendly settings
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=config.num_workers,  # 0 for CPU
        pin_memory=False  # Disable for CPU training
    )
    
    # Optimizer for the encoder/decoder networks
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    
    losses = {'forward': [], 'identity': []}
    
    for epoch in range(config.num_epochs):
        epoch_fwd_loss = 0
        epoch_id_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config.num_epochs}')
        
        for batch_idx, (state_seq, state_next_seq) in enumerate(progress_bar):
            state_seq = state_seq.to(device)      # [B, T, 2, 64, 64]
            state_next_seq = state_next_seq.to(device)  # [B, T, 2, 64, 64]
            
            optimizer.zero_grad()
            
            # Compute DMD loss
            loss_fwd, loss_identity, C_fwd = model.compute_loss(state_seq, state_next_seq)
            
            # Total loss
            total_loss = loss_fwd + config.lambda_identity * loss_identity
            
            total_loss.backward()
            optimizer.step()
            
            epoch_fwd_loss += loss_fwd.item()
            epoch_id_loss += loss_identity.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Fwd Loss': f'{loss_fwd.item():.6f}',
                'ID Loss': f'{loss_identity.item():.6f}'
            })
        
        scheduler.step()
        
        # Average losses
        avg_fwd_loss = epoch_fwd_loss / len(dataloader)
        avg_id_loss = epoch_id_loss / len(dataloader)
        
        losses['forward'].append(avg_fwd_loss)
        losses['identity'].append(avg_id_loss)
        
        print(f'Epoch {epoch+1}: Forward Loss = {avg_fwd_loss:.6f}, Identity Loss = {avg_id_loss:.6f}')
        
        # Save model checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_model(model, config.model_save_path, epoch + 1)
    
    # Final save
    save_model(model, config.model_save_path, 'final')
    
    # Plot losses
    plot_losses(losses, config.model_save_path)
    
    return losses

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

def plot_losses(losses, save_path):
    """Plot and save training losses"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses['forward'], label='Forward Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Forward (Prediction) Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(losses['identity'], label='Identity Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Identity (Reconstruction) Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    loss_plot_path = os.path.join(save_path, 'training_losses.png')
    plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'[INFO] Saved loss plots to {loss_plot_path}')

def test_model(model, dataset, device='cpu', num_test_samples=4):
    """Test the trained model"""
    print(f"[INFO] Testing model with {num_test_samples} samples")
    
    model.eval()
    model.to(device)
    
    # Get some test data
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=num_test_samples, shuffle=True)
    state_seq, state_next_seq = next(iter(test_loader))
    
    state_seq = state_seq.to(device)
    state_next_seq = state_next_seq.to(device)
    
    with torch.no_grad():
        # Test autoencoder reconstruction
        B, T = state_seq.shape[:2]
        state_flat = state_seq.view(B*T, *state_seq.shape[2:])
        
        # Encode and decode
        z = model.phi_S(state_flat)
        reconstructed = model.phi_inv_S(z)
        
        recon_error = torch.mean((state_flat - reconstructed) ** 2)
        print(f"[INFO] Reconstruction error: {recon_error.item():.6f}")
        
        # Test forward prediction
        loss_fwd, loss_identity, C_fwd = model.compute_loss(state_seq, state_next_seq)
        print(f"[INFO] Forward loss: {loss_fwd.item():.6f}")
        print(f"[INFO] DMD matrix shape: {C_fwd.shape}")
        print(f"[INFO] DMD matrix condition number: {torch.linalg.cond(C_fwd).item():.2f}")

def main():
    """Main training function"""
    # Load configuration
    config_path = 'config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config = dict2namespace(config)
    
    # Device setup - Force CPU usage
    if config.use_cpu:
        device = "cpu"
        print(f"[INFO] Forced to use CPU (GPU disabled in config)")
    else:
        device = f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Using device: {device}")
        if torch.cuda.is_available():
            print(f"[INFO] GPU: {torch.cuda.get_device_name(config.gpu_id)}")
    
    # CPU-specific optimizations
    if device == "cpu":
        print(f"[INFO] Applying CPU optimizations...")
        torch.set_num_threads(4)  # Limit CPU threads
        print(f"[INFO] CPU threads set to: {torch.get_num_threads()}")
    
    # Create dataset
    dataset = KolmogorovDataset(
        data_path=config.data_path,
        seq_length=config.seq_length
    )
    
    # Create model with updated config
    model_config = KolmogorovConfig()
    model_config.input_channels = config.input_channels  # 1 instead of 2
    model_config.latent_dim = config.latent_dim
    model_config.seq_length = config.seq_length
    
    model = KOL_forward_model(model_config)
    
    # Test model architecture
    print("\n[INFO] Testing model architecture...")
    sample_data = torch.randn(1, 1, 64, 64)  # [B, 1, H, W] for single channel
    model.verify_model(sample_data)
    
    # Train model
    print(f"\n[INFO] Starting training...")
    losses = train_forward_model(model, dataset, config, device)
    
    # Test trained model
    print(f"\n[INFO] Testing trained model...")
    test_model(model, dataset, device)
    
    print(f"\n[INFO] Training completed! Models saved to {config.model_save_path}")

if __name__ == "__main__":
    main()