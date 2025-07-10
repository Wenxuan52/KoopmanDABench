import os
import sys
import numpy as np
import torch
import random
import json
import pickle
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

from utils import dict2namespace, count_parameters
from Dataset import KolDynamicsDataset

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


def evaluate_forward_model(forward_model, val_dataset, device='cpu', weight_matrix=None, batch_size=64, lamb=0.3):
    """Evaluate forward model on validation set"""
    forward_model.eval()
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
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
            total_loss += (loss_fwd + lamb * loss_id).item()  # Using same lamb=0.3
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_fwd_loss = total_fwd_loss / num_batches
    avg_id_loss = total_id_loss / num_batches
    
    forward_model.train()
    return avg_loss, avg_fwd_loss, avg_id_loss


def train_forward_model(forward_model, 
                       train_dataset,
                       val_dataset,
                       model_save_folder: str,
                       learning_rate: float = 1e-3,
                       lamb: float = 0.3,
                       weight_decay: float = 0,
                       batch_size: int = 64,
                       num_epochs: int = 20,
                       gradclip: float = 1,
                       device: str = 'cpu',
                       best_val_loss=None,
                       weight_matrix=None):
    
    print(f"[INFO] Training forward model for {num_epochs} epochs")
    if not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)
    
    # Set to training split first
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    num_samples = train_dataset.total_sample
    
    train_loss = {'forward': [[] for _ in range(num_epochs)], 
                  'id': [[] for _ in range(num_epochs)],
                  'total': [[] for _ in range(num_epochs)]}
    
    val_loss = {'forward': [], 'id': [], 'total': []}
    
    trainable_parameters = filter(lambda p: p.requires_grad, forward_model.parameters())
    count_parameters(forward_model)
    
    optimizer = Adam(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)
    forward_model.train()
    forward_model.to(device)
    
    if best_val_loss == None:
        best_val_loss = np.inf
    else:
        best_val_loss = best_val_loss
    patience = 20
    patience_counter = 0
    
    for epoch in range(num_epochs):
        
        all_loss = []
        C_fwd = torch.zeros((forward_model.hidden_dim, forward_model.hidden_dim), dtype=torch.float32).to(device)
        
        for batch_idx, batch_data in tqdm(enumerate(train_dataloader), 
                                desc=f'Training Epochs with lr={optimizer.param_groups[0]["lr"]:.6f}:', 
                                total=len(train_dataloader)):
            
            B = batch_data[0].shape[0]
            pre_sequences, post_sequences = [data.to(device) for data in batch_data]
            
            # # Debug: print tensor shapes for first batch
            # if epoch == 0 and batch_idx == 0:
            #     print(f"[DEBUG] pre_sequences shape: {pre_sequences.shape}")
            #     print(f"[DEBUG] post_sequences shape: {post_sequences.shape}")
            #     print(f"[DEBUG] Expected format: [batch_size, seq_length, channels, height, width]")
            
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
        val_loss_epoch, val_fwd_loss_epoch, val_id_loss_epoch = evaluate_forward_model(
            forward_model, val_dataset, device, weight_matrix, batch_size=B, lamb=lamb)
        
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
    return train_loss, val_loss, best_val_loss


def save_training_log(train_loss, val_loss, save_path: str, stage_num: int = 0):
    os.makedirs(save_path, exist_ok=True)
    save_dict = {
        'stage': stage_num,
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    file_path = os.path.join(save_path, f'training_loss_stage{stage_num}.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(save_dict, f)
    print(f"[INFO] Training loss log for stage {stage_num} saved to: {file_path}")


def main():
    from kol_model import KOL_C_FORWARD

    set_seed()
    torch.set_default_dtype(torch.float32)
    
    # Device setup
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using {device} device")
    if torch.cuda.is_available():
        print(f"[INFO] {torch.cuda.get_device_properties(0)}")
    
    # Paths
    model_save_folder = "kol_model_weights"
    
    # Training configuration
    config = {
        'seq_length': 10,
        'num_epochs': 100,
        'decay_step': 20,
        'batch_size': 12,
        'learning_rate': 0.001,
        'decay_rate': 0.8,
        'lamb': 1.0,
        'nu': 0.1
    }
    
    print("[INFO] Starting Kolmogorov Flow Model Training")
    print(f"[INFO] Configuration: {config}")
    
    # ========================================
    # Train Forward Model
    # ========================================
    print("\n" + "="*50)
    print("TRAINING FORWARD MODEL")
    print("="*50)
    
    # Load dynamics dataset
    kol_train_dataset = KolDynamicsDataset(data_path="../../../data/kolmogorov/kolmogorov_train_data.npy",
                seq_length = config['seq_length'],
                mean=None,
                std=None)
    
    kol_val_dataset = KolDynamicsDataset(data_path="../../../data/kolmogorov/kolmogorov_val_data.npy",
                seq_length = config['seq_length'],
                mean=kol_train_dataset.mean,
                std=kol_train_dataset.std)
    
    # Create forward model
    forward_model = KOL_C_FORWARD()

    best_loss = None
    
    # Train forward model with learning rate decay
    for i in range(config['num_epochs'] // config['decay_step']):
        batch_size = config['batch_size']
        num_epochs = config['decay_step']
        learning_rate = config['learning_rate'] * (config['decay_rate'] ** i)
        
        print(f"\n[INFO] Training forward model stage {i+1}")
        print(f"[INFO] Epochs: {num_epochs}, Batch size: {batch_size}, Learning rate: {learning_rate}")
        
        train_loss_info, val_loss_info, best_loss = train_forward_model(forward_model, 
                                       kol_train_dataset,
                                       kol_val_dataset,
                                       batch_size=batch_size,
                                       num_epochs=num_epochs,
                                       learning_rate=learning_rate,
                                       model_save_folder=model_save_folder,
                                       device=device,
                                       best_val_loss=best_loss,
                                       lamb=config['lamb'])
        
        save_training_log(train_loss_info, val_loss_info, model_save_folder+'/losses', i)
    
    # Load best forward model
    forward_model.load_state_dict(torch.load(os.path.join(model_save_folder, 'forward_model.pt'), weights_only=True))
    try:
        forward_model.C_forward = torch.load(os.path.join(model_save_folder, 'C_forward.pt'), weights_only=True)
    except:
        forward_model.C_forward = torch.load(os.path.join(model_save_folder, 'C_fwd.pt'), weights_only=True)
    
    forward_model.to(device)

if __name__ == "__main__":
    main()