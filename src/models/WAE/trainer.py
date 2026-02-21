import numpy as np
import torch
import random
import json
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import yaml
from typing import Optional
import time
import psutil

import os
import sys
current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", ".."))
sys.path.append(src_directory)
from src.utils.utils import dict2namespace, count_parameters

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

def get_memory_usage():
    """Get current memory usage for CPU and GPU"""
    # CPU memory
    cpu_memory = psutil.virtual_memory().used / 1024**3  # GB
    
    # GPU memory
    gpu_memory = 0
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
    
    return cpu_memory, gpu_memory

def save_monitoring_stats(stats_dict, save_folder):
    """Save training monitoring statistics"""
    stats_path = os.path.join(save_folder, 'training_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats_dict, f)
    print(f"[INFO] Training statistics saved to: {stats_path}")

def evaluate_forward_model(forward_model, val_dataset, device='cpu', weight_matrix=None, batch_size=64, lamb=0.3, lamb_linear=0.1):
    """Evaluate forward model on validation set"""
    forward_model.eval()
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    total_loss = 0.0
    total_fwd_loss = 0.0
    total_id_loss = 0.0
    total_linear_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in val_dataloader:
            pre_sequences, post_sequences = [data.to(device) for data in batch_data]
            
            if weight_matrix is not None:
                B = batch_data[0].shape[0]
                C = batch_data[0].shape[2]
                weight_matrix_batch = torch.tensor(weight_matrix, dtype=torch.float32).repeat(B, C, 1, 1).to(device)
                loss_fwd, loss_id, loss_linear, _ = forward_model.compute_loss(pre_sequences, post_sequences, weight_matrix_batch)
            else:
                loss_fwd, loss_id, loss_linear, _ = forward_model.compute_loss(pre_sequences, post_sequences)
            
            total_fwd_loss += loss_fwd.item()
            total_id_loss += loss_id.item()
            total_linear_loss += loss_linear.item()
            total_loss += (loss_fwd + lamb * loss_id + lamb_linear * loss_linear).item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_fwd_loss = total_fwd_loss / num_batches
    avg_id_loss = total_id_loss / num_batches
    avg_linear_loss = total_linear_loss / num_batches
    
    forward_model.train()
    return avg_loss, avg_fwd_loss, avg_id_loss, avg_linear_loss


def evaluate_ms_latent_linear_model(forward_model,
                                    val_dataset,
                                    device: str = 'cpu',
                                    weight_matrix=None,
                                    batch_size: int = 64,
                                    lamb: float = 0.3,
                                    lamb_latent: float = 0.3,
                                    lamb_multi: float = 0.3,
                                    multi_step: int = 3):
    """Evaluate forward model with latent and multi-step penalties."""
    forward_model.eval()
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    total_fwd_loss = 0.0
    total_id_loss = 0.0
    total_latent_loss = 0.0
    total_ms_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_data in val_dataloader:
            pre_sequences, post_sequences = [data.to(device) for data in batch_data]

            if weight_matrix is not None:
                B = pre_sequences.shape[0]
                C = pre_sequences.shape[2]
                weight_matrix_batch = torch.tensor(weight_matrix, dtype=torch.float32).repeat(B, C, 1, 1).to(device)
                loss_fwd, loss_id, loss_multi, loss_latent, _ = forward_model.compute_loss_ms_latent_linear(
                    pre_sequences, post_sequences, multi_step, weight_matrix_batch)
            else:
                loss_fwd, loss_id, loss_multi, loss_latent, _ = forward_model.compute_loss_ms_latent_linear(
                    pre_sequences, post_sequences, multi_step)

            total_fwd_loss += loss_fwd.item()
            total_id_loss += loss_id.item()
            total_latent_loss += loss_latent.item()
            total_ms_loss += loss_multi.item()
            total_loss += (loss_fwd + lamb * loss_id + lamb_latent * loss_latent + lamb_multi * loss_multi).item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_fwd_loss = total_fwd_loss / num_batches
    avg_id_loss = total_id_loss / num_batches
    avg_latent_loss = total_latent_loss / num_batches
    avg_ms_loss = total_ms_loss / num_batches

    forward_model.train()
    return avg_loss, avg_fwd_loss, avg_id_loss, avg_latent_loss, avg_ms_loss


def train_jointly_forward_model(forward_model, 
                       train_dataset,
                       val_dataset,
                       model_save_folder: str,
                       learning_rate: float = 1e-3,
                       lamb: float = 0.3,
                       lamb_linear: float = 0.1,
                       weight_decay: float = 0,
                       batch_size: int = 64,
                       num_epochs: int = 20,
                       decay_step: int = 20,
                       decay_rate: float = 0.8,
                       gradclip: float = 1,
                       device: str = 'cpu',
                       weight_matrix=None,
                       patience: int =50):
    
    print(f"[INFO] Training forward model jointly for {num_epochs} epochs")
    if not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    num_samples = train_dataset.total_sample
    
    train_loss = {'forward': [], 'id': [], 'linear': [], 'total': []}
    val_loss = {'forward': [], 'id': [], 'linear': [], 'total': []}
    
    # Initialize monitoring variables
    epoch_times = []
    max_cpu_memory = 0
    max_gpu_memory = 0

    # Unfreeze all parameters
    for param in forward_model.parameters():
        param.requires_grad = True
    
    trainable_parameters = filter(lambda p: p.requires_grad, forward_model.parameters())
    count_parameters(forward_model)
    
    optimizer = Adam(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=decay_step, gamma=decay_rate)
    
    forward_model.train()
    forward_model.to(device)
    
    best_val_loss = np.inf
    patience = patience
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Start timing for this epoch
        epoch_start_time = time.time()

        epoch_train_fwd = []
        epoch_train_id = []
        epoch_train_linear = []
        epoch_train_total = []
        
        for batch_idx, batch_data in tqdm(enumerate(train_dataloader), 
                                desc=f'Training Epochs with lr={optimizer.param_groups[0]["lr"]:.6f}:', 
                                total=len(train_dataloader)):
            
            B = batch_data[0].shape[0]
            pre_sequences, post_sequences = [data.to(device) for data in batch_data]
            
            if weight_matrix is not None:
                C = batch_data[0].shape[2]
                weight_matrix_batch = torch.tensor(weight_matrix, dtype=torch.float32).repeat(B, C, 1, 1).to(device)
                loss_fwd, loss_id, loss_linear, _ = forward_model.compute_loss(pre_sequences, post_sequences, weight_matrix_batch)
            else:
                loss_fwd, loss_id, loss_linear, _ = forward_model.compute_loss(pre_sequences, post_sequences)
            
            loss = loss_fwd + lamb * loss_id + lamb_linear * loss_linear
            
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(forward_model.parameters(), gradclip)
            optimizer.step()
            
            epoch_train_fwd.append(loss_fwd.item())
            epoch_train_id.append(loss_id.item())
            epoch_train_linear.append(loss_linear.item())
            epoch_train_total.append(loss.item())

            # Monitor peak memory usage during training
            cpu_mem, gpu_mem = get_memory_usage()
            max_cpu_memory = max(max_cpu_memory, cpu_mem)
            max_gpu_memory = max(max_gpu_memory, gpu_mem)
        
        # Calculate average training losses
        train_loss['forward'].append(np.mean(epoch_train_fwd))
        train_loss['id'].append(np.mean(epoch_train_id))
        train_loss['linear'].append(np.mean(epoch_train_linear))
        train_loss['total'].append(np.mean(epoch_train_total))
        
        # Validation phase
        val_loss_epoch, val_fwd_loss_epoch, val_id_loss_epoch, val_linear_loss_epoch = evaluate_forward_model(
            forward_model, val_dataset, device, weight_matrix, batch_size=batch_size, lamb=lamb, lamb_linear=lamb_linear)
        
        val_loss['total'].append(val_loss_epoch)
        val_loss['forward'].append(val_fwd_loss_epoch)
        val_loss['id'].append(val_id_loss_epoch)
        val_loss['linear'].append(val_linear_loss_epoch)

        # Record epoch time
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.2f}s')
        print(f'Train - Total: {train_loss["total"][-1]:.4f}, Forward: {train_loss["forward"][-1]:.4f}, ID: {train_loss["id"][-1]:.4f}, Linear: {train_loss["linear"][-1]:.4f}')
        print(f'Val   - Total: {val_loss_epoch:.4f}, Forward: {val_fwd_loss_epoch:.4f}, ID: {val_id_loss_epoch:.4f}, Linear: {val_linear_loss_epoch:.4f}')
        
        # Save model if validation loss improved
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            patience_counter = 0
            print(f"[INFO] New best validation loss: {val_loss_epoch:.4f} - Saving model")
            forward_model.save_model(model_save_folder)
            forward_model.to(device)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[INFO] Early stopping triggered after {patience} epochs without improvement")
                break
        
        scheduler.step()
        print('')

    # Save monitoring statistics
    avg_epoch_time = np.mean(epoch_times)
    save_monitoring_stats({
        'avg_epoch_time': avg_epoch_time,
        'max_cpu_memory': max_cpu_memory,
        'max_gpu_memory': max_gpu_memory,
        'epoch_times': epoch_times
    }, model_save_folder)
    
    print(f"[INFO] Training completed. Best validation loss: {best_val_loss:.4f}")
    print(f"[INFO] Average epoch time: {avg_epoch_time:.2f}s")
    print(f"[INFO] Peak memory usage - CPU: {max_cpu_memory:.2f}GB, GPU: {max_gpu_memory:.2f}GB")
    return train_loss, val_loss


def train_ms_latent_linear_model(forward_model,
                                 train_dataset,
                                 val_dataset,
                                 model_save_folder: str,
                                 learning_rate: float = 1e-3,
                                 lamb: float = 0.3,
                                 lamb_latent: float = 0.3,
                                 lamb_multi: float = 0.3,
                                 weight_decay: float = 0,
                                 batch_size: int = 64,
                                 num_epochs: int = 20,
                                 decay_step: int = 20,
                                 decay_rate: float = 0.8,
                                 gradclip: float = 1,
                                 device: str = 'cpu',
                                 weight_matrix=None,
                                 patience: int = 50,
                                 multi_step: int = 3):
    
    print(f"[INFO] Training latent multi-step forward model for {num_epochs} epochs")
    if not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    train_loss = {'forward': [], 'id': [], 'latent': [], 'multi_step': [], 'total': []}
    val_loss = {'forward': [], 'id': [], 'latent': [], 'multi_step': [], 'total': []}
    
    epoch_times = []
    max_cpu_memory = 0
    max_gpu_memory = 0
    
    for param in forward_model.parameters():
        param.requires_grad = True
    
    count_parameters(forward_model)
    optimizer = Adam(filter(lambda p: p.requires_grad, forward_model.parameters()),
                     lr=learning_rate, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=decay_step, gamma=decay_rate)
    
    forward_model.train()
    forward_model.to(device)
    
    best_val_loss = np.inf
    patience_counter = 0
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        epoch_train_fwd = []
        epoch_train_id = []
        epoch_train_latent = []
        epoch_train_ms = []
        epoch_train_total = []
        
        for batch_idx, batch_data in tqdm(enumerate(train_dataloader),
                                          desc=f'Training Epochs with lr={optimizer.param_groups[0]["lr"]:.6f}:',
                                          total=len(train_dataloader)):
            
            pre_sequences, post_sequences = [data.to(device) for data in batch_data]
            B = pre_sequences.shape[0]
            
            if weight_matrix is not None:
                C = pre_sequences.shape[2]
                weight_matrix_batch = torch.tensor(weight_matrix, dtype=torch.float32).repeat(B, C, 1, 1).to(device)
                loss_fwd, loss_id, loss_multi, loss_latent, _ = forward_model.compute_loss_ms_latent_linear(
                    pre_sequences, post_sequences, multi_step, weight_matrix_batch)
            else:
                loss_fwd, loss_id, loss_multi, loss_latent, _ = forward_model.compute_loss_ms_latent_linear(
                    pre_sequences, post_sequences, multi_step)
            
            loss = loss_fwd + lamb * loss_id + lamb_latent * loss_latent + lamb_multi * loss_multi
            
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(forward_model.parameters(), gradclip)
            optimizer.step()
            
            epoch_train_fwd.append(loss_fwd.item())
            epoch_train_id.append(loss_id.item())
            epoch_train_latent.append(loss_latent.item())
            epoch_train_ms.append(loss_multi.item())
            epoch_train_total.append(loss.item())
            
            cpu_mem, gpu_mem = get_memory_usage()
            max_cpu_memory = max(max_cpu_memory, cpu_mem)
            max_gpu_memory = max(max_gpu_memory, gpu_mem)
        
        train_loss['forward'].append(np.mean(epoch_train_fwd))
        train_loss['id'].append(np.mean(epoch_train_id))
        train_loss['latent'].append(np.mean(epoch_train_latent))
        train_loss['multi_step'].append(np.mean(epoch_train_ms))
        train_loss['total'].append(np.mean(epoch_train_total))
        
        val_total, val_fwd, val_id, val_latent, val_ms = evaluate_ms_latent_linear_model(
            forward_model, val_dataset, device, weight_matrix, batch_size, lamb, lamb_latent, lamb_multi, multi_step)
        
        val_loss['total'].append(val_total)
        val_loss['forward'].append(val_fwd)
        val_loss['id'].append(val_id)
        val_loss['latent'].append(val_latent)
        val_loss['multi_step'].append(val_ms)
        
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.2f}s")
        print(f"Train - Total: {train_loss['total'][-1]:.4f}, Forward: {train_loss['forward'][-1]:.4f}, ID: {train_loss['id'][-1]:.4f}, Latent: {train_loss['latent'][-1]:.4f}, MS: {train_loss['multi_step'][-1]:.4f}")
        print(f"Val   - Total: {val_total:.4f}, Forward: {val_fwd:.4f}, ID: {val_id:.4f}, Latent: {val_latent:.4f}, MS: {val_ms:.4f}")
        
        if val_total < best_val_loss:
            best_val_loss = val_total
            patience_counter = 0
            print(f"[INFO] New best validation loss: {val_total:.4f} - Saving model")
            forward_model.save_model(model_save_folder)
            forward_model.to(device)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[INFO] Early stopping triggered after {patience} epochs without improvement")
                break
        
        scheduler.step()
        print('')
    
    avg_epoch_time = np.mean(epoch_times)
    save_monitoring_stats({
        'avg_epoch_time': avg_epoch_time,
        'max_cpu_memory': max_cpu_memory,
        'max_gpu_memory': max_gpu_memory,
        'epoch_times': epoch_times
    }, model_save_folder)
    
    print(f"[INFO] Training completed. Best validation loss: {best_val_loss:.4f}")
    print(f"[INFO] Average epoch time: {avg_epoch_time:.2f}s")
    print(f"[INFO] Peak memory usage - CPU: {max_cpu_memory:.2f}GB, GPU: {max_gpu_memory:.2f}GB")
    
    return train_loss, val_loss


def save_training_log(train_loss, val_loss, training_mode, save_path: str, stage_num: int = 0):
    os.makedirs(save_path, exist_ok=True)
    save_dict = {
        'stage': stage_num,
        'training_mode': training_mode,
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    file_path = os.path.join(save_path, f'training_loss_stage{stage_num}_{training_mode}.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(save_dict, f)
    print(f"[INFO] Training loss log for stage {stage_num} mode {training_mode} saved to: {file_path}")


def load_best_model(forward_model, model_save_folder: str, device: str = 'cpu'):
    """Load the best model from save folder"""
    model_path = os.path.join(model_save_folder, 'forward_model.pt')
    if os.path.exists(model_path):
        print(f"[INFO] Loading best model from {model_path}")
        forward_model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
        forward_model.to(device)
    else:
        print(f"[WARNING] No saved model found at {model_path}")
    return forward_model
