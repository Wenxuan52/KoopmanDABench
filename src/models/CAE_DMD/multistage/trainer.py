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
import gc

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


def evaluate_reconstruction_model(dmd_model, val_dataset, device='cpu', weight_matrix=None, batch_size=64):
    """Evaluate CAE reconstruction on validation set (Stage 1)"""
    dmd_model.eval()
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in val_dataloader:
            sequences = batch_data[0].to(device)  # Only need current states for reconstruction
            B, T = sequences.shape[:2]
            
            total_recon_loss = 0.0
            for i in range(T):
                state = sequences[:, i, :]
                z = dmd_model.K_S(state)
                recon_state = dmd_model.K_S_preimage(z)
                
                if weight_matrix is not None:
                    C = sequences.shape[2]
                    weight_matrix_batch = torch.tensor(weight_matrix, dtype=torch.float32).repeat(B, C, 1, 1).to(device)
                    from src.utils.utils import weighted_MSELoss
                    recon_loss = weighted_MSELoss()(recon_state, state, weight_matrix_batch[:, i]).sum()
                else:
                    recon_loss = torch.nn.functional.mse_loss(recon_state, state)
                
                total_recon_loss += recon_loss
            
            total_loss += total_recon_loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    dmd_model.train()
    return avg_loss


def evaluate_dmd_model(dmd_model, val_dataset, device='cpu', weight_matrix=None, batch_size=64, lamb=0.3):
    """Evaluate DMD forward model on validation set (Stage 3)"""
    dmd_model.eval()
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
                loss_fwd, loss_id, _ = dmd_model.compute_loss(pre_sequences, post_sequences, weight_matrix_batch)
            else:
                loss_fwd, loss_id, _ = dmd_model.compute_loss(pre_sequences, post_sequences)
            
            total_fwd_loss += loss_fwd.item()
            total_id_loss += loss_id.item()
            total_loss += (loss_fwd + lamb * loss_id).item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_fwd_loss = total_fwd_loss / num_batches
    avg_id_loss = total_id_loss / num_batches
    
    dmd_model.train()
    return avg_loss, avg_fwd_loss, avg_id_loss


def train_stage1_reconstruction(dmd_model, 
                               train_dataset,
                               val_dataset,
                               model_save_folder: str,
                               learning_rate: float = 1e-3,
                               weight_decay: float = 0,
                               batch_size: int = 64,
                               num_epochs: int = 50,
                               decay_step: int = 20,
                               decay_rate: float = 0.8,
                               gradclip: float = 1,
                               device: str = 'cpu',
                               weight_matrix=None,
                               patience: int = 50):
    
    print(f"[INFO] Stage 1: Training CAE reconstruction for {num_epochs} epochs")
    if not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    train_loss = []
    val_loss = []
    
    # Initialize monitoring variables
    epoch_times = []
    max_cpu_memory = 0
    max_gpu_memory = 0
    
    # Only train encoder and decoder
    for param in dmd_model.parameters():
        param.requires_grad = False
    for param in dmd_model.K_S.parameters():
        param.requires_grad = True
    for param in dmd_model.K_S_preimage.parameters():
        param.requires_grad = True
    
    trainable_parameters = filter(lambda p: p.requires_grad, dmd_model.parameters())
    count_parameters(dmd_model)
    
    optimizer = Adam(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=decay_step, gamma=decay_rate)
    
    dmd_model.train()
    dmd_model.to(device)
    
    best_val_loss = np.inf
    patience_counter = 0
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_train_loss = []
        
        for batch_idx, batch_data in tqdm(enumerate(train_dataloader), 
                                desc=f'Stage 1 - Epoch {epoch+1}/{num_epochs} (lr={optimizer.param_groups[0]["lr"]:.6f}):', 
                                total=len(train_dataloader)):
            
            sequences = batch_data[0].to(device)  # Only need current states
            B, T = sequences.shape[:2]
            
            total_recon_loss = 0.0
            for i in range(T):
                state = sequences[:, i, :]
                z = dmd_model.K_S(state)
                recon_state = dmd_model.K_S_preimage(z)
                
                if weight_matrix is not None:
                    C = sequences.shape[2]
                    weight_matrix_batch = torch.tensor(weight_matrix, dtype=torch.float32).repeat(B, C, 1, 1).to(device)
                    from src.utils.utils import weighted_MSELoss
                    recon_loss = weighted_MSELoss()(recon_state, state, weight_matrix_batch[:, i]).sum()
                else:
                    recon_loss = torch.nn.functional.mse_loss(recon_state, state)
                
                total_recon_loss += recon_loss
            
            optimizer.zero_grad()
            total_recon_loss.backward()
            clip_grad_norm_(dmd_model.parameters(), gradclip)
            optimizer.step()
            
            epoch_train_loss.append(total_recon_loss.item())
            
            # Monitor peak memory usage during training
            cpu_mem, gpu_mem = get_memory_usage()
            max_cpu_memory = max(max_cpu_memory, cpu_mem)
            max_gpu_memory = max(max_gpu_memory, gpu_mem)
        
        # Calculate average training loss
        train_loss.append(np.mean(epoch_train_loss))
        
        # Validation phase
        val_loss_epoch = evaluate_reconstruction_model(dmd_model, val_dataset, device, weight_matrix, batch_size)
        val_loss.append(val_loss_epoch)
        
        # Record epoch time
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.2f}s')
        print(f'Train Reconstruction Loss: {train_loss[-1]:.4f}')
        print(f'Val   Reconstruction Loss: {val_loss_epoch:.4f}')
        
        # Save model if validation loss improved
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            patience_counter = 0
            print(f"[INFO] New best validation loss: {val_loss_epoch:.4f} - Saving model")
            stage1_save_path = os.path.join(model_save_folder, 'stage1')
            os.makedirs(stage1_save_path, exist_ok=True)
            dmd_model.save_model(stage1_save_path)
            dmd_model.to(device)
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
        'stage': 1,
        'avg_epoch_time': avg_epoch_time,
        'max_cpu_memory': max_cpu_memory,
        'max_gpu_memory': max_gpu_memory,
        'epoch_times': epoch_times
    }, model_save_folder)
    
    print(f"[INFO] Stage 1 completed. Best validation loss: {best_val_loss:.4f}")
    print(f"[INFO] Average epoch time: {avg_epoch_time:.2f}s")
    return train_loss, val_loss


def train_stage2_dmd_fitting(dmd_model,
                           train_dataset,
                           model_save_folder: str,
                           batch_size: int = 64,
                           device: str = 'cpu'):
    
    print(f"[INFO] Stage 2: Fitting DMD algorithm on latent representations")
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    # Collect all latent representations
    all_z_current = []
    all_z_next = []
    
    dmd_model.eval()
    dmd_model.to(device)
    
    print("[INFO] Collecting latent representations...")
    with torch.no_grad():
        for batch_data in tqdm(train_dataloader, desc="Encoding states to latent space"):
            pre_sequences, post_sequences = [data.to(device) for data in batch_data]
            B, T = pre_sequences.shape[:2]
            
            for i in range(T):
                z_current = dmd_model.K_S(pre_sequences[:, i, :])
                z_next = dmd_model.K_S(post_sequences[:, i, :])
                
                all_z_current.append(z_current)
                all_z_next.append(z_next)
    
    # Stack all latent representations
    Z_current = torch.cat(all_z_current, dim=0)  # [N, hidden_dim]
    Z_next = torch.cat(all_z_next, dim=0)       # [N, hidden_dim]
    
    print(f"[INFO] Collected {Z_current.shape[0]} latent state pairs")
    print(f"[INFO] Latent dimension: {Z_current.shape[1]}")
    
    # Prepare data matrices for DMD: X and Y should be [hidden_dim, N]
    X = Z_current.T  # [hidden_dim, N]
    Y = Z_next.T     # [hidden_dim, N]
    
    print("[INFO] Computing DMD matrices...")
    start_time = time.time()
    
    # Compute DMD matrices
    A_dmd = dmd_model.compute_dmd_matrices(X, Y)
    
    dmd_time = time.time() - start_time
    print(f"[INFO] DMD computation completed in {dmd_time:.2f}s")
    
    # Analyze DMD results
    if dmd_model.Lambda is not None:
        stability_info = dmd_model.analyze_stability()
        print(f"[INFO] System is {'stable' if stability_info['is_stable'] else 'unstable'}")
        print(f"[INFO] Max eigenvalue magnitude: {stability_info['max_eigenvalue_magnitude']:.4f}")
        print(f"[INFO] Number of unstable modes: {stability_info['num_unstable_modes']}")
        
        growth_rates, frequencies = dmd_model.get_growth_rates_and_frequencies()
        print(f"[INFO] Growth rate range: [{torch.min(growth_rates):.4f}, {torch.max(growth_rates):.4f}]")
        print(f"[INFO] Frequency range: [{torch.min(frequencies):.4f}, {torch.max(frequencies):.4f}]")
    
    # Save DMD components
    stage2_save_path = os.path.join(model_save_folder, 'stage2')
    os.makedirs(stage2_save_path, exist_ok=True)
    dmd_model.save_dmd_components(stage2_save_path)
    
    # Save stage 2 statistics
    stage2_stats = {
        'stage': 2,
        'dmd_computation_time': dmd_time,
        'num_data_points': Z_current.shape[0],
        'latent_dimension': Z_current.shape[1],
        'stability_info': stability_info if dmd_model.Lambda is not None else None
    }
    
    stats_path = os.path.join(stage2_save_path, 'dmd_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stage2_stats, f)
    
    print(f"[INFO] Stage 2 completed.")
    print(f"[INFO] DMD components saved to: {stage2_save_path}")
    
    return stage2_stats


def train_stage3_joint_optimization(dmd_model,
                                  train_dataset,
                                  val_dataset,
                                  model_save_folder: str,
                                  learning_rate: float = 1e-4,
                                  lamb: float = 0.3,
                                  weight_decay: float = 0,
                                  batch_size: int = 64,
                                  num_epochs: int = 30,
                                  decay_step: int = 15,
                                  decay_rate: float = 0.8,
                                  gradclip: float = 1,
                                  device: str = 'cpu',
                                  weight_matrix=None,
                                  patience: int = 30):
    
    print(f"[INFO] Stage 3: Joint optimization of CAE+DMD for {num_epochs} epochs")
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    num_samples = train_dataset.total_sample
    
    train_loss = {'forward': [], 'id': [], 'total': []}
    val_loss = {'forward': [], 'id': [], 'total': []}
    
    # Initialize monitoring variables
    epoch_times = []
    max_cpu_memory = 0
    max_gpu_memory = 0
    
    # Unfreeze all parameters for joint training
    for param in dmd_model.parameters():
        param.requires_grad = True
    
    trainable_parameters = filter(lambda p: p.requires_grad, dmd_model.parameters())
    count_parameters(dmd_model)
    
    optimizer = Adam(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=decay_step, gamma=decay_rate)
    
    dmd_model.train()
    dmd_model.to(device)
    
    best_val_loss = np.inf
    patience_counter = 0
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        epoch_train_fwd = []
        epoch_train_id = []
        epoch_train_total = []
        
        for batch_idx, batch_data in tqdm(enumerate(train_dataloader), 
                                desc=f'Stage 3 - Epoch {epoch+1}/{num_epochs} (lr={optimizer.param_groups[0]["lr"]:.6f}):', 
                                total=len(train_dataloader)):
            
            B = batch_data[0].shape[0]
            if weight_matrix is not None:
                C = batch_data[0].shape[2]
                weight_matrix_batch = torch.tensor(weight_matrix, dtype=torch.float32).repeat(B, C, 1, 1).to(device)
                pre_sequences, post_sequences = [each_data.to(device) for each_data in batch_data]
                loss_fwd, loss_id, A_dmd = dmd_model.compute_loss(pre_sequences, post_sequences, weight_matrix_batch)
            else:
                pre_sequences, post_sequences = [each_data.to(device) for each_data in batch_data]
                loss_fwd, loss_id, A_dmd = dmd_model.compute_loss(pre_sequences, post_sequences)
            
            loss = loss_fwd + lamb * loss_id
            
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(dmd_model.parameters(), gradclip)
            optimizer.step()
            
            epoch_train_fwd.append(loss_fwd.item())
            epoch_train_id.append(loss_id.item())
            epoch_train_total.append(loss.item())
            
            # Monitor peak memory usage during training
            cpu_mem, gpu_mem = get_memory_usage()
            max_cpu_memory = max(max_cpu_memory, cpu_mem)
            max_gpu_memory = max(max_gpu_memory, gpu_mem)
        
        # Analyze DMD matrix properties
        if A_dmd is not None:
            U, S, Vh = torch.linalg.svd(A_dmd.detach())
            print(f"DMD matrix max singular value: {S.max():.6f}")
        
        # Calculate average training losses
        train_loss['forward'].append(np.mean(epoch_train_fwd))
        train_loss['id'].append(np.mean(epoch_train_id))
        train_loss['total'].append(np.mean(epoch_train_total))
        
        # Validation phase
        val_loss_epoch, val_fwd_loss_epoch, val_id_loss_epoch = evaluate_dmd_model(
            dmd_model, val_dataset, device, weight_matrix, batch_size=batch_size, lamb=lamb)
        
        val_loss['total'].append(val_loss_epoch)
        val_loss['forward'].append(val_fwd_loss_epoch)
        val_loss['id'].append(val_id_loss_epoch)
        
        # Record epoch time
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.2f}s')
        print(f'Train - Total: {train_loss["total"][-1]:.4f}, Forward: {train_loss["forward"][-1]:.4f}, ID: {train_loss["id"][-1]:.4f}')
        print(f'Val   - Total: {val_loss_epoch:.4f}, Forward: {val_fwd_loss_epoch:.4f}, ID: {val_id_loss_epoch:.4f}')
        
        # Save model if validation loss improved
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            patience_counter = 0
            print(f"[INFO] New best validation loss: {val_loss_epoch:.4f} - Saving model")
            stage3_save_path = os.path.join(model_save_folder, 'stage3')
            os.makedirs(stage3_save_path, exist_ok=True)
            dmd_model.save_model(stage3_save_path)
            dmd_model.save_dmd_components(stage3_save_path)
            dmd_model.to(device)
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
        'stage': 3,
        'avg_epoch_time': avg_epoch_time,
        'max_cpu_memory': max_cpu_memory,
        'max_gpu_memory': max_gpu_memory,
        'epoch_times': epoch_times
    }, model_save_folder)
    
    print(f"[INFO] Stage 3 completed. Best validation loss: {best_val_loss:.4f}")
    print(f"[INFO] Average epoch time: {avg_epoch_time:.2f}s")
    print(f"[INFO] Peak memory usage - CPU: {max_cpu_memory:.2f}GB, GPU: {max_gpu_memory:.2f}GB")
    
    return train_loss, val_loss


def save_monitoring_stats(stats_dict, save_folder):
    """Save training monitoring statistics"""
    stage = stats_dict.get('stage', 0)
    stats_path = os.path.join(save_folder, f'training_stats_stage{stage}.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats_dict, f)
    print(f"[INFO] Stage {stage} training statistics saved to: {stats_path}")


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


def train_cae_dmd_multistage(dmd_model,
                           train_dataset,
                           val_dataset,
                           model_save_folder: str,
                           stage1_config: dict,
                           stage2_config: dict,
                           stage3_config: dict,
                           device: str = 'cpu'):
    """
    Complete multi-stage training pipeline for CAE+DMD
    """
    print("[INFO] Starting multi-stage CAE+DMD training pipeline")
    print("=" * 60)
    
    # Stage 1: CAE Reconstruction Training
    print("\n" + "=" * 20 + " STAGE 1 " + "=" * 20)
    stage1_train_loss, stage1_val_loss = train_stage1_reconstruction(
        dmd_model, train_dataset, val_dataset, model_save_folder, 
        device=device, **stage1_config
    )
    save_training_log(stage1_train_loss, stage1_val_loss, model_save_folder, stage_num=1)
    
    # Stage 2: DMD Fitting
    print("\n" + "=" * 20 + " STAGE 2 " + "=" * 20)
    stage2_stats = train_stage2_dmd_fitting(
        dmd_model, train_dataset, model_save_folder,
        device=device, **stage2_config
    )
    
    # Stage 3: Joint Optimization
    print("\n" + "=" * 20 + " STAGE 3 " + "=" * 20)
    stage3_train_loss, stage3_val_loss = train_stage3_joint_optimization(
        dmd_model, train_dataset, val_dataset, model_save_folder,
        device=device, **stage3_config
    )
    save_training_log(stage3_train_loss, stage3_val_loss, model_save_folder, stage_num=3)
    
    print("\n" + "=" * 50)
    print("[INFO] Multi-stage training pipeline completed successfully!")
    print("=" * 50)
    
    return {
        'stage1': {'train_loss': stage1_train_loss, 'val_loss': stage1_val_loss},
        'stage2': stage2_stats,
        'stage3': {'train_loss': stage3_train_loss, 'val_loss': stage3_val_loss}
    }