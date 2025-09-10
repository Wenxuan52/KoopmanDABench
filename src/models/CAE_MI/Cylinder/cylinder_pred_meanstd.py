import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
import random

from cylinder_model import CYLINDER_C_FORWARD

import os
import sys
current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
sys.path.append(src_directory)

from src.utils.Dataset import CylinderDynamicsDataset


def compute_mse_ssim(groundtruth, rollout):
    pred_np = rollout.detach().cpu().numpy()
    gt_np = groundtruth.detach().cpu().numpy()
    
    # MSE
    mse = F.mse_loss(rollout, groundtruth).item()
    
    # SSIM
    ssim_values = []
    for t in range(pred_np.shape[0]):
        ssim_val = ssim(gt_np[t, 0], pred_np[t, 0], data_range=gt_np[t, 0].max() - gt_np[t, 0].min())
        ssim_values.append(ssim_val)
    avg_ssim = np.mean(ssim_values)
    
    return {'MSE': mse, 'SSIM': avg_ssim}


def rollout_prediction(forward_model, initial_state, n_steps):
    predictions = []
    current_state = initial_state.unsqueeze(0)
    
    with torch.no_grad():
        z_current = forward_model.K_S(current_state)
        
        for step in range(n_steps):
            z_next = forward_model.latent_forward(z_current)
            next_state = forward_model.K_S_preimage(z_next)
            predictions.append(next_state)
            z_current = z_next
    
    rollout = torch.cat(predictions, dim=0)
    return rollout


if __name__ == '__main__':
    prediction_steps_list = [5, 10, 20]  # 多种预测步长
    num_starts = 50
    min_start_frame = 500
    val_idx = 3
    forward_step = 12

    seed = 42
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    cyl_train_dataset = CylinderDynamicsDataset(data_path="../../../../data/cylinder/cylinder_train_data.npy",
                seq_length=forward_step,
                mean=None,
                std=None)
    
    cyl_val_dataset = CylinderDynamicsDataset(data_path="../../../../data/cylinder/cylinder_val_data.npy",
                seq_length=forward_step,
                mean=cyl_train_dataset.mean,
                std=cyl_train_dataset.std)

    denorm = cyl_val_dataset.denormalizer()
    
    forward_model = CYLINDER_C_FORWARD()
    forward_model.load_state_dict(torch.load('../../../../results/CAE_MI/Cylinder/cylinder_model_weights_L512/forward_model.pt', weights_only=True, map_location='cpu'))
    forward_model.C_forward = torch.load('../../../../results/CAE_MI/Cylinder/cylinder_model_weights_L512/C_forward.pt', weights_only=True, map_location='cpu')
    forward_model.eval()
    
    total_frames = cyl_val_dataset.data.shape[1]
    
    # 对每种预测步长进行评估
    for prediction_steps in prediction_steps_list:
        print(f"\n{'='*60}")
        print(f"EVALUATING {prediction_steps}-STEP PREDICTIONS")
        print(f"{'='*60}")
        
        max_start_frame = total_frames - prediction_steps - 1
        start_frames = random.sample(range(min_start_frame, max_start_frame), num_starts)
        print(f"Selected start frames: {start_frames}")
        
        all_metrics = []
        
        for i, start_frame in enumerate(start_frames):
            print(f"Processing start frame {start_frame} ({i+1}/{num_starts})", end='\r')
            
            initial_state = cyl_val_dataset.data[val_idx, start_frame, ...]
            initial_state = torch.tensor(initial_state, dtype=torch.float32)
            
            groundtruth = cyl_val_dataset.data[val_idx, start_frame+1:start_frame+1+prediction_steps, ...]
            groundtruth = torch.tensor(groundtruth, dtype=torch.float32)
            
            raw_data_uv = (groundtruth[:, 0, :, :] ** 2 + groundtruth[:, 1, :, :] ** 2) ** 0.5
            raw_data_uv = raw_data_uv.unsqueeze(1)
            
            normalize_initial = cyl_val_dataset.normalize(initial_state.unsqueeze(0))[0]
            
            rollout = rollout_prediction(forward_model, normalize_initial, prediction_steps)
            
            de_rollout = denorm(rollout)
            de_rollout_uv = (de_rollout[:, 0, :, :] ** 2 + de_rollout[:, 1, :, :] ** 2) ** 0.5
            de_rollout_uv = de_rollout_uv.unsqueeze(1)
            
            metrics = compute_mse_ssim(raw_data_uv, de_rollout_uv)
            all_metrics.append(metrics)

        print(f"\nFINAL RESULTS - {prediction_steps}-step predictions (Mean ± Std)")
        print("-" * 50)
        
        for metric_name in ['MSE', 'SSIM']:
            values = [metrics[metric_name] for metrics in all_metrics]
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{metric_name}: {mean_val:.6f} ± {std_val:.6f}")
        
        print(f"Total evaluated trajectories: {num_starts}")
    
    print(f"\n{'='*60}")
    print("ALL EVALUATIONS COMPLETED")
    print(f"{'='*60}")