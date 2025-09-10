import torch
import numpy as np
from scipy import signal
import pickle

from dam_model import DAM_C_FORWARD

import os
import sys
current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
sys.path.append(src_directory)

from src.utils.Dataset import DamDynamicsDataset


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


def compute_temporal_spectrum(data, dt=1.0):
    T, H, W = data.shape
    
    psd_sum = np.zeros(T//2)
    
    for i in range(H):
        for j in range(W):
            time_series = data[:, i, j]
            freqs, psd = signal.periodogram(time_series, fs=1/dt)
            psd_sum += psd[:T//2]
    
    psd_avg = psd_sum / (H * W)
    frequencies = np.fft.fftfreq(T, dt)[:T//2]
    
    return frequencies, psd_avg


def compute_spatial_spectrum(data, dx=1.0):
    T, H, W = data.shape
    
    kx = np.fft.fftfreq(W, dx)[:W//2]
    ky = np.fft.fftfreq(H, dx)[:H//2]
    
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k_radial = np.sqrt(kx_grid**2 + ky_grid**2)
    
    k_bins = np.linspace(0, np.max(k_radial), min(W//4, H//4))
    k_centers = (k_bins[1:] + k_bins[:-1]) / 2
    
    psd_radial = np.zeros(len(k_centers))
    
    for t in range(T):
        fft_2d = np.fft.fft2(data[t])
        power_2d = np.abs(fft_2d[:H//2, :W//2])**2
        
        for i, k_center in enumerate(k_centers):
            mask = (k_radial >= k_bins[i]) & (k_radial < k_bins[i+1])
            if np.sum(mask) > 0:
                psd_radial[i] += np.mean(power_2d[mask])
    
    psd_radial /= T
    
    return k_centers, psd_radial


def main():
    prediction_steps = 50
    start_frame = 50
    val_idx = -1
    forward_step = 12
    
    print("Loading datasets...")
    dam_train_dataset = DamDynamicsDataset(
        data_path="../../../../data/dam/dam_train_data.npy",
        seq_length=forward_step,
        mean=None,
        std=None
    )
    
    dam_val_dataset = DamDynamicsDataset(
        data_path="../../../../data/dam/dam_val_data.npy",
        seq_length=forward_step,
        mean=dam_train_dataset.mean,
        std=dam_train_dataset.std
    )
    
    denorm = dam_val_dataset.denormalizer()
    
    print("Loading model...")
    forward_model = DAM_C_FORWARD()
    forward_model.load_state_dict(torch.load(
        '../../../../results/CAE_MLP/Dam/model_weights_L512/forward_model.pt', 
        weights_only=True, map_location='cpu'))
    forward_model.eval()
    
    print(f"Computing spectra from frame {start_frame} with {prediction_steps} steps...")
    
    initial_state = dam_val_dataset.data[val_idx, start_frame, ...]
    initial_state = torch.tensor(initial_state, dtype=torch.float32)
    
    groundtruth = dam_val_dataset.data[val_idx, start_frame+1:start_frame+1+prediction_steps, ...]
    groundtruth = torch.tensor(groundtruth, dtype=torch.float32)
    
    gt_uv = (groundtruth[:, 0, :, :] ** 2 + groundtruth[:, 1, :, :] ** 2) ** 0.5
    gt_uv = gt_uv.numpy()
    
    normalize_initial = dam_val_dataset.normalize(initial_state.unsqueeze(0))[0]
    rollout = rollout_prediction(forward_model, normalize_initial, prediction_steps)
    de_rollout = denorm(rollout)
    pred_uv = (de_rollout[:, 0, :, :] ** 2 + de_rollout[:, 1, :, :] ** 2) ** 0.5
    pred_uv = pred_uv.numpy()
    
    print("Computing temporal spectra...")
    gt_freq, gt_temp_psd = compute_temporal_spectrum(gt_uv)
    pred_freq, pred_temp_psd = compute_temporal_spectrum(pred_uv)
    
    print("Computing spatial spectra...")
    gt_k, gt_spatial_psd = compute_spatial_spectrum(gt_uv)
    pred_k, pred_spatial_psd = compute_spatial_spectrum(pred_uv)
    
    results = {
        'temporal': {
            'frequencies': gt_freq,
            'groundtruth_psd': gt_temp_psd,
            'prediction_psd': pred_temp_psd
        },
        'spatial': {
            'wavenumbers': gt_k,
            'groundtruth_psd': gt_spatial_psd,
            'prediction_psd': pred_spatial_psd
        },
        'parameters': {
            'prediction_steps': prediction_steps,
            'start_frame': start_frame,
            'val_idx': val_idx
        }
    }
    
    output_file = 'dam_spectrum_results.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {output_file}")
    print("Spectrum analysis completed!")


if __name__ == '__main__':
    main()