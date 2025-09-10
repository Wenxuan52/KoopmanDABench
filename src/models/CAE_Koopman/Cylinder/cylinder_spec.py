import torch
import numpy as np
from scipy import signal
import pickle

from cylinder_model import CYLINDER_C_FORWARD

import os
import sys
current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
sys.path.append(src_directory)

from src.utils.Dataset import CylinderDynamicsDataset


def rollout_prediction(forward_model, initial_state, n_steps):
    """预测未来n步"""
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
    """计算时间谱
    data: shape (T, H, W) - 时间序列数据
    """
    T, H, W = data.shape
    
    # 在每个空间点计算时间频谱，然后空间平均
    psd_sum = np.zeros(T//2)
    
    for i in range(H):
        for j in range(W):
            time_series = data[:, i, j]
            freqs, psd = signal.periodogram(time_series, fs=1/dt)
            psd_sum += psd[:T//2]
    
    # 空间平均
    psd_avg = psd_sum / (H * W)
    frequencies = np.fft.fftfreq(T, dt)[:T//2]
    
    return frequencies, psd_avg


def compute_spatial_spectrum(data, dx=1.0):
    """计算空间谱
    data: shape (T, H, W) - 时间序列数据
    """
    T, H, W = data.shape
    
    # 计算空间波数
    kx = np.fft.fftfreq(W, dx)[:W//2]
    ky = np.fft.fftfreq(H, dx)[:H//2]
    
    # 计算径向波数
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k_radial = np.sqrt(kx_grid**2 + ky_grid**2)
    
    # 定义径向波数bins
    k_bins = np.linspace(0, np.max(k_radial), min(W//4, H//4))
    k_centers = (k_bins[1:] + k_bins[:-1]) / 2
    
    psd_radial = np.zeros(len(k_centers))
    
    # 时间平均的空间功率谱
    for t in range(T):
        # 2D FFT
        fft_2d = np.fft.fft2(data[t])
        power_2d = np.abs(fft_2d[:H//2, :W//2])**2
        
        # 径向平均
        for i, k_center in enumerate(k_centers):
            mask = (k_radial >= k_bins[i]) & (k_radial < k_bins[i+1])
            if np.sum(mask) > 0:
                psd_radial[i] += np.mean(power_2d[mask])
    
    # 时间平均
    psd_radial /= T
    
    return k_centers, psd_radial


def main():
    # 参数设置
    prediction_steps = 10  # 预测步数
    start_frame = 700      # 起始帧
    val_idx = 3           # 验证集索引
    forward_step = 12
    
    print("Loading datasets...")
    # 加载数据集
    cyl_train_dataset = CylinderDynamicsDataset(
        data_path="../../../../data/cylinder/cylinder_train_data.npy",
        seq_length=forward_step,
        mean=None,
        std=None
    )
    
    cyl_val_dataset = CylinderDynamicsDataset(
        data_path="../../../../data/cylinder/cylinder_val_data.npy",
        seq_length=forward_step,
        mean=cyl_train_dataset.mean,
        std=cyl_train_dataset.std
    )
    
    denorm = cyl_val_dataset.denormalizer()
    
    print("Loading model...")
    # 加载模型
    forward_model = CYLINDER_C_FORWARD()
    forward_model.load_state_dict(torch.load(
        '../../../../results/CAE_Koopman/Cylinder/cylinder_model_weights_L512/forward_model.pt', 
        weights_only=True, map_location='cpu'))
    forward_model.C_forward = torch.load(
        '../../../../results/CAE_Koopman/Cylinder/cylinder_model_weights_L512/C_forward.pt', 
        weights_only=True, map_location='cpu')
    forward_model.eval()
    
    print(f"Computing spectra from frame {start_frame} with {prediction_steps} steps...")
    
    # 获取初始状态和真实值
    initial_state = cyl_val_dataset.data[val_idx, start_frame, ...]
    initial_state = torch.tensor(initial_state, dtype=torch.float32)
    
    groundtruth = cyl_val_dataset.data[val_idx, start_frame+1:start_frame+1+prediction_steps, ...]
    groundtruth = torch.tensor(groundtruth, dtype=torch.float32)
    
    # 计算速度幅值
    gt_uv = (groundtruth[:, 0, :, :] ** 2 + groundtruth[:, 1, :, :] ** 2) ** 0.5
    gt_uv = gt_uv.numpy()
    
    # 模型预测
    normalize_initial = cyl_val_dataset.normalize(initial_state.unsqueeze(0))[0]
    rollout = rollout_prediction(forward_model, normalize_initial, prediction_steps)
    de_rollout = denorm(rollout)
    pred_uv = (de_rollout[:, 0, :, :] ** 2 + de_rollout[:, 1, :, :] ** 2) ** 0.5
    pred_uv = pred_uv.numpy()
    
    print("Computing temporal spectra...")
    # 计算时间谱
    gt_freq, gt_temp_psd = compute_temporal_spectrum(gt_uv)
    pred_freq, pred_temp_psd = compute_temporal_spectrum(pred_uv)
    
    print("Computing spatial spectra...")
    # 计算空间谱
    gt_k, gt_spatial_psd = compute_spatial_spectrum(gt_uv)
    pred_k, pred_spatial_psd = compute_spatial_spectrum(pred_uv)
    
    # 保存结果
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
    
    # 保存到文件
    output_file = 'cylinder_spectrum_results.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {output_file}")
    print("Spectrum analysis completed!")


if __name__ == '__main__':
    main()