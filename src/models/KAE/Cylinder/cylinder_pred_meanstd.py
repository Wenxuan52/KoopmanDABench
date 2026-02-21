import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
import random
import os
import sys
from tqdm import tqdm

from cylinder_model import CYLINDER_C_FORWARD

# 路径注册
current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
sys.path.append(src_directory)

from src.utils.Dataset import CylinderDynamicsDataset


# ===========================
# 计算指标函数
# ===========================
def compute_metrics(gt, pred):
    """
    输入张量格式: (T, 1, H, W)
    返回: MSE, RRMSE, SSIM (逐帧)
    """
    gt_np = gt.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    
    mse_list, rrmse_list, ssim_list = [], [], []
    
    for t in range(gt_np.shape[0]):
        g = gt_np[t, 0]
        p = pred_np[t, 0]
        mse_val = np.mean((p - g) ** 2)
        rrmse_val = np.sqrt(mse_val) / (np.sqrt(np.mean(g ** 2)) + 1e-8)
        ssim_val = ssim(g, p, data_range=g.max() - g.min())
        
        mse_list.append(mse_val)
        rrmse_list.append(rrmse_val)
        ssim_list.append(ssim_val)
    
    return np.array(mse_list), np.array(rrmse_list), np.array(ssim_list)


# ===========================
# Rollout 函数
# ===========================
def rollout_prediction(forward_model, initial_state, n_steps):
    predictions = []
    current_state = initial_state.unsqueeze(0)
    
    with torch.no_grad():
        z_current = forward_model.K_S(current_state)
        for _ in range(n_steps):
            z_next = forward_model.latent_forward(z_current)
            next_state = forward_model.K_S_preimage(z_next)
            predictions.append(next_state)
            z_current = z_next
    
    rollout = torch.cat(predictions, dim=0)
    return rollout


# ===========================
# 主流程
# ===========================
if __name__ == '__main__':
    # -------------------
    # 参数配置
    # -------------------
    prediction_steps = 30
    num_starts = 50
    start_min, start_max = 700, 950
    val_idx = 3
    forward_step = 12
    seed = 42

    model_name = 'KAE'

    # 随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # -------------------
    # 数据加载
    # -------------------
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

    # -------------------
    # 模型加载
    # -------------------
    forward_model = CYLINDER_C_FORWARD()
    forward_model.load_state_dict(torch.load(
        f'../../../../results/{model_name}/Cylinder/4loss_model/forward_model.pt',
        weights_only=True, map_location='cpu'
    ))
    forward_model.eval()

    # -------------------
    # 起点采样
    # -------------------
    start_frames = random.sample(range(start_min, start_max), num_starts)
    print(f"\nSelected start frames ({num_starts} total): {start_frames}")

    # 存储每条轨迹的指标
    all_mse = []
    all_rrmse = []
    all_ssim = []

    # -------------------
    # rollout + 评估
    # -------------------
    for start_frame in tqdm(start_frames, desc="Evaluating rollouts"):
        initial_state = torch.tensor(
            cyl_val_dataset.data[val_idx, start_frame, ...],
            dtype=torch.float32
        )
        groundtruth = torch.tensor(
            cyl_val_dataset.data[val_idx, start_frame+1:start_frame+1+prediction_steps, ...],
            dtype=torch.float32
        )

        raw_uv = (groundtruth[:, 0, :, :] ** 2 + groundtruth[:, 1, :, :] ** 2) ** 0.5
        raw_uv = raw_uv.unsqueeze(1)

        normalize_initial = cyl_val_dataset.normalize(initial_state.unsqueeze(0))[0]
        rollout = rollout_prediction(forward_model, normalize_initial, prediction_steps)

        de_rollout = denorm(rollout)
        de_rollout_uv = (de_rollout[:, 0, :, :] ** 2 + de_rollout[:, 1, :, :] ** 2) ** 0.5
        de_rollout_uv = de_rollout_uv.unsqueeze(1)

        mse_seq, rrmse_seq, ssim_seq = compute_metrics(raw_uv, de_rollout_uv)
        all_mse.append(mse_seq)
        all_rrmse.append(rrmse_seq)
        all_ssim.append(ssim_seq)

    # -------------------
    # 汇总统计
    # -------------------
    all_mse = np.stack(all_mse)      # [50, 30]
    all_rrmse = np.stack(all_rrmse)
    all_ssim = np.stack(all_ssim)

    # (1) 整体平均指标（所有步平均）
    overall_mse = np.mean(all_mse)
    overall_rrmse = np.mean(all_rrmse)
    overall_ssim = np.mean(all_ssim)
    overall_std_mse = np.std(all_mse)
    overall_std_rrmse = np.std(all_rrmse)
    overall_std_ssim = np.std(all_ssim)

    print("\n================ FINAL METRICS (Overall across 50 trajectories) ================")
    print(f"MSE   : {overall_mse:.6e} ± {overall_std_mse:.6e}")
    print(f"RRMSE : {overall_rrmse:.6e} ± {overall_std_rrmse:.6e}")
    print(f"SSIM  : {overall_ssim:.6e} ± {overall_std_ssim:.6e}")
    print("===============================================================================")

    # (2) 每步均值与方差
    mse_mean_per_step = np.mean(all_mse, axis=0)
    mse_std_per_step = np.std(all_mse, axis=0)
    rrmse_mean_per_step = np.mean(all_rrmse, axis=0)
    rrmse_std_per_step = np.std(all_rrmse, axis=0)
    ssim_mean_per_step = np.mean(all_ssim, axis=0)
    ssim_std_per_step = np.std(all_ssim, axis=0)

    print("\nPer-step metric summary (mean ± std):")
    for t in range(prediction_steps):
        print(f"Step {t+1:02d}: "
              f"MSE={mse_mean_per_step[t]:.6e}±{mse_std_per_step[t]:.2e}, "
              f"RRMSE={rrmse_mean_per_step[t]:.6e}±{rrmse_std_per_step[t]:.2e}, "
              f"SSIM={ssim_mean_per_step[t]:.6f}±{ssim_std_per_step[t]:.3f}")

    # -------------------
    # 保存结果
    # -------------------
    os.makedirs(f"../../../../results/{model_name}/figures", exist_ok=True)
    np.savez(
        f"../../../../results/{model_name}/figures/metrics_cylinder_forward.npz",
        all_mse=all_mse,
        all_rrmse=all_rrmse,
        all_ssim=all_ssim,
        mse_mean_per_step=mse_mean_per_step,
        mse_std_per_step=mse_std_per_step,
        rrmse_mean_per_step=rrmse_mean_per_step,
        rrmse_std_per_step=rrmse_std_per_step,
        ssim_mean_per_step=ssim_mean_per_step,
        ssim_std_per_step=ssim_std_per_step,
        overall_mse=overall_mse,
        overall_rrmse=overall_rrmse,
        overall_ssim=overall_ssim,
        overall_std_mse=overall_std_mse,
        overall_std_rrmse=overall_std_rrmse,
        overall_std_ssim=overall_std_ssim,
        start_frames=np.array(start_frames)
    )

    print(f"\nSaved metrics to: results/{model_name}/figures/metrics_cylinder_forward.npz")
    print("All evaluations completed successfully ")
