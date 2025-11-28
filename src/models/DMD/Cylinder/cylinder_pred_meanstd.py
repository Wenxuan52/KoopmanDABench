import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
import random
import os
import sys
from tqdm import tqdm

# 路径注册
current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
sys.path.append(src_directory)

from src.utils.Dataset import CylinderDynamicsDataset
from src.models.DMD.base import TorchDMD


# ===========================
# 计算指标函数（逐帧）
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
        # RRMSE: RMSE / RMS(gt)
        rrmse_val = np.sqrt(mse_val) / (np.sqrt(np.mean(g ** 2)) + 1e-8)
        ssim_val = ssim(g, p, data_range=g.max() - g.min())

        mse_list.append(mse_val)
        rrmse_list.append(rrmse_val)
        ssim_list.append(ssim_val)

    return np.array(mse_list), np.array(rrmse_list), np.array(ssim_list)


# ===========================
# DMD rollout（矢量化）
# ===========================
def dmd_rollout_states(dmd: TorchDMD, x0_flat: torch.Tensor, n_steps: int) -> torch.Tensor:
    """
    使用 DMD 在“归一化后的物理空间（flatten）”上 rollout。
    x0_flat: shape [D]，是初始帧的扁平化张量（实数），已归一化
    返回: [D, n_steps] 的张量，分别对应 t=1..n_steps 的预测
    """
    # 1) 最小二乘得到初始 latent 系数 b0  (m,)
    x0_c = x0_flat.to(torch.complex64)
    # dmd.modes: [D, m] (complex), solve modes @ b = x
    b0 = torch.linalg.lstsq(dmd.modes, x0_c).solution  # (m,)

    # 2) 预生成每一步的 Λ^t b0（Λ 为 diag(eigvals)）
    # eigvals: (m,)
    eig = dmd.eigenvalues  # complex
    # 生成 [m, n_steps]，第 t 列为 eig ** (t) 元素
    # 注意：第1步对应 t=1
    powers = torch.stack([torch.pow(eig, t) for t in range(1, n_steps + 1)], dim=1)  # (m, n_steps)
    B = powers * b0.unsqueeze(1)  # (m, n_steps)

    # 3) 解码到物理空间（一次性矩阵乘）
    # modes: [D, m], B: [m, n_steps] -> X: [D, n_steps]
    X = (dmd.modes @ B).real  # 取实部
    return X  # [D, n_steps]


def flatten_frame(frame: torch.Tensor) -> torch.Tensor:
    """(C, H, W) -> [D]"""
    return frame.reshape(-1)


def unflatten_traj(traj_flat: torch.Tensor, C: int, H: int, W: int) -> torch.Tensor:
    """[D, T] -> (T, C, H, W)"""
    D, T = traj_flat.shape
    return traj_flat.T.reshape(T, C, H, W)


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
    forward_step = 12  # 用于数据集窗口长度（与训练一致即可）
    seed = 42

    model_name = 'DMD'  # 仅用于输出路径组织

    # 随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

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

    denorm = cyl_val_dataset.denormalizer()  # 还原到物理量域

    # -------------------
    # DMD 模型加载
    # -------------------
    dmd = TorchDMD(svd_rank=512, device=device)
    dmd.load_dmd('../../../../results/DMD/Cylinder/dmd_model.pth')  # modes/eigvals 已加载到 dmd
    # dmd.modes: [D, m] (complex), dmd.eigenvalues: [m] (complex)

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
    for start_frame in tqdm(start_frames, desc="Evaluating DMD rollouts"):
        # 取初始帧与真值序列（未归一化，物理域）
        initial_state = torch.tensor(
            cyl_val_dataset.data[val_idx, start_frame, ...], dtype=torch.float32
        )  # (C, H, W)
        groundtruth = torch.tensor(
            cyl_val_dataset.data[val_idx, start_frame + 1:start_frame + 1 + prediction_steps, ...],
            dtype=torch.float32
        )  # (T, C, H, W)

        # 计算真值的速度模（物理域）
        raw_uv = (groundtruth[:, 0, :, :] ** 2 + groundtruth[:, 1, :, :] ** 2) ** 0.5  # (T, H, W)
        raw_uv = raw_uv.unsqueeze(1)  # (T, 1, H, W)

        # 归一化到与 DMD 训练一致的域
        norm_initial = cyl_val_dataset.normalize(initial_state.unsqueeze(0))[0]  # (C, H, W)

        # flatten 初始帧
        x0_flat = flatten_frame(norm_initial).to(device)  # [D]

        # DMD 矢量化 rollout（得到 [D, T] 的归一化预测）
        with torch.no_grad():
            rollout_flat = dmd_rollout_states(dmd, x0_flat, prediction_steps)  # [D, T]
            rollout_norm = unflatten_traj(rollout_flat.cpu(), *initial_state.shape)  # (T, C, H, W)

            # 反归一化回物理域
            rollout_denorm = denorm(rollout_norm)  # (T, C, H, W)

            # 计算速度模并对齐形状
            rollout_uv = (rollout_denorm[:, 0, :, :] ** 2 + rollout_denorm[:, 1, :, :] ** 2) ** 0.5
            rollout_uv = rollout_uv.unsqueeze(1)  # (T, 1, H, W)

        # 计算逐步指标
        mse_seq, rrmse_seq, ssim_seq = compute_metrics(raw_uv, rollout_uv)
        all_mse.append(mse_seq)
        all_rrmse.append(rrmse_seq)
        all_ssim.append(ssim_seq)

    # -------------------
    # 汇总统计
    # -------------------
    all_mse = np.stack(all_mse)      # [num_starts, prediction_steps]
    all_rrmse = np.stack(all_rrmse)
    all_ssim = np.stack(all_ssim)

    # (1) 整体平均指标（所有步、所有起点的均值）
    overall_mse = np.mean(all_mse)
    overall_rrmse = np.mean(all_rrmse)
    overall_ssim = np.mean(all_ssim)
    overall_std_mse = np.std(all_mse)
    overall_std_rrmse = np.std(all_rrmse)
    overall_std_ssim = np.std(all_ssim)

    print("\n================ FINAL METRICS (Overall across rollouts) ================")
    print(f"MSE   : {overall_mse:.6e} ± {overall_std_mse:.6e}")
    print(f"RRMSE : {overall_rrmse:.6e} ± {overall_std_rrmse:.6e}")
    print(f"SSIM  : {overall_ssim:.6e} ± {overall_std_ssim:.6e}")
    print("=========================================================================")

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
    out_dir = f"../../../../results/{model_name}/figures"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "metrics_cylinder_forward.npz")

    np.savez(
        out_path,
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

    print(f"\nSaved metrics to: {out_path}")
    print("All DMD evaluations completed successfully.")
