import torch
import numpy as np
import random
import os
import sys
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from torchdiffeq import odeint

# ===========================
# 路径注册
# ===========================
current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
sys.path.append(src_directory)

# ------------------- CGKN 模块导入 -------------------
from src.utils.Dataset import CylinderDynamicsDataset
from cylinder_model import CylinderEncoder, CylinderDecoder
from cylinder_trainer import CGN, CGKN_ODE, ProbeSampler, cfg


# ===========================
# 工具函数
# ===========================
def compute_metrics(gt, pred):
    """
    输入: gt, pred 形状 (T, 1, H, W)
    输出: 每帧 MSE, RRMSE, SSIM
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


def velocity_magnitude(field):
    """输入 (T, 2, H, W)，输出 (T, 1, H, W)"""
    u = field[:, 0]
    v = field[:, 1]
    mag = torch.sqrt(torch.clamp(u ** 2 + v ** 2, min=0.0))
    return mag.unsqueeze(1)


def rollout_prediction(encoder, decoder, ode_func, probe_sampler, normalized_seq, dt):
    """
    执行CGKN rollout:
    编码 -> 构造u_ext0 -> ODE积分 -> 解码
    返回 (T, C, H, W)
    """
    device = normalized_seq.device
    seq = normalized_seq.unsqueeze(0)  # [1, T, C, H, W]

    with torch.no_grad():
        v_seq = encoder(seq)            # [1, T, dim_z]
        u1_seq = probe_sampler.sample(seq)  # [1, T, dim_u1]
        uext0 = torch.cat([u1_seq[:, 0, :], v_seq[:, 0, :]], dim=-1)  # [1, dim_u1+dim_z]

        horizon = normalized_seq.shape[0]
        tspan = torch.linspace(0.0, (horizon - 1) * dt, horizon, device=device)
        uext_pred = odeint(ode_func, uext0, tspan, method="rk4", options={"step_size": dt}).transpose(0, 1)
        dim_u1 = u1_seq.shape[-1]
        v_rollout = uext_pred[:, :, dim_u1:]
        rollout = decoder(v_rollout).squeeze(0)
    return rollout  # (T, C, H, W)


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
    forward_step = cfg.seq_length
    seed = 42
    model_name = 'CGKN'

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    denorm = cyl_val_dataset.denormalizer()

    # -------------------
    # Probe Sampler
    # -------------------
    checkpoint_dir = "../../../../results/CGKN/Cylinder"
    probe_file = f"{checkpoint_dir}/probe_coords.npy"
    coords = np.load(probe_file)
    coords = [tuple(map(int, entry)) for entry in coords.tolist()]
    channels = list(range(2))  # 两个速度通道
    probe_sampler = ProbeSampler(coords, channels)
    dim_u1 = probe_sampler.dim_u1  # 自动确定输入观测维度

    # -------------------
    # 模型加载
    # -------------------
    encoder = CylinderEncoder(dim_z=cfg.dim_z).to(device)
    decoder = CylinderDecoder(dim_z=cfg.dim_z).to(device)
    cgn = CGN(dim_u1=dim_u1, dim_z=cfg.dim_z, hidden=cfg.hidden).to(device)

    encoder.load_state_dict(torch.load(f"{checkpoint_dir}/stage2_encoder.pt", map_location=device, weights_only=True))
    decoder.load_state_dict(torch.load(f"{checkpoint_dir}/stage2_decoder.pt", map_location=device, weights_only=True))
    cgn.load_state_dict(torch.load(f"{checkpoint_dir}/stage2_cgn.pt", map_location=device, weights_only=True))

    encoder.eval()
    decoder.eval()
    cgn.eval()

    ode_func = CGKN_ODE(cgn).to(device)
    ode_func.eval()

    # -------------------
    # Probe Sampler
    # -------------------
    probe_file = f"{checkpoint_dir}/probe_coords.npy"
    coords = np.load(probe_file)
    coords = [tuple(map(int, entry)) for entry in coords.tolist()]
    channels = list(range(2))  # 2 velocity components
    probe_sampler = ProbeSampler(coords, channels)

    # -------------------
    # 起点采样
    # -------------------
    start_frames = random.sample(range(start_min, start_max), num_starts)
    print(f"\nSelected start frames ({num_starts} total): {start_frames}")

    all_mse, all_rrmse, all_ssim = [], [], []

    # -------------------
    # rollout + 评估
    # -------------------
    for start_frame in tqdm(start_frames, desc="Evaluating CGKN rollouts"):
        # Ground truth
        groundtruth = torch.tensor(
            cyl_val_dataset.data[val_idx, start_frame + 1:start_frame + 1 + prediction_steps, ...],
            dtype=torch.float32, device=device
        )
        raw_uv = velocity_magnitude(groundtruth)

        # 初始帧 (归一化)
        initial_seq = torch.tensor(
            cyl_val_dataset.data[val_idx, start_frame:start_frame + prediction_steps, ...],
            dtype=torch.float32, device=device
        )
        mean = cyl_val_dataset.mean.view(1, -1, 1, 1).to(device)
        std = cyl_val_dataset.std.view(1, -1, 1, 1).to(device)
        normalized_seq = (initial_seq - mean) / std

        # rollout 预测
        rollout_norm = rollout_prediction(encoder, decoder, ode_func, probe_sampler, normalized_seq, cfg.dt)
        rollout_field = rollout_norm * std + mean
        rollout_uv = velocity_magnitude(rollout_field)

        # 计算指标
        mse_seq, rrmse_seq, ssim_seq = compute_metrics(raw_uv, rollout_uv)
        all_mse.append(mse_seq)
        all_rrmse.append(rrmse_seq)
        all_ssim.append(ssim_seq)

    # -------------------
    # 汇总统计
    # -------------------
    all_mse = np.stack(all_mse)
    all_rrmse = np.stack(all_rrmse)
    all_ssim = np.stack(all_ssim)

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
    np.savez(
        f"{out_dir}/metrics_cylinder_forward.npz",
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
    print("All CGKN evaluations completed successfully.")
