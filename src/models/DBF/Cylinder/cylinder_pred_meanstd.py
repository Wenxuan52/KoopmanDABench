import torch
import numpy as np
import random
import os
import sys
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

# ---------------- 路径注册 ----------------
current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
sys.path.append(src_directory)

from src.utils.Dataset import CylinderDynamicsDataset
from src.models.DBF.Cylinder.cylinder_model import CylinderDecoder
from src.models.DBF.Cylinder.cylinder_trainer import (
    ObservationMask,
    CylinderIOONetwork,
    SpectralKoopmanOperator,
)
from src.models.DBF.Cylinder.cylinder_predictor import (
    initial_latent_state,
    gaussian_update,
    decode_pairs_sequence,
)


# ===========================
# Metric utilities
# ===========================
def compute_metrics(gt, pred):
    """
    输入 (T, 1, H, W)，输出每帧 MSE, RRMSE, SSIM。
    """
    gt_np = gt.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    mse_list, rrmse_list, ssim_list = [], [], []
    for t in range(gt_np.shape[0]):
        g = gt_np[t, 0]
        p = pred_np[t, 0]
        mse = np.mean((p - g) ** 2)
        rrmse = np.sqrt(mse) / (np.sqrt(np.mean(g ** 2)) + 1e-8)
        ssim_val = ssim(g, p, data_range=g.max() - g.min())
        mse_list.append(mse)
        rrmse_list.append(rrmse)
        ssim_list.append(ssim_val)
    return np.array(mse_list), np.array(rrmse_list), np.array(ssim_list)


def velocity_magnitude(field: torch.Tensor) -> torch.Tensor:
    """(T, 2, H, W) -> (T, 1, H, W)"""
    u = field[:, 0]
    v = field[:, 1]
    mag = torch.sqrt(torch.clamp(u ** 2 + v ** 2, min=0.0))
    return mag.unsqueeze(1)


@torch.no_grad()
def run_dbf_rollout(
    sequence: torch.Tensor,
    observation_mask: ObservationMask,
    ioo: CylinderIOONetwork,
    koopman: SpectralKoopmanOperator,
    decoder: CylinderDecoder,
    init_cov: float,
    cov_epsilon: float,
):
    """
    单条 rollout：仅使用 Koopman 预测（不进行DA）
    返回 rollout reconstructions (T, C, H, W)
    """
    device = next(decoder.parameters()).device
    seq = sequence.unsqueeze(0).to(device)
    obs_all = observation_mask.sample(seq)

    batch_size, time_steps = seq.shape[0], seq.shape[1]
    num_pairs = koopman.num_pairs

    mu_prev, cov_prev = initial_latent_state(batch_size, num_pairs, device, init_cov)
    # 初始 assimilation
    mu_obs_flat, sigma2_obs_flat = ioo(obs_all[:, 0, :])
    mu_obs_pairs = mu_obs_flat.view(batch_size, num_pairs, 2)
    cov_obs = torch.diag_embed(sigma2_obs_flat.view(batch_size, num_pairs, 2))
    mu_first, cov_first = gaussian_update(mu_prev, cov_prev, mu_obs_pairs, cov_obs, cov_epsilon)

    rollout_pairs = [mu_first]
    mu_roll_prev, cov_roll_prev = mu_first, cov_first

    # rollout
    for t in range(1, time_steps):
        mu_prior_roll, cov_prior_roll = koopman.predict(mu_roll_prev, cov_roll_prev)
        mu_roll_post, cov_roll_post = mu_prior_roll, cov_prior_roll
        rollout_pairs.append(mu_roll_post)
        mu_roll_prev, cov_roll_prev = mu_roll_post, cov_roll_post

    rollout_pairs = torch.stack(rollout_pairs, dim=1)
    rollout_frames = decode_pairs_sequence(decoder, rollout_pairs)
    return rollout_frames.squeeze(0)


# ===========================
# 主流程
# ===========================
if __name__ == "__main__":
    # -------------------
    # 参数设置
    # -------------------
    prediction_steps = 30
    num_starts = 50
    start_min, start_max = 700, 950
    val_idx = 3
    seed = 42
    model_name = "DBF"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # -------------------
    # 加载 checkpoint
    # -------------------
    ckpt_path = "../../../../results/DBF/Cylinder/best_model.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    cfg = checkpoint.get("config", {})
    latent_dim = int(cfg.get("latent_dim", 512))
    seq_length = int(cfg.get("seq_length", 64))
    data_path = cfg.get("val_data", "../../../../data/cylinder/cylinder_val_data.npy")
    init_cov = float(cfg.get("init_cov", 1.0))
    cov_epsilon = float(cfg.get("cov_epsilon", 1e-6))

    # -------------------
    # 数据集加载
    # -------------------
    cyl_train_dataset = CylinderDynamicsDataset(
        data_path="../../../../data/cylinder/cylinder_train_data.npy",
        seq_length=seq_length,
        mean=None,
        std=None,
    )
    cyl_val_dataset = CylinderDynamicsDataset(
        data_path=data_path,
        seq_length=seq_length,
        mean=checkpoint["train_mean"],
        std=checkpoint["train_std"],
    )
    denorm = cyl_val_dataset.denormalizer()

    # -------------------
    # 模型加载
    # -------------------
    observation_mask = ObservationMask(checkpoint["mask"].bool()).to(device)

    decoder = CylinderDecoder(dim_z=latent_dim).to(device)
    decoder.load_state_dict(checkpoint["decoder"])
    decoder.eval()

    ioo = CylinderIOONetwork(
        obs_dim=observation_mask.obs_dim,
        latent_dim=latent_dim,
        hidden_dims=cfg.get("ioo_hidden_dims", [1024, 1024]),
    ).to(device)
    ioo.load_state_dict(checkpoint["ioo"])
    ioo.eval()

    koopman = SpectralKoopmanOperator(
        latent_dim=latent_dim,
        rho_clip=float(cfg.get("rho_clip", 0.2)),
        process_noise_init=float(cfg.get("process_noise_init", -2.0)),
    ).to(device)
    koopman.load_state_dict(checkpoint["koopman"])
    koopman.eval()

    # -------------------
    # 起点采样
    # -------------------
    start_frames = random.sample(range(start_min, start_max), num_starts)
    print(f"\nSelected start frames ({num_starts} total): {start_frames}")

    all_mse, all_rrmse, all_ssim = [], [], []

    # -------------------
    # rollout + 评估
    # -------------------
    for start_frame in tqdm(start_frames, desc="Evaluating DBF rollouts"):
        groundtruth = torch.tensor(
            cyl_val_dataset.data[val_idx, start_frame + 1:start_frame + 1 + prediction_steps, ...],
            dtype=torch.float32,
            device=device,
        )
        raw_uv = velocity_magnitude(groundtruth)

        sequence = torch.tensor(
            cyl_val_dataset.data[val_idx, start_frame:start_frame + prediction_steps, ...],
            dtype=torch.float32,
            device=device,
        )
        normalized_seq = cyl_val_dataset.normalize(sequence)

        rollout_norm = run_dbf_rollout(
            normalized_seq, observation_mask, ioo, koopman, decoder, init_cov, cov_epsilon
        )
        rollout_field = denorm(rollout_norm)
        rollout_uv = velocity_magnitude(rollout_field)

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

    print("\n================ FINAL METRICS (Overall across rollouts) ================")
    print(f"MSE   : {overall_mse:.6e} ± {overall_std_mse:.6e}")
    print(f"RRMSE : {overall_rrmse:.6e} ± {overall_std_rrmse:.6e}")
    print(f"SSIM  : {overall_ssim:.6e} ± {overall_std_ssim:.6e}")
    print("=========================================================================")

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
    out_dir = "../../../../results/DBF/figures"
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
        start_frames=np.array(start_frames),
    )

    print(f"\nSaved metrics to: {out_dir}/metrics_cylinder_forward.npz")
    print("All DBF evaluations completed successfully.")
