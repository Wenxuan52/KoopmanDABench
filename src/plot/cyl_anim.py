import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from datetime import datetime, timedelta
from matplotlib import gridspec

current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", ".."))
sys.path.append(src_directory)

from src.utils.Dataset import CylinderDynamicsDataset


def _to_numpy(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def compute_velocity_magnitude(data):
    """Compute velocity magnitude from u,v components."""
    u = data[:, 0, ...]  # u component
    v = data[:, 1, ...]  # v component
    return np.sqrt(u**2 + v**2)


def compute_row_ranges(datas, time_indices):
    """Return per-channel (vmin, vmax)."""
    vals = []
    for d in datas:
        mag = compute_velocity_magnitude(d[time_indices])
        vals.append(mag)
    vals = np.stack(vals, axis=0)
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    return vmin, vmax


def make_cylinder_gif(
    groundtruth, dmd, cae_dmd, cae_koopman, cae_linear, cae_weaklinear, cae_mlp,
    out_path="figures/cylinder_fields_2x7_with_errors.gif",
    time_indices=None,
    start_timestep=700,
    timesteps_per_frame=1,
    titles=('Groundtruth','DMD','DMD ROM','Koopman ROM','Linear ROM','Weaklinear ROM','MLP ROM'),
    cmap="viridis",
    cmap_err="magma",
    fps=2
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 转换为numpy并准备velocity magnitude
    # groundtruth已经是velocity magnitude格式 [T, 1, H, W]
    if isinstance(groundtruth, torch.Tensor):
        gt = groundtruth.numpy()
    else:
        gt = groundtruth
    
    models = [gt, dmd, cae_dmd, cae_koopman, cae_linear, cae_weaklinear, cae_mlp]

    T = gt.shape[0]
    if time_indices is None:
        time_indices = list(range(min(160, T)))

    # 计算原始值范围（参考您的代码中的方式）
    all_pred_data = np.concatenate([d for d in models], axis=0)
    vmin_pred, vmax_pred = all_pred_data.min(), all_pred_data.max()

    # 计算error范围
    errors = []
    for t in time_indices:
        raw_img = gt[t, 0]  # [H, W]
        for j in range(1, 7):  # 跳过groundtruth
            err = np.abs(models[j][t, 0] - raw_img)
            errors.append(err)
    errors_all = np.stack(errors)
    vmin_err, vmax_err = errors_all.min(), errors_all.max()

    # 网格设置：2行（数据+误差），7列
    total_rows = 2
    ncols = 7
    fig = plt.figure(figsize=(ncols * 4.0, total_rows * 3.0))
    gs = gridspec.GridSpec(
        total_rows, ncols + 2,
        width_ratios=[0.6] + [1]*ncols + [0.08],
        wspace=0.12, hspace=0.15, top=0.82
    )

    BIG, MID = 20, 16

    # 建立axes
    axes_img = [[None for _ in range(ncols)] for __ in range(total_rows)]
    axes_label = []
    axes_cbar = []

    for r in range(total_rows):
        # 左侧标签
        ax_label = fig.add_subplot(gs[r, 0])
        ax_label.axis("off")
        
        if r == 0:
            label_text = "Prediction"
        else:
            label_text = "Error"
            
        ax_label.text(0.5, 0.5, label_text,
                      fontsize=BIG, fontweight="bold",
                      va="center", ha="center", rotation=0)
        axes_label.append(ax_label)

        # 图像axes
        for j in range(ncols):
            ax = fig.add_subplot(gs[r, j+1])
            ax.axis("off")
            axes_img[r][j] = ax

        # colorbar axes
        ax_cbar = fig.add_subplot(gs[r, -1])
        axes_cbar.append(ax_cbar)

    # 顶部标题（只在第一行显示）
    for j in range(ncols):
        axes_img[0][j].set_title(titles[j], fontsize=BIG, fontweight="bold", pad=16)

    # 初始化图像
    images = [[None for _ in range(ncols)] for __ in range(total_rows)]
    t0 = time_indices[0]

    for r in range(total_rows):
        is_error_row = (r == 1)
        
        # 选择colormap和范围
        if not is_error_row:
            vmin, vmax = vmin_pred, vmax_pred
            this_cmap = cmap
        else:
            vmin, vmax = vmin_err, vmax_err
            this_cmap = cmap_err

        for j in range(ncols):
            # error行的第一列（groundtruth）不显示
            if is_error_row and j == 0:
                axes_img[r][j].axis("off")
                images[r][j] = None
                continue

            if not is_error_row:
                # 显示velocity magnitude
                arr = models[j][t0, 0]  # [H, W]
            else:
                # 显示误差
                raw_img = gt[t0, 0]
                arr = np.abs(models[j][t0, 0] - raw_img)

            images[r][j] = axes_img[r][j].imshow(arr, cmap=this_cmap, vmin=vmin, vmax=vmax)

        # 每行的colorbar
        mappable = None
        for j in range(ncols):
            if images[r][j] is not None:
                mappable = images[r][j]
                break
        if mappable is not None:
            cbar = fig.colorbar(mappable, cax=axes_cbar[r])
            cbar.ax.tick_params(labelsize=MID)
        else:
            axes_cbar[r].axis("off")

    # 时间管理
    def frame_timestep(frame_idx):
        return start_timestep + timesteps_per_frame * frame_idx

    suptitle = fig.suptitle(
        f"Cylinder Flow Comparison — Timestep {frame_timestep(0)}",
        fontsize=BIG+4, fontweight="bold", y=0.98
    )

    def update(frame_idx):
        t = time_indices[frame_idx]
        updated = []
        
        for r in range(total_rows):
            is_error_row = (r == 1)
            
            if not is_error_row:
                vmin, vmax = vmin_pred, vmax_pred
            else:
                vmin, vmax = vmin_err, vmax_err

            for j in range(ncols):
                im = images[r][j]
                if im is None:
                    continue
                    
                if not is_error_row:
                    arr = models[j][t, 0]
                else:
                    raw_img = gt[t, 0]
                    arr = np.abs(models[j][t, 0] - raw_img)
                    
                im.set_data(arr)
                im.set_clim(vmin, vmax)
                updated.append(im)

        suptitle.set_text(f"Cylinder Flow Velocity Comparison — Timestep {frame_timestep(frame_idx)}")
        return updated + [suptitle]

    ani = FuncAnimation(fig, update, frames=len(time_indices), interval=1000, blit=False)
    writer = PillowWriter(fps=fps)
    ani.save(out_path, writer=writer)
    plt.close(fig)
    print(f"[INFO] GIF saved to: {out_path}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fig_save_path = "../../results/Comparison/figures/"
    os.makedirs(fig_save_path, exist_ok=True)
    gif_path = os.path.join(fig_save_path, "cylinder_velocity_2x7.gif")

    start_T = 700
    prediction_step = 300
    val_idx = 3
    
    # Load dataset
    print("[INFO] Loading datasets...")
    cyl_train_dataset = CylinderDynamicsDataset(
        data_path="../../data/cylinder/cylinder_train_data.npy",
        seq_length=12,
        mean=None,
        std=None
    )
    
    cyl_val_dataset = CylinderDynamicsDataset(
        data_path="../../data/cylinder/cylinder_val_data.npy",
        seq_length=12,
        mean=cyl_train_dataset.mean,
        std=cyl_train_dataset.std
    )


    raw_test_data = cyl_val_dataset.data
    groundtruth = raw_test_data[val_idx, start_T:start_T + prediction_step, ...]
    groundtruth = torch.tensor(groundtruth, dtype=torch.float32)
    groundtruth_data = (groundtruth[:, 0, :, :] ** 2 + groundtruth[:, 1, :, :] ** 2) ** 0.5
    groundtruth_data = groundtruth_data.unsqueeze(1)

    dmd_data            = np.load('../../results/DMD/figures/cyl_rollout.npy')
    cae_dmd_data        = np.load('../../results/CAE_DMD/figures/cyl_rollout.npy')
    cae_koopman_data    = np.load('../../results/CAE_Koopman/figures/cyl_rollout.npy')
    cae_linear_data     = np.load('../../results/CAE_Linear/figures/cyl_rollout.npy')
    cae_weaklinear_data = np.load('../../results/CAE_Weaklinear/figures/cyl_rollout.npy')
    cae_mlp_data        = np.load('../../results/CAE_MLP/figures/cyl_rollout.npy')

    # 打印各结果数组的shape
    print(f"groundtruth_data shape: {groundtruth_data.shape}")
    print(f"dmd_data shape: {dmd_data.shape}")
    print(f"cae_dmd_data shape: {cae_dmd_data.shape}")
    print(f"cae_koopman_data shape: {cae_koopman_data.shape}")
    print(f"cae_linear_data shape: {cae_linear_data.shape}")
    print(f"cae_weaklinear_data shape: {cae_weaklinear_data.shape}")
    print(f"cae_mlp_data shape: {cae_mlp_data.shape}")

    make_cylinder_gif(
        groundtruth_data, dmd_data, cae_dmd_data, cae_koopman_data,
        cae_linear_data, cae_weaklinear_data, cae_mlp_data,
        out_path=gif_path,
        start_timestep=700,
        timesteps_per_frame=1,
        cmap="viridis",
        fps=10
    )