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

from src.utils.Dataset import ERA5Dataset


def _to_numpy(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def compute_row_ranges(datas, time_indices):
    """Return per-channel (vmin, vmax)."""
    vmins, vmaxs = [], []
    for ch in range(5):
        vals = []
        for d in datas:
            vals.append(d[time_indices, ch, ...])
        vals = np.stack(vals, axis=0)
        vmins.append(float(np.min(vals)))
        vmaxs.append(float(np.max(vals)))
    return vmins, vmaxs


def make_era5_gif(
    groundtruth, dmd, cae_dmd, cae_koopman, cae_linear, cae_weaklinear, cae_mlp,
    out_path="figures/era5_fields_10x7_with_errors.gif",
    time_indices=None,
    start_datetime_str="2018-05-05 00:00",
    hours_per_frame=4,
    titles=('Groundtruth','DMD','DMD ROM','Koopman ROM','Linear ROM','Weaklinear ROM','MLP ROM'),
    channel_names=('Geopotential','Temperature','Humidity','Wind_u','Wind_v'),
    cmap="viridis",
    cmap_err="magma"   # error 使用发散色图，更清晰；需要同原图一致可改成 cmap
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # to numpy
    gt = _to_numpy(groundtruth)
    models = [
        gt, _to_numpy(dmd), _to_numpy(cae_dmd), _to_numpy(cae_koopman),
        _to_numpy(cae_linear), _to_numpy(cae_weaklinear), _to_numpy(cae_mlp)
    ]

    T = gt.shape[0]
    if time_indices is None:
        time_indices = list(range(min(10, T)))

    # 原始值范围（按物理场/行统一）
    vmins, vmaxs = compute_row_ranges(models, time_indices)

    # error 范围（与 GT 做差，按物理场/行统一，且对称到0）
    err_maxabs = []
    for ch in range(5):
        diffs = []
        for j in range(1, 7):  # 跳过 groundtruth 本身
            diffs.append(models[j][time_indices, ch, ...] - gt[time_indices, ch, ...])
        diffs = np.stack(diffs, axis=0)  # [6, T, H, W]
        m = float(np.max(np.abs(diffs)))
        err_maxabs.append(m if m > 0 else 1e-6)

    # 网格：10 行（5 行数据 + 5 行 error），7 列 + 左标签 + 右侧 colorbar
    data_rows = 5
    total_rows = data_rows * 2
    ncols = 7
    fig = plt.figure(figsize=(ncols * 4.0 + 4.0, total_rows * 2.0))
    # 调大左右间距 wspace；上下间距 hspace 轻微增大
    gs = gridspec.GridSpec(
        total_rows, ncols + 2,
        width_ratios=[0.6] + [1]*ncols + [0.08],
        wspace=0.12, hspace=0.10
    )

    BIG, MID = 24, 18

    # 建立 axes
    axes_img = [[None for _ in range(ncols)] for __ in range(total_rows)]
    axes_label = []
    axes_cbar = []

    for r in range(total_rows):
        ax_label = fig.add_subplot(gs[r, 0])
        ax_label.axis("off")

        ch = r // 2
        is_error_row = (r % 2 == 1)
        label_text = channel_names[ch] + ("" if not is_error_row else "  Error")
        ax_label.text(0.5, 0.5, label_text,
                      fontsize=BIG, fontweight="bold",
                      va="center", ha="center", rotation=0)
        axes_label.append(ax_label)

        for j in range(ncols):
            ax = fig.add_subplot(gs[r, j+1])
            ax.axis("off")
            axes_img[r][j] = ax

        ax_cbar = fig.add_subplot(gs[r, -1])
        axes_cbar.append(ax_cbar)

    # 顶部列标题（只在最上面一行显示）
    for j in range(ncols):
        axes_img[0][j].set_title(titles[j], fontsize=BIG, fontweight="bold", pad=16)

    # 初始化图像
    images = [[None for _ in range(ncols)] for __ in range(total_rows)]
    t0 = time_indices[0]

    for r in range(total_rows):
        ch = r // 2
        is_error_row = (r % 2 == 1)

        # 选择色图与范围
        if not is_error_row:
            vmin, vmax = vmins[ch], vmaxs[ch]
            this_cmap = cmap
        else:
            m = err_maxabs[ch]
            vmin, vmax = 0, m
            this_cmap = cmap_err

        # 每行逐列放图
        for j in range(ncols):
            if is_error_row and j == 0:
                # Groundtruth 的 error 不绘制
                axes_img[r][j].axis("off")
                images[r][j] = None
                continue

            if not is_error_row:
                arr = models[j][t0, ch, ...].T
            else:
                arr = np.abs(models[j][t0, ch, ...] - gt[t0, ch, ...]).T

            images[r][j] = axes_img[r][j].imshow(arr, cmap=this_cmap, vmin=vmin, vmax=vmax)

        # 每行 colorbar：选一个已存在的图像作为 mappable
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
    start_dt = datetime.strptime(start_datetime_str, "%Y-%m-%d %H:%M")
    def frame_datetime(frame_idx):
        return start_dt + timedelta(hours=hours_per_frame * frame_idx)

    suptitle = fig.suptitle(
        f"ERA5 Fields Comparison — {frame_datetime(0).strftime('%Y-%m-%d %H:%M')}",
        fontsize=BIG+6, fontweight="bold", y=0.95
    )

    def update(frame_idx):
        t = time_indices[frame_idx]
        updated = []
        for r in range(total_rows):
            ch = r // 2
            is_error_row = (r % 2 == 1)

            if not is_error_row:
                vmin, vmax = vmins[ch], vmaxs[ch]
            else:
                m = err_maxabs[ch]
                vmin, vmax = 0, m   # ← 这里改为从0开始

            for j in range(ncols):
                im = images[r][j]
                if im is None:
                    continue
                if not is_error_row:
                    arr = models[j][t, ch, ...].T
                else:
                    arr = np.abs(models[j][t, ch, ...] - gt[t, ch, ...]).T  # ← 这里改为绝对值
                im.set_data(arr)
                im.set_clim(vmin, vmax)
                updated.append(im)

        suptitle.set_text(f"ERA5 Fields Comparison — {frame_datetime(frame_idx).strftime('%Y-%m-%d %H:%M')}")
        return updated + [suptitle]

    ani = FuncAnimation(fig, update, frames=len(time_indices), interval=1000, blit=False)
    writer = PillowWriter(fps=1)
    ani.save(out_path, writer=writer)
    plt.close(fig)
    print(f"[INFO] GIF saved to: {out_path}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fig_save_path = "../../results/Comparison/figures/"
    os.makedirs(fig_save_path, exist_ok=True)
    gif_path = os.path.join(fig_save_path, "era5_all_fields_5x7.gif")

    start_T, prediction_step = 1000, 100
    era5_test_dataset = ERA5Dataset(
        data_path="../../data/ERA5/ERA5_data/test_seq_state.h5",
        seq_length=12,
        min_path="../../data/ERA5/ERA5_data/min_val.npy",
        max_path="../../data/ERA5/ERA5_data/max_val.npy"
    )

    raw_test_data = era5_test_dataset.data
    groundtruth = raw_test_data[start_T:start_T+prediction_step, ...]
    groundtruth = torch.tensor(groundtruth, dtype=torch.float32).permute(0,3,1,2)

    dmd_data            = np.load('../../results/DMD/figures/era5_dmd_rollout.npy')
    cae_dmd_data        = np.load('../../results/CAE_DMD/figures/era5_rollout.npy')
    cae_koopman_data    = np.load('../../results/CAE_Koopman/figures/era5_rollout.npy')
    cae_linear_data     = np.load('../../results/CAE_Linear/figures/era5_rollout.npy')
    cae_weaklinear_data = np.load('../../results/CAE_Weaklinear/figures/era5_rollout.npy')
    cae_mlp_data        = np.load('../../results/CAE_MLP/figures/era5_rollout.npy')

    time_indices = list(range(15))
    make_era5_gif(
        groundtruth, dmd_data, cae_dmd_data, cae_koopman_data,
        cae_linear_data, cae_weaklinear_data, cae_mlp_data,
        out_path=gif_path,
        time_indices=time_indices,
        start_datetime_str="2018-05-05 00:00",
        hours_per_frame=4,
        cmap="viridis"
    )
