import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import BoundaryNorm

# ============================================================
# 1. 加载数据
# ============================================================
current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", ".."))
sys.path.append(src_directory)

from src.utils.Dataset import ERA5Dataset

era5_test_dataset = ERA5Dataset(
    data_path="../../data/ERA5/ERA5_data/test_seq_state.h5",
    seq_length=12,
    min_path="../../data/ERA5/ERA5_data/min_val.npy",
    max_path="../../data/ERA5/ERA5_data/max_val.npy",
)
raw_test_data = era5_test_dataset.data  # (N, H, W, C)

prediction_steps = 50
visual_channel = 2

groundtruth = (
    torch.tensor(raw_test_data[1 : 1 + prediction_steps, ...], dtype=torch.float32)
    .permute(0, 3, 1, 2)
)

# 时间步索引
time_indices = [4, 46]  # 对应两个时间点

# 模型顺序（列顺序）
model_order = ["KAE", "KKR", "PFNN", "Ours"]

# 两个时间点对应的 result_paths
result_paths_t1 = {
    "Ours": "../../results/CAE_Linear/figures/era5_ICLR.npy",
    "PFNN": "../../results/CAE_Koopman/figures/era5_ICLR.npy",
    "KKR": "../../results/CAE_MLP/figures/era5_ICLR.npy",
    "KAE": "../../results/CAE_MI/figures/era5_ICLR.npy",
}

result_paths_t2 = {
    "Ours": "../../results/CAE_MI/figures/era5_ICLR.npy",
    "PFNN": "../../results/CAE_MLP/figures/era5_ICLR.npy",
    "KKR": "../../results/CAE_Koopman/figures/era5_ICLR.npy",
    "KAE": "../../results/CAE_Linear/figures/era5_ICLR.npy",
}

# ============================================================
# 2. 构建 pred_dict_t1, pred_dict_t2
# ============================================================
def load_pred_dict(paths):
    d = {}
    for label, p in paths.items():
        d[label] = torch.tensor(np.load(p), dtype=torch.float32)
    return d

pred_t1 = load_pred_dict(result_paths_t1)
pred_t2 = load_pred_dict(result_paths_t2)

# 计算误差范围（用于统一 error 色标）
all_errors = []
for pred_dict in [pred_t1, pred_t2]:
    for name in model_order:
        err = torch.abs(pred_dict[name] - groundtruth)
        all_errors.append(err[:, visual_channel])
vmax_err = torch.max(torch.stack(all_errors)).item()
vmin_err = 0.0

# ============================================================
# 3. 图布局
# ============================================================
row_labels = [
    "2018-01-02\n00:00", "Error",
    "2018-01-09\n00:00", "Error"
]

col_labels = ["Ground Truth"] + model_order

nrows = 4
ncols = 5

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(22, 16),
    subplot_kw={"projection": ccrs.Robinson()},
    constrained_layout=False,
)
plt.subplots_adjust(wspace=0.12, hspace=0.14)

# 主图颜色范围（统一）
cmap_main = plt.cm.RdBu_r
vmin_main = groundtruth[:, visual_channel].min().item()
vmax_main = groundtruth[:, visual_channel].max().item()

# error 图颜色范围（统一）
cmap_err = plt.cm.RdBu_r

# 数据网格范围（用于 imshow 的 extent）
lon = np.linspace(-180, 180, 64)
lat = np.linspace(-90, 90, 32)
extent = [lon[0], lon[-1], lat[0], lat[-1]]

# ============================================================
# 4. 逐格绘图（方案 B：imshow + nearest）
# ============================================================
for row in range(nrows):
    is_error_row = (row % 2 == 1)
    time_idx = time_indices[row // 2]
    time_str = row_labels[row]

    pred_dict = pred_t1 if row < 2 else pred_t2

    for col in range(ncols):
        ax = axes[row, col]
        ax.coastlines(linewidth=0.4)
        ax.gridlines(draw_labels=False, color="gray", linestyle=":", alpha=0.3)

        model_name = col_labels[col]

        # Ground Truth 列：Error 行保持空白
        if model_name == "Ground Truth":
            if is_error_row:
                fig.delaxes(ax)
                blank_ax = fig.add_subplot(nrows, ncols, row * ncols + col + 1)
                blank_ax.axis("off")
                continue

            data = groundtruth[time_idx, visual_channel].cpu().numpy()
            levels = np.linspace(vmin_main, vmax_main, 30)
            norm = BoundaryNorm(levels, cmap_main.N)
            cmap = cmap_main

        else:
            if is_error_row:
                err = torch.abs(
                    pred_dict[model_name][time_idx, visual_channel]
                    - groundtruth[time_idx, visual_channel]
                )
                data = err.cpu().numpy()
                levels = np.linspace(vmin_err, vmax_err, 20)
                norm = BoundaryNorm(levels, cmap_err.N)
                cmap = cmap_err
            else:
                data = pred_dict[model_name][time_idx, visual_channel].cpu().numpy()
                levels = np.linspace(vmin_main, vmax_main, 30)
                norm = BoundaryNorm(levels, cmap_main.N)
                cmap = cmap_main

        # --- 方案 B：像素绘制（imshow + nearest） ---
        img = ax.imshow(
            data.T,  # 保持你原来的转置逻辑
            origin="lower",
            extent=extent,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            interpolation="nearest",  # 关键：像素块
            rasterized=True,          # PDF 里保持像素块且文件小
        )

        # 行标签
        if col == 0:
            if not is_error_row:
                ax.text(
                    -0.2, 0.5, time_str,
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=18, fontweight="bold",
                    rotation=90,
                )
            else:
                ax.text(
                    -0.2, 0.5, "Error",
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=18, fontweight="bold",
                    rotation=90,
                )

        # 列标题
        if row == 0:
            ax.set_title(model_name, fontsize=22, fontweight="bold", pad=20)

        # colorbar（用 img 本身，保证和显示一致）
        cbar = fig.colorbar(
            img,
            ax=ax,
            orientation="horizontal",
            fraction=0.035,
            pad=0.035,
            aspect=18,
            extend="both",
        )
        ticks = np.linspace(levels[0], levels[-1], num=4)
        cbar.set_ticks(ticks)
        cbar.ax.tick_params(labelsize=11)

# ============================================================
# 5. 保存
# ============================================================
plt.savefig("era5_final_comparison.pdf", dpi=100, bbox_inches="tight")
