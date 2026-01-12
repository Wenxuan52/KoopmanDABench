import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import BoundaryNorm
from matplotlib.animation import FuncAnimation, PillowWriter

# ============================================================
# 1. 加载数据
# ============================================================
groundtruth = np.load("era5_gt.npy")          # (T, C, lat, lon)  (按你的注释)
groundtruth = torch.tensor(groundtruth, dtype=torch.float32)

# 模型顺序（列顺序）
model_order = ["KAE", "KKR", "PFNN", "Ours"]

DateTime = [
    "2018-01-01 04:00",
    "2018-01-01 08:00",
    "2018-01-01 12:00",
    "2018-01-01 16:00",
    "2018-01-01 20:00",
    "2018-01-02 00:00",
    "2018-01-02 04:00",
    "2018-01-02 08:00",
    "2018-01-02 12:00",
    "2018-01-02 16:00",
    "2018-01-02 20:00",
]

# GIF 使用的 paths（所有时间步用同一套结果文件）
result_paths_t_gif = {
    "Ours": "era5_Linear.npy",
    "PFNN": "era5_MLP.npy",
    "KKR": "era5_Koopman.npy",
    "KAE": "era5_MI.npy",
}

def load_pred_dict(paths):
    d = {}
    for label, p in paths.items():
        d[label] = torch.tensor(np.load(p), dtype=torch.float32)
    return d

pred_gif = load_pred_dict(result_paths_t_gif)

# ============================================================
# 2. 颜色范围（主场 per-channel，Error 也 per-channel）
# ============================================================
T_max = 7                     # 时间索引 0~T_max
time_slice = slice(0, T_max + 1)

C = groundtruth.shape[1]       # 通道数（假定至少 5 个）
assert C >= 5, "groundtruth 至少要有 5 个通道."

# ---- 主场：每个通道自己的 vmin/vmax ----
vmin_main = []
vmax_main = []
for ch in range(5):  # 0:Geo,1:Temp,2:Hum,3:Wind_u,4:Wind_v
    vmin_main.append(groundtruth[:, ch].min().item())
    vmax_main.append(groundtruth[:, ch].max().item())

# ---- Error：每个通道自己的 vmin_err/vmax_err ----
# all_err 形状: (num_models, T, C, lat, lon)
all_err = []
for name in model_order:
    err = torch.abs(pred_gif[name] - groundtruth)  # (T, C, lat, lon)
    err = err[time_slice]
    all_err.append(err)
all_err = torch.stack(all_err)                     # (num_models, T, C, lat, lon)

vmin_err = []
vmax_err = []
for ch in range(5):
    vmin_err.append(all_err[:, :, ch].min().item())
    vmax_err.append(all_err[:, :, ch].max().item())

# 经纬度
lon = np.linspace(-180, 180, 64)
lat = np.linspace(-90, 90, 32)
extent = [lon.min(), lon.max(), lat.min(), lat.max()]  # PlateCarree 下的 extent

# 颜色映射
cmap_main = plt.cm.RdBu_r
cmap_err = plt.cm.RdBu_r

# ============================================================
# 3. 子图布局：10 行 × 5 列（行距增大）
# ============================================================
row_labels = [
    "Geopotential", "Error",
    "Temperature", "Error",
    "Humidity", "Error",
    "Wind u direction", "Error",
    "Wind v direction", "Error"
]
col_labels = ["Ground Truth"] + model_order

nrows = 10
ncols = 5

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(22, 26),
    subplot_kw={"projection": ccrs.Robinson()},
    constrained_layout=False,
)

# 增大行间距
plt.subplots_adjust(wspace=0.12, hspace=0.4)

# ============================================================
# 4. 工具函数：row/col 映射 + levels/norm/cmap
# ============================================================
def is_error_row(row):
    return (row % 2 == 1)

def channel_idx_for_row(row):
    # 0:Geo,1:Temp,2:Hum,3:Wind_u,4:Wind_v
    return row // 2

def get_levels_norm_cmap(row, col):
    """
    根据行/列确定 levels, norm, cmap（包括 per-channel error 范围）
    imshow 会用 norm（BoundaryNorm）来实现离散分段着色。
    """
    if col == 0:  # Ground Truth 列
        if is_error_row(row):
            return None, None, None
        ch = channel_idx_for_row(row)
        levels = np.linspace(vmin_main[ch], vmax_main[ch], 30)
        norm = BoundaryNorm(levels, cmap_main.N, clip=True)
        return levels, norm, cmap_main
    else:
        ch = channel_idx_for_row(row)
        if is_error_row(row):  # Error 行
            levels = np.linspace(vmin_err[ch], vmax_err[ch], 20)
            norm = BoundaryNorm(levels, cmap_err.N, clip=True)
            return levels, norm, cmap_err
        else:  # 预测主场
            levels = np.linspace(vmin_main[ch], vmax_main[ch], 30)
            norm = BoundaryNorm(levels, cmap_main.N, clip=True)
            return levels, norm, cmap_main

# ============================================================
# 5. 画某一帧的通用函数（使用 imshow）
# ============================================================
def draw_frame(t_idx, with_colorbar=False):
    """在已有 axes 上画出时间 t_idx 的所有子图。
       with_colorbar=True 时创建 colorbar（只在 init 调用一次）。"""
    fig.suptitle(f"{DateTime[t_idx]}", fontsize=26, fontweight="bold", y=0.93)

    for row in range(nrows):
        for col in range(ncols):
            ax = axes[row, col]
            model_name = col_labels[col]

            # 清空 axes（保留投影类型）
            ax.clear()

            # Ground Truth 列的 Error 行：空白 + “Error” 文本
            if model_name == "Ground Truth" and is_error_row(row):
                ax.axis("off")
                ax.text(
                    -0.2, 0.5, "Error",
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=16, fontweight="bold", rotation=90
                )
                continue

            ax.set_global()
            ax.coastlines(linewidth=0.4)
            ax.gridlines(draw_labels=False, color="gray", linestyle=":", alpha=0.3)

            levels, norm, cmap = get_levels_norm_cmap(row, col)
            if levels is None:
                continue

            ch = channel_idx_for_row(row)

            if model_name == "Ground Truth":
                data = groundtruth[t_idx, ch].cpu().numpy()
            else:
                if is_error_row(row):
                    err = torch.abs(pred_gif[model_name][t_idx, ch] - groundtruth[t_idx, ch])
                    data = err.cpu().numpy()
                else:
                    data = pred_gif[model_name][t_idx, ch].cpu().numpy()

            # 注意：你原先 contourf 用了 data.T，这里保持一致（避免方向翻转）
            plot_data = data.T

            # 用 imshow 绘制
            im = ax.imshow(
                plot_data,
                origin="lower",
                extent=extent,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
            )

            # 行标签（第一列）
            if col == 0:
                ax.text(
                    -0.2, 0.5, row_labels[row],
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=16, fontweight="bold", rotation=90
                )

            # 列标题（第一行）
            if row == 0:
                ax.set_title(model_name, fontsize=20, fontweight="bold", pad=20)

            # 只在 init 时创建 colorbar，后续帧不再重复创建
            if with_colorbar:
                cbar = fig.colorbar(
                    im,
                    ax=ax,
                    orientation="horizontal",
                    fraction=0.05,
                    pad=0.04,
                    aspect=24,
                )
                ticks = np.linspace(levels[0], levels[-1], num=4)
                cbar.set_ticks(ticks)
                cbar.ax.tick_params(labelsize=10)

# ============================================================
# 6. init & update：给 FuncAnimation 用
# ============================================================
def init():
    draw_frame(t_idx=0, with_colorbar=True)
    return []

def update(t_idx):
    draw_frame(t_idx=t_idx, with_colorbar=False)
    return []

# ============================================================
# 7. 创建动画并保存为 GIF
# ============================================================
frames = list(range(0, T_max + 1))
anim = FuncAnimation(
    fig,
    update,
    frames=frames,
    init_func=init,
    blit=False,
    interval=1000,   # 1 秒一帧（交互播放）
    repeat=True,
)

writer = PillowWriter(fps=1.0)
anim.save("era5_comparison.gif", writer=writer)

plt.close(fig)
print("GIF saved to era5_comparison.gif")
