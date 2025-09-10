#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec

# =======================
# 数据
# =======================
era5_results = { 
    "Geopotential": {
        "DMD": {"NoDA": 677464.2336, "Improvement": 0.53, "DA": 321441.8388},
        "DMD_ROM": {"NoDA": 3344710.0640, "Improvement": 0.77, "DA": 760169.4832},
        "Koopman_ROM": {"NoDA": 3355980.7839, "Improvement": 0.07, "DA": 3134611.9097},
        "Linear_ROM": {"NoDA": 1280254.7029, "Improvement": 0.49, "DA": 651035.0307},
        "Weaklinear_ROM": {"NoDA": 1557716.2144, "Improvement": 0.44, "DA": 868969.4510},
        "MLP_ROM": {"NoDA": 1315957.2988, "Improvement": 0.36, "DA": 837921.7157},
    },
    "Temperature": {
        "DMD": {"NoDA": 11.4215, "Improvement": 0.33, "DA": 7.6915},
        "DMD_ROM": {"NoDA": 73.2691, "Improvement": 0.77, "DA": 16.7614},
        "Koopman_ROM": {"NoDA": 95.0483, "Improvement": 0.06, "DA": 89.0598},
        "Linear_ROM": {"NoDA": 24.2716, "Improvement": 0.53, "DA": 11.3596},
        "Weaklinear_ROM": {"NoDA": 20.4803, "Improvement": 0.27, "DA": 15.0006},
        "MLP_ROM": {"NoDA": 23.6935, "Improvement": 0.35, "DA": 15.3893},
    },
    "Humidity": {
        "DMD": {"NoDA": 0.000001, "Improvement": 0.15, "DA": 0.000001},
        "DMD_ROM": {"NoDA": 0.000005, "Improvement": 0.71, "DA": 0.000002},
        "Koopman_ROM": {"NoDA": 0.000006, "Improvement": 0.06, "DA": 0.000006},
        "Linear_ROM": {"NoDA": 0.000002, "Improvement": 0.40, "DA": 0.000001},
        "Weaklinear_ROM": {"NoDA": 0.000003, "Improvement": 0.36, "DA": 0.000002},
        "MLP_ROM": {"NoDA": 0.000003, "Improvement": 0.38, "DA": 0.000002},
    },
    "Wind_u": {
        "DMD": {"NoDA": 25.8978, "Improvement": 0.32, "DA": 17.7011},
        "DMD_ROM": {"NoDA": 48.7167, "Improvement": 0.46, "DA": 26.0931},
        "Koopman_ROM": {"NoDA": 40.2465, "Improvement": 0.04, "DA": 38.7182},
        "Linear_ROM": {"NoDA": 43.6828, "Improvement": 0.39, "DA": 26.7271},
        "Weaklinear_ROM": {"NoDA": 45.3610, "Improvement": 0.39, "DA": 27.4795},
        "MLP_ROM": {"NoDA": 41.2047, "Improvement": 0.27, "DA": 30.0316},
    },
    "Wind_v": {
        "DMD": {"NoDA": 26.6243, "Improvement": 0.32, "DA": 18.0013},
        "DMD_ROM": {"NoDA": 47.0608, "Improvement": 0.37, "DA": 29.4247},
        "Koopman_ROM": {"NoDA": 33.8409, "Improvement": 0.04, "DA": 32.3777},
        "Linear_ROM": {"NoDA": 31.5746, "Improvement": 0.29, "DA": 22.3010},
        "Weaklinear_ROM": {"NoDA": 33.9103, "Improvement": 0.22, "DA": 26.4652},
        "MLP_ROM": {"NoDA": 33.9515, "Improvement": 0.14, "DA": 29.1003},
    },
}

# =======================
# 画图配置
# =======================
models = ["DMD", "DMD_ROM", "Koopman_ROM", "Linear_ROM", "Weaklinear_ROM", "MLP_ROM"]
model_colors = {
    "DMD": "#66c2a5",
    "DMD_ROM": "#fc8d62",
    "Koopman_ROM": "#8da0cb",
    "Linear_ROM": "#e78ac3",
    "Weaklinear_ROM": "#a6d854",
    "MLP_ROM": "#e6c229",
}

alpha_noda = 0.35
alpha_da   = 0.70
alpha_imp  = 0.50

bar_width = 0.26
dpi = 100

plt.rcParams["font.size"] = 16
plt.rcParams["axes.titleweight"] = "bold"

def add_percent_labels(ax2, rects):
    for r in rects:
        h = r.get_height()
        if np.isnan(h):
            continue
        ax2.annotate(f"{h:.0f}%", xy=(r.get_x() + r.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10)

def use_scientific_if_needed(ax):
    sf = ScalarFormatter(useMathText=True)
    sf.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(sf)

def plot_field(ax, field_name):
    results = era5_results[field_name]
    noda_vals = [results[m]["NoDA"] for m in models]
    da_vals   = [results[m]["DA"] for m in models]
    imp_vals_pct = [results[m]["Improvement"] * 100.0 for m in models]

    x = np.arange(len(models))
    x_noda = x - bar_width
    x_da   = x
    x_imp  = x + bar_width

    ax2 = ax.twinx()

    rects_noda = ax.bar(
        x_noda, noda_vals, width=bar_width,
        color=[model_colors[m] for m in models], alpha=alpha_noda,
        label="NoDA", edgecolor='black', linewidth=1
    )
    rects_da = ax.bar(
        x_da, da_vals, width=bar_width,
        color=[model_colors[m] for m in models], alpha=alpha_da,
        label="DA", edgecolor='black', linewidth=1
    )
    rects_imp = ax2.bar(
        x_imp, imp_vals_pct, width=bar_width,
        color=[model_colors[m] for m in models], alpha=alpha_imp,
        hatch='///', edgecolor='black', label="Improvement", linewidth=1
    )

    ax.set_title(f"{field_name}", fontsize=24, pad=14)

    # 需求：取消每个子图 x 轴上的基线模型标注
    ax.set_xticks([])
    ax.set_xlabel("")  # 防止意外空白
    ax.set_ylabel("MSE")
    ax2.set_ylabel("Improvement (%)")

    use_scientific_if_needed(ax)
    ax.margins(x=0.02)
    ax2.set_ylim(0, max(100, max(imp_vals_pct) * 1.15))

    add_percent_labels(ax2, rects_imp)

    return rects_noda, rects_da, rects_imp

def main():
    # 2 行 6 列：上面 3 个子图各占 2 列；下面 2 个子图各占 2 列；右下角 2 列作为“图注面板”
    fig = plt.figure(figsize=(22, 10), dpi=dpi, constrained_layout=False)
    gs = GridSpec(nrows=2, ncols=6, figure=fig, height_ratios=[1.0, 1.0])

    # 第一行三图：0:2, 2:4, 4:6
    ax_geo = fig.add_subplot(gs[0, 0:2])
    ax_tmp = fig.add_subplot(gs[0, 2:4])
    ax_hum = fig.add_subplot(gs[0, 4:6])

    # 第二行两图：0:2, 2:4（居左留白）；右下角 4:6 为图注面板
    ax_uw  = fig.add_subplot(gs[1, 0:2])
    ax_vw  = fig.add_subplot(gs[1, 2:4])
    ax_leg = fig.add_subplot(gs[1, 4:6])

    # 绘制各子图
    plot_field(ax_geo, "Geopotential")
    plot_field(ax_tmp, "Temperature")
    plot_field(ax_hum, "Humidity")
    plot_field(ax_uw,  "Wind_u")
    plot_field(ax_vw,  "Wind_v")

    # ========== 图注面板 ==========
    # 1) 柱形类型图例（NoDA / DA / Improvement）
    type_handles = [
        Patch(facecolor="#666666", alpha=alpha_noda, label="NoDA MSE", edgecolor='black', linewidth=1),
        Patch(facecolor="#666666", alpha=alpha_da,   label="DA MSE",   edgecolor='black', linewidth=1),
        Patch(facecolor="#666666", alpha=alpha_imp,  hatch='///',  edgecolor='black',
            label="Improvement", linewidth=1)
    ]

    # 2) 模型颜色图例（加 alpha）
    model_handles = [Patch(facecolor=c, label=m, alpha=0.6) for m, c in model_colors.items()]

    # 在单独的轴上叠放两个图例
    ax_leg.axis("off")

    # 上半：模型颜色
    leg_models = ax_leg.legend(handles=model_handles, loc="upper left", frameon=False, ncols=2, title="Models", fontsize=18)

    # 下半：柱形类型
    leg_types = ax_leg.legend(handles=type_handles, loc="lower left", frameon=False, ncols=1, title="Bar Types", fontsize=18)

    # 让上面的 legend 保持显示
    ax_leg.add_artist(leg_models)

    # 子图间距
    fig.subplots_adjust(top=0.94, bottom=0.10, left=0.06, right=0.99, wspace=0.95, hspace=0.25)

    fig.suptitle("ERA5 4D-Var Performance Comparison", fontsize=36, fontweight="bold", y=1.08)

    out_name = "era5_all_fields_bars.png"
    plt.savefig(out_name, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_name}")

if __name__ == "__main__":
    main()
