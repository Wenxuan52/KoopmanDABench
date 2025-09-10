#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Patch

# =======================
# 数据（按你表格录入）——MSE通道
# =======================
dam_overall_mse = {
    "DMD":            {"NoDA": 0.0015, "DA": 0.0014, "Improvement": 0.0563},
    "DMD_ROM":        {"NoDA": 0.0015, "DA": 0.0015, "Improvement": 0.0197},
    "Koopman_ROM":    {"NoDA": 0.0070, "DA": 0.0063, "Improvement": 0.0908},
    "Linear_ROM":     {"NoDA": 0.0015, "DA": 0.0014, "Improvement": 0.1154},
    "Weaklinear_ROM": {"NoDA": 0.0013, "DA": 0.0013, "Improvement": 0.0241},
    "MLP_ROM":        {"NoDA": 0.0108, "DA": 0.0021, "Improvement": 0.8060},
}

dam_background_mse = {
    "DMD":            {"NoDA": 0.0012, "DA": 0.0010, "Improvement": 0.1715},
    "DMD_ROM":        {"NoDA": 0.0012, "DA": 0.0011, "Improvement": 0.0161},
    "Koopman_ROM":    {"NoDA": 0.0021, "DA": 0.0016, "Improvement": 0.2292},
    "Linear_ROM":     {"NoDA": 0.0018, "DA": 0.0011, "Improvement": 0.4082},
    "Weaklinear_ROM": {"NoDA": 0.0012, "DA": 0.0010, "Improvement": 0.1812},
    "MLP_ROM":        {"NoDA": 0.0037, "DA": 0.0013, "Improvement": 0.6547},
}

# 选择绘制哪套数据： "overall" 或 "background"
DATASET = "overall"   # 改成 "background" 可画 t=50 背景帧结果
dam_results = dam_overall_mse if DATASET == "overall" else dam_background_mse

# =======================
# 画图配置（与 ERA5 脚本保持一致）
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
bar_width  = 0.26

plt.rcParams["font.size"] = 14
plt.rcParams["axes.titleweight"] = "bold"

def use_scientific_if_needed(ax):
    sf = ScalarFormatter(useMathText=True)
    sf.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(sf)

def add_percent_labels(ax2, rects):
    for r in rects:
        h = r.get_height()
        if np.isnan(h):
            continue
        ax2.annotate(f"{h:.0f}%", xy=(r.get_x() + r.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=12)

def plot_dam_single_wide(dam_results, dataset_name="overall"):
    # 尽可能宽一些的单图布局
    fig, ax = plt.subplots(figsize=(15, 5))  # 超宽
    ax2 = ax.twinx()

    noda_vals = [dam_results[m]["NoDA"] for m in models]
    da_vals   = [dam_results[m]["DA"] for m in models]
    imp_vals  = [dam_results[m]["Improvement"] * 100.0 for m in models]

    x = np.arange(len(models))
    x_noda = x - bar_width/1.0
    x_da   = x + 0.0
    x_imp  = x + bar_width/1.0

    # 柱状图（带黑色边框）
    rects_noda = ax.bar(
        x_noda, noda_vals, width=bar_width,
        color=[model_colors[m] for m in models], alpha=alpha_noda,
        label="NoDA MSE", edgecolor='black', linewidth=1
    )
    rects_da = ax.bar(
        x_da, da_vals, width=bar_width,
        color=[model_colors[m] for m in models], alpha=alpha_da,
        label="DA MSE", edgecolor='black', linewidth=1
    )
    rects_imp = ax2.bar(
        x_imp, imp_vals, width=bar_width,
        color=[model_colors[m] for m in models], alpha=alpha_imp,
        hatch='///', edgecolor='black', linewidth=1,
        label="Improvement"
    )

    # 轴与标题
    ax.set_title(f"Dam Flow 4D-Var Performance", fontsize=24, pad=16)
    ax.set_ylabel("MSE")
    ax2.set_ylabel("Improvement (%)")
    ax.set_xticks(x)
    # 如需去掉模型名刻度，与 ERA5 风格一致可取消下一行注释：
    # ax.set_xticklabels([])

    # 模型名刻度（如果保留刻度）
    ax.set_xticklabels(models, rotation=0, ha="center")

    use_scientific_if_needed(ax)
    ax.margins(x=0.02)
    ax2.set_ylim(0, max(100, max(imp_vals) * 1.15))

    add_percent_labels(ax2, rects_imp)

    # # ================= 图注放下面 =================
    # # 1) 柱形类型（单色示意）
    # type_handles = [
    #     Patch(facecolor="#666666", alpha=alpha_noda, label="NoDA MSE", edgecolor='black', linewidth=1),
    #     Patch(facecolor="#666666", alpha=alpha_da,   label="DA MSE",   edgecolor='black', linewidth=1),
    #     Patch(facecolor="#666666", alpha=alpha_imp,  hatch='///',  edgecolor='black',
    #           label="Improvement", linewidth=1),
    # ]
    # # 2) 模型颜色
    # model_handles = [Patch(facecolor=model_colors[m], alpha=0.6, label=m) for m in models]

    # # 调整底部留白，叠放两个 legend
    # fig.subplots_adjust(bottom=0.30, left=0.06, right=0.98, top=0.88)

    # # 下方先放“模型颜色”（多列）
    # leg_models = fig.legend(handles=model_handles, loc="lower center",
    #                         bbox_to_anchor=(0.5, 0.10), frameon=False, ncols=6,
    #                         title="Models", title_fontsize=18, fontsize=16)

    # # 再放“柱形类型”
    # leg_types = fig.legend(handles=type_handles, loc="lower center",
    #                        bbox_to_anchor=(0.5, 0.01), frameon=False, ncols=3,
    #                        title="Bar Types", title_fontsize=18, fontsize=16)

    # 输出
    out_name = f"dam_flow_bar.png"
    plt.savefig(out_name, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_name}")

if __name__ == "__main__":
    plot_dam_single_wide(dam_results, dataset_name=DATASET)
