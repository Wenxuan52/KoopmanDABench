import os
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 绘图配置
# ==============================
plt.style.use('seaborn-v0_8-muted')
plt.rcParams['font.size'] = 13
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['figure.dpi'] = 150

# 自定义颜色组（7个模型）
COLORS = [
    "#1f77b4",  # 蓝
    "#ff7f0e",  # 橙
    "#2ca02c",  # 绿
    "#d62728",  # 红
    "#9467bd",  # 紫
    "#8c564b",  # 棕
    "#17becf",  # 青蓝
]


# ==============================
# 核心绘图函数
# ==============================
def plot_rollout_metrics(metric_paths, labels=None, save_dir='../../results/Comparison/figures'):
    os.makedirs(save_dir, exist_ok=True)

    if labels is None:
        labels = [os.path.basename(os.path.dirname(p)) for p in metric_paths]

    metric_keys = [
        ('mse_mean_per_step', 'mse_std_per_step', 'MSE', 'Mean Squared Error'),
        ('rrmse_mean_per_step', 'rrmse_std_per_step', 'RRMSE', 'Relative RMSE'),
        ('ssim_mean_per_step', 'ssim_std_per_step', 'SSIM', 'Structural Similarity'),
    ]

    overall_summary = {k[2]: [] for k in metric_keys}  # 保存overall统计结果

    for mean_key, std_key, short_name, full_name in metric_keys:
        plt.figure(figsize=(6, 4.5))

        for idx, (path, label) in enumerate(zip(metric_paths, labels)):
            color = COLORS[idx % len(COLORS)]
            data = np.load(path)
            mean = data[mean_key]
            std = data[std_key]
            steps = np.arange(1, len(mean) + 1)
            plt.plot(steps, mean, label=label, color=color, linewidth=2)
            plt.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)

            # 保存整体统计信息
            overall_mean = float(data[f"overall_{short_name.lower()}"])
            overall_std = float(data[f"overall_std_{short_name.lower()}"])
            overall_summary[short_name].append((label, overall_mean, overall_std))

        plt.xlabel("Rollout Step", fontsize=13)
        plt.ylabel(full_name, fontsize=13)
        plt.title(f"{short_name} vs Rollout Step", fontsize=14, pad=10)
        plt.legend(frameon=False)
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"rollout_{short_name.lower()}_comparison.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path}")

    # 打印整体指标统计表
    print("\n================ OVERALL SUMMARY (mean ± std) ================")
    for metric_name in ["MSE", "RRMSE", "SSIM"]:
        print(f"\n--- {metric_name} ---")
        for label, mean, std in overall_summary[metric_name]:
            if metric_name == "SSIM":
                print(f"{label:<12s}: {mean:.4f} ± {std:.4f}")
            else:
                print(f"{label:<12s}: {mean:.3e} ± {std:.1e}")
    print("==============================================================")


# ==============================
# 主程序入口
# ==============================
if __name__ == "__main__":
    metric_paths = [
        '../../results/DMD/figures/metrics_cylinder_forward.npz',
        '../../results/CAE_Koopman/figures/metrics_cylinder_forward.npz',
        '../../results/CAE_Linear/figures/metrics_cylinder_forward.npz',
        '../../results/CAE_Weaklinear/figures/metrics_cylinder_forward.npz',
        '../../results/CAE_MLP/figures/metrics_cylinder_forward.npz',
        '../../results/CGKN/figures/metrics_cylinder_forward.npz',
        '../../results/DBF/figures/metrics_cylinder_forward.npz',
    ]

    labels = [
        'DMD',
        'KoopmanAE',
        'Linear',
        'Weaklinear',
        'MLP',
        'CGKN',
        'DBF',
    ]

    plot_rollout_metrics(metric_paths, labels, save_dir='../../results/Comparison/figures')
