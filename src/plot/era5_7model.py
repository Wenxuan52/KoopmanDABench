import os
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 绘图配置
# ==============================
plt.style.use("seaborn-v0_8-muted")
plt.rcParams["font.size"] = 13
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["figure.dpi"] = 150

# 颜色与通道名称
COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#17becf",
]

CHANNEL_NAMES = ["Geopotential", "Temperature", "Humidity", "Wind_u", "Wind_v"]

METRIC_SPECS = [
    (
        "mse_mean_channel_step",
        "mse_std_channel_step",
        "mse_mean_channel",
        "mse_std_channel",
        "MSE",
        "Mean Squared Error",
    ),
    (
        "rrmse_mean_channel_step",
        "rrmse_std_channel_step",
        "rrmse_mean_channel",
        "rrmse_std_channel",
        "RRMSE",
        "Relative RMSE",
    ),
    (
        "ssim_mean_channel_step",
        "ssim_std_channel_step",
        "ssim_mean_channel",
        "ssim_std_channel",
        "SSIM",
        "Structural Similarity",
    ),
]


def sanitize_filename(name: str) -> str:
    return name.lower().replace(" ", "_")


def plot_era5_rollout_metrics(metric_paths, labels=None, save_dir="../../results/Comparison/figures/ERA5"):
    os.makedirs(save_dir, exist_ok=True)

    if labels is None:
        labels = [os.path.basename(os.path.dirname(p)) for p in metric_paths]

    data_list = [np.load(path) for path in metric_paths]
    num_channels = data_list[0]["mse_mean_channel_step"].shape[0]
    channels = CHANNEL_NAMES[:num_channels]

    # overall summary dict: metric -> channel -> list[(label, mean, std)]
    overall_summary = {spec[4]: {ch: [] for ch in channels} for spec in METRIC_SPECS}

    for ch_idx, ch_name in enumerate(channels):
        fig, axes = plt.subplots(1, len(METRIC_SPECS), figsize=(5 * len(METRIC_SPECS), 4.5), sharex=True)

        for col_idx, (mean_key, std_key, overall_mean_key, overall_std_key, short_name, full_name) in enumerate(METRIC_SPECS):
            ax = axes[col_idx]
            for model_idx, (data, label) in enumerate(zip(data_list, labels)):
                color = COLORS[model_idx % len(COLORS)]
                mean = data[mean_key][ch_idx]
                std = data[std_key][ch_idx]
                steps = np.arange(1, len(mean) + 1)

                ax.plot(steps, mean, label=label, color=color, linewidth=2)
                ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)

                overall_mean = float(data[overall_mean_key][ch_idx])
                overall_std = float(data[overall_std_key][ch_idx])
                overall_summary[short_name][ch_name].append((label, overall_mean, overall_std))

            ax.set_title(full_name, fontsize=13)
            ax.set_xlabel("Rollout Step")
            if col_idx == 0:
                ax.set_ylabel(ch_name)
            ax.grid(True, linestyle="--", alpha=0.4)

        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=min(len(labels), 4),
            frameon=False,
        )
        fig.suptitle(f"{ch_name} Metrics Comparison", fontsize=15)
        fig.tight_layout(rect=[0, 0.05, 1, 0.92])

        filename = f"era5_{sanitize_filename(ch_name)}_metrics_nolinear.png"
        save_path = os.path.join(save_dir, filename)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {save_path}")

    # # Print overall statistics
    # print("\n================ ERA5 OVERALL SUMMARY (mean ± std per channel) ================")
    # for short_name, channel_stats in overall_summary.items():
    #     print(f"\n--- {short_name} ---")
    #     for ch_name in channels:
    #         print(f"{ch_name}:")
    #         for label, mean, std in channel_stats[ch_name]:
    #             if short_name == "SSIM":
    #                 print(f"  {label:<12s}: {mean:.4f} ± {std:.4f}")
    #             else:
    #                 print(f"  {label:<12s}: {mean:.3e} ± {std:.1e}")
    # print("===========================================================================")


if __name__ == "__main__":
    metric_paths = [
        "../../results/DMD/figures/metrics_era5_forward.npz",
        "../../results/KKR/figures/metrics_era5_forward.npz",
        "../../results/KAE/figures/metrics_era5_forward.npz",
        "../../results/WAE/figures/metrics_era5_forward.npz",
        "../../results/AE/figures/metrics_era5_forward.npz",
        "../../results/CGKN/figures/metrics_era5_forward.npz",
        "../../results/DBF/figures/metrics_era5_forward.npz",
    ]

    labels = ["DMD",
    "KoopmanAE",
    "Linear",
    "Weaklinear",
    "MLP", 
    "CGKN", 
    "DBF"]

    plot_era5_rollout_metrics(metric_paths, labels)
