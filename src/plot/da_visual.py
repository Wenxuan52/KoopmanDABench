import argparse
import os
import sys
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _default_paths() -> tuple[str, str]:
    """Return default input directory and save directory based on repo layout."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    da_dir = os.path.join(repo_root, "results", "CAE_Koopman", "ERA5", "DA")
    save_dir = os.path.join(repo_root, "src", "temp_results")
    return da_dir, save_dir


def plot_sample_analysis(
    analysis: np.ndarray,
    save_path: str,
    time_indices: Sequence[int] = (0, 24, 49),
    channel_names: Sequence[str] = ("GPH", "Temp", "Humidity", "Wind_u", "Wind_v"),
) -> None:
    """
    Plot selected channels across several time steps from the DA analysis result.

    Args:
        analysis: Array shaped (T, C, H, W).
        save_path: Output path for the figure.
        time_indices: Time steps to visualize.
        channel_names: Names for each channel.
    """
    num_channels = analysis.shape[1]
    time_indices = [idx for idx in time_indices if idx < analysis.shape[0]]
    fig, axes = plt.subplots(
        nrows=num_channels, ncols=len(time_indices), figsize=(4 * len(time_indices), 2 * num_channels)
    )

    if num_channels == 1:
        axes = axes.reshape(1, -1)

    for c in range(num_channels):
        channel_data = analysis[:, c, :, :]
        vmin, vmax = channel_data.min(), channel_data.max()
        for col, t in enumerate(time_indices):
            ax = axes[c, col]
            im = ax.imshow(channel_data[t], origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{channel_names[c] if c < len(channel_names) else f'Ch{c}'} @ t={t}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("DA Analysis Sample", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_mean_std(per_step_mean: np.ndarray, per_step_std: np.ndarray, save_path: str) -> None:
    """Plot mean and std across assimilation steps."""
    steps = np.arange(len(per_step_mean))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, per_step_mean, label="Per-step MSE mean")
    ax.fill_between(steps, per_step_mean - per_step_std, per_step_mean + per_step_std, alpha=0.3, label="±1 std")
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE")
    ax.set_title("Per-step DA Error Statistics")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    default_input_dir, default_save_dir = _default_paths()

    parser = argparse.ArgumentParser(description="Visualize DA results (multi.npy and multi_meanstd.npz).")
    parser.add_argument("--input_dir", type=str, default=default_input_dir, help="Directory containing DA outputs.")
    parser.add_argument("--save_dir", type=str, default=default_save_dir, help="Directory to save figures.")
    parser.add_argument("--multi_file", type=str, default="multi.npy", help="Filename of DA sample array.")
    parser.add_argument("--meanstd_file", type=str, default="multi_meanstd.npz", help="Filename of mean/std stats.")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    multi_path = os.path.join(args.input_dir, args.multi_file)
    stats_path = os.path.join(args.input_dir, args.meanstd_file)

    if not os.path.exists(multi_path):
        raise FileNotFoundError(f"Cannot find multi.npy at {multi_path}")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Cannot find multi_meanstd.npz at {stats_path}")

    analysis = np.load(multi_path)  # Expected shape: (T, C, H, W)
    stats = np.load(stats_path)
    per_step_mean = stats["per_step_mean"]
    per_step_std = stats["per_step_std"]

    sample_fig_path = os.path.join(args.save_dir, "da_sample.png")
    stats_fig_path = os.path.join(args.save_dir, "da_meanstd.png")

    plot_sample_analysis(analysis, sample_fig_path)
    plot_mean_std(per_step_mean, per_step_std, stats_fig_path)

    print(f"Saved sample visualization to: {sample_fig_path}")
    print(f"Saved per-step mean/std visualization to: {stats_fig_path}")


if __name__ == "__main__":
    main()
