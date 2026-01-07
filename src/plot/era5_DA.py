"""Aggregate visualization of ERA5 DA metrics across models.

This script loads ``multi_meanstd.npz`` files produced by each model's
assimilation experiment and renders a 5x3 grid comparing per-channel
MSE/RRMSE/SSIM across models.

Update:
- Add SAVE_PREFIX so you can switch among different experiment folders/files.
  Example: SAVE_PREFIX = "fullobs_direct_era5_"
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Ensure matplotlib can write cache files in restricted environments
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_config")

# =========================
# Experiment prefix switch
# =========================
# Example: "fullobs_direct_era5_"
# Set "" to use the original paths with no prefix.
SAVE_PREFIX = "fullobs_direct_era5_"

# Channel and metric labels
CHANNEL_LABELS: List[str] = [
    "Geopotential",
    "Temperature",
    "Humidity",
    "Wind U-direction",
    "Wind V-direction",
]
METRIC_KEYS: List[Tuple[str, str]] = [
    ("mse", "MSE"),
    ("rrmse", "RRMSE"),
    ("ssim", "SSIM"),
]

# Models to compare (name, relative results path from repo root)
# We will inject SAVE_PREFIX before "multi_meanstd.npz".
MODEL_SPECS: List[Tuple[str, Path]] = [
    ("CAE_Koopman", Path("results/CAE_Koopman/ERA5/DA/multi_meanstd.npz")),
    ("CAE_Linear", Path("results/CAE_Linear/ERA5/DA/multi_meanstd.npz")),
    ("CAE_Weaklinear", Path("results/CAE_Weaklinear/ERA5/DA/multi_meanstd.npz")),
    ("CAE_MLP", Path("results/CAE_MLP/ERA5/DA/multi_meanstd.npz")),
    ("DMD", Path("results/DMD/ERA5/DA/multi_meanstd.npz")),
    ("CGKN", Path("results/CGKN/ERA5/DA/multi_meanstd.npz")),
    ("DBF", Path("results/DBF/ERA5/DA/multi_meanstd.npz")),
]

# Distinct colors for models
MODEL_COLORS: List[str] = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#17becf",
]


def with_prefix(npz_path: Path, prefix: str) -> Path:
    """Insert prefix before the filename.

    Example:
      results/.../DA/multi_meanstd.npz
    -> results/.../DA/{prefix}multi_meanstd.npz
    """
    if not prefix:
        return npz_path
    return npz_path.with_name(f"{prefix}{npz_path.name}")


def load_model_metrics(repo_root: Path, prefix: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Load aggregated metrics for each model.

    Returns a mapping: model_name -> metric_key -> ndarray with shape
    (steps, channels, 2). Only DA values (index 0 on the last axis) are
    used in the visualization.
    """
    model_data: Dict[str, Dict[str, np.ndarray]] = {}
    for (name, rel_path), color in zip(MODEL_SPECS, MODEL_COLORS):
        base_path = repo_root / rel_path
        npz_path = with_prefix(base_path, prefix)

        if not npz_path.exists():
            # Helpful fallback: if prefixed doesn't exist, try unprefixed
            fallback = base_path
            if fallback.exists():
                print(f"[WARN] Missing prefixed metrics for {name}: {npz_path} (fallback to {fallback})")
                npz_path = fallback
            else:
                print(f"[WARN] Missing metrics for {name}: {npz_path} (and fallback {fallback} not found)")
                continue

        data = np.load(npz_path)
        model_data[name] = {"color": color}
        for key, _ in METRIC_KEYS:
            model_data[name][key] = data[f"{key}_mean"]
            model_data[name][f"{key}_std"] = data[f"{key}_std"]
        print(f"Loaded metrics for {name} from {npz_path}")
    return model_data


def aggregate_da_stats(mean_arr: np.ndarray, std_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute step-averaged DA mean and std per channel."""
    mean_steps = mean_arr[:, :, 0]  # (steps, channels)
    std_steps = std_arr[:, :, 0]  # (steps, channels)
    da_mean = mean_steps.mean(axis=0)
    da_std = std_steps.mean(axis=0)
    return da_mean, da_std


def build_comparison_figure(model_data: Dict[str, Dict[str, np.ndarray]], output_path: Path) -> None:
    fig, axes = plt.subplots(
        nrows=len(CHANNEL_LABELS),
        ncols=len(METRIC_KEYS),
        figsize=(18, 14),
        sharex=True,
    )

    model_names = list(model_data.keys())
    handles = []

    for row, channel_label in enumerate(CHANNEL_LABELS):
        for col, (metric_key, metric_label) in enumerate(METRIC_KEYS):
            ax = axes[row, col]
            for name in model_names:
                mean_arr = model_data[name][metric_key][:, row, 0]
                std_arr = model_data[name][f"{metric_key}_std"][:, row, 0]
                steps = np.arange(1, len(mean_arr) + 1)
                (line,) = ax.plot(steps, mean_arr, color=model_data[name]["color"], label=name)
                ax.fill_between(
                    steps,
                    mean_arr - std_arr,
                    mean_arr + std_arr,
                    color=model_data[name]["color"],
                    alpha=0.15,
                )
                if row == 0 and col == 0:
                    handles.append(line)

            ax.set_title(metric_label if row == 0 else "")
            if col == 0:
                ax.set_ylabel(channel_label)
            if row == len(CHANNEL_LABELS) - 1:
                ax.set_xlabel("Assimilation step")
            ax.grid(True, linestyle="--", alpha=0.4)

    # Unified legend at the very bottom
    legend_fontsize = 13
    fig.legend(
        handles,
        model_names,
        loc="lower center",
        ncol=min(len(model_names), 7),
        bbox_to_anchor=(0.5, 0.01),
        frameon=False,
        fontsize=legend_fontsize,
        handlelength=2.5,
        columnspacing=1.2,
        handletextpad=0.6,
    )

    # Leave space at bottom for legend
    fig.tight_layout(rect=[0, 0.06, 1, 1.0])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved comparison figure to {output_path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    model_data = load_model_metrics(repo_root, prefix=SAVE_PREFIX)
    if not model_data:
        print("No metrics found; aborting plot generation.")
        return

    figures_dir = repo_root / "results" / "Comparison" / "figures"
    suffix = SAVE_PREFIX if SAVE_PREFIX else "default_"
    output_path = figures_dir / f"{suffix}era5_DA_comparison.png"
    build_comparison_figure(model_data, output_path)


if __name__ == "__main__":
    main()
