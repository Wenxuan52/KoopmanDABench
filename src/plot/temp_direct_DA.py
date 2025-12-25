"""
Visualization utilities for ERA5 data assimilation outputs.

This script loads a saved assimilation trajectory (`multi.npy`) and the
aggregated metrics (`multi_meanstd.npz`) to create two figures:
1) Ground truth vs. data-assimilated fields with error maps across 5
   evenly spaced steps in the assimilation window.
2) Per-step curves of MSE, RRMSE, and SSIM (mean ± std over channels).

Images are saved alongside this script.
"""

import os
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils.Dataset import ERA5Dataset

# Ensure matplotlib can write cache files in restricted environments
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_config")


DEFAULT_MULTI_PATH = "../../results/CAE_Koopman/ERA5/DA/multi.npy"
DEFAULT_METRICS_PATH = "../../results/CAE_Koopman/ERA5/DA/multi_meanstd.npz"
DEFAULT_DATA_PATH = "../../data/ERA5/ERA5_data/test_seq_state.h5"
DEFAULT_MIN_PATH = "../../data/ERA5/ERA5_data/min_val.npy"
DEFAULT_MAX_PATH = "../../data/ERA5/ERA5_data/max_val.npy"


def load_groundtruth(
    start_T: int, window_length: int, data_path: str, min_path: str, max_path: str
) -> torch.Tensor:
    """Load a ground-truth slice matching the assimilation window."""

    dataset = ERA5Dataset(
        data_path=data_path,
        seq_length=12,
        min_path=min_path,
        max_path=max_path,
    )
    raw_data = dataset.data[start_T : start_T + window_length + 1, ...]
    groundtruth = torch.tensor(raw_data, dtype=torch.float32).permute(0, 3, 1, 2)
    return groundtruth


def _evenly_spaced_indices(total_steps: int, num_samples: int = 5) -> np.ndarray:
    """Return `num_samples` indices evenly spread over `[0, total_steps)`."""

    num_samples = max(1, min(num_samples, total_steps))
    if num_samples == 1:
        return np.array([0], dtype=int)
    return np.linspace(0, total_steps - 1, num_samples, dtype=int)


def plot_field_comparison(
    groundtruth: torch.Tensor,
    da_states: np.ndarray,
    channel: int,
    output_path: str,
) -> None:
    """Plot ground truth, DA result, and error for evenly spaced steps."""

    assert da_states.ndim == 4, "DA states should have shape (T, C, H, W)."
    steps = da_states.shape[0]
    indices = _evenly_spaced_indices(steps, num_samples=5)

    fig, axes = plt.subplots(3, len(indices), figsize=(3 * len(indices), 8))
    row_labels = ["Ground Truth", "DA", "Error"]

    for col, step in enumerate(indices):
        target = groundtruth[step + 1, channel].numpy()
        da_field = da_states[step, channel]
        error = da_field - target

        vmin = min(target.min(), da_field.min())
        vmax = max(target.max(), da_field.max())
        err_abs = np.abs(error).max()

        images = [
            (target, "coolwarm", vmin, vmax),
            (da_field, "coolwarm", vmin, vmax),
            (error, "bwr", -err_abs, err_abs),
        ]

        for row, (data, cmap, mn, mx) in enumerate(images):
            ax = axes[row, col]
            im = ax.imshow(data, cmap=cmap, vmin=mn, vmax=mx)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=10)
            if row == 0:
                ax.set_title(f"step {step + 1}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_metric_curves(
    metrics_path: str, output_path: str, channels: Sequence[int] = None
) -> None:
    """Plot MSE, RRMSE, and SSIM mean ± std curves over assimilation steps."""

    data = np.load(metrics_path)
    steps = data.get("steps")
    if steps is None:
        steps = np.arange(1, data["mse_mean"].shape[0] + 1)

    def _channel_slice(array: np.ndarray) -> np.ndarray:
        if channels is None:
            return array
        return array[:, channels, :]

    metrics: Sequence[Tuple[str, str]] = (
        ("mse", "MSE"),
        ("rrmse", "RRMSE"),
        ("ssim", "SSIM"),
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (key, label) in zip(axes, metrics):
        mean_arr = _channel_slice(data[f"{key}_mean"])  # (steps, C, 2)
        std_arr = _channel_slice(data[f"{key}_std"])    # (steps, C, 2)

        da_mean = mean_arr[:, :, 0].mean(axis=1)
        da_std = std_arr[:, :, 0].mean(axis=1)

        ax.plot(steps, da_mean, label="DA", color="tab:blue")
        ax.fill_between(steps, da_mean - da_std, da_mean + da_std, color="tab:blue", alpha=0.2)
        ax.set_xlabel("Step")
        ax.set_ylabel(label)
        ax.set_title(f"{label} over steps")
        ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    multi_path = os.path.abspath(os.path.join(os.path.dirname(__file__), DEFAULT_MULTI_PATH))
    metrics_path = os.path.abspath(os.path.join(os.path.dirname(__file__), DEFAULT_METRICS_PATH))
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), DEFAULT_DATA_PATH))
    min_path = os.path.abspath(os.path.join(os.path.dirname(__file__), DEFAULT_MIN_PATH))
    max_path = os.path.abspath(os.path.join(os.path.dirname(__file__), DEFAULT_MAX_PATH))

    da_states = np.load(multi_path)
    window_length = da_states.shape[0]
    start_T = 1000  # must match the assimilation run

    groundtruth = load_groundtruth(start_T, window_length, data_path, min_path, max_path)

    fields_output = os.path.join(os.path.dirname(__file__), "direct_DA_fields.png")
    plot_field_comparison(groundtruth, da_states, channel=0, output_path=fields_output)
    print(f"Saved field comparison to {fields_output}")

    metrics_output = os.path.join(os.path.dirname(__file__), "direct_DA_metrics.png")
    plot_metric_curves(metrics_path, metrics_output)
    print(f"Saved metric curves to {metrics_output}")


if __name__ == "__main__":
    main()
