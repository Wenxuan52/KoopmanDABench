"""
Comparison plot for ERA5 DA vs. no-DA trajectories at a chosen channel/frame.

新增：
- 计算并输出每个 channel 的 DA 相对 No-DA 的误差提升百分比：
  (no_da_err - da_err) / no_da_err
  其中 err 定义为该 channel 在该 frame 的 spatial RMSE
"""
import os
import sys
from typing import Callable, Tuple
import csv

import matplotlib.pyplot as plt
import numpy as np
import torch
import cartopy.crs as ccrs
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable

# Ensure matplotlib can write cache files in restricted environments
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_config")

# Add repository root to path to resolve project modules
CURRENT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.append(REPO_ROOT)

from src.models.CAE_Koopman.ERA5.era5_model_FTF import ERA5_C_FORWARD
from src.utils.Dataset import ERA5Dataset
from src.models.CAE_Koopman.dabase import set_device


DEFAULT_MULTI_PATH = "../../results/CAE_Koopman/ERA5/DA/multi.npy"
DEFAULT_DATA_PATH = "../../data/ERA5/ERA5_data/test_seq_state.h5"
DEFAULT_MIN_PATH = "../../data/ERA5/ERA5_data/min_val.npy"
DEFAULT_MAX_PATH = "../../data/ERA5/ERA5_data/max_val.npy"
DEFAULT_MODEL_DIR = "../../results/CAE_Koopman/ERA5/3loss_model"


def load_groundtruth(
    start_T: int, window_length: int, data_path: str, min_path: str, max_path: str
) -> Tuple[torch.Tensor, ERA5Dataset]:
    """Load ground truth slice and dataset for normalization utilities."""
    dataset = ERA5Dataset(
        data_path=data_path,
        seq_length=12,
        min_path=min_path,
        max_path=max_path,
    )
    raw_data = dataset.data[start_T : start_T + window_length + 1, ...]
    groundtruth = torch.tensor(raw_data, dtype=torch.float32).permute(0, 3, 1, 2)
    return groundtruth, dataset


def load_forward_model(model_dir: str, device: torch.device) -> ERA5_C_FORWARD:
    """Load pretrained Koopman forward model."""
    forward_model = ERA5_C_FORWARD().to(device)
    forward_model.load_state_dict(
        torch.load(
            os.path.join(model_dir, "forward_model.pt"),
            weights_only=True,
            map_location=device,
        )
    )
    forward_model.C_forward = torch.load(
        os.path.join(model_dir, "C_forward.pt"),
        weights_only=True,
        map_location=device,
    ).to(device)
    forward_model.eval()
    return forward_model


def run_no_da_forecast(
    normalized_state: torch.Tensor,
    forward_model: ERA5_C_FORWARD,
    steps: int,
    device: torch.device,
    denormalize: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Propagate a normalized state forward `steps` times in latent space."""
    with torch.no_grad():
        latent = forward_model.K_S(normalized_state.unsqueeze(0).to(device))
        for _ in range(steps):
            latent = forward_model.latent_forward(latent)
        decoded = forward_model.K_S_preimage(latent).cpu()
    return denormalize(decoded).squeeze(0).cpu()


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Spatial RMSE."""
    diff = a - b
    return float(np.sqrt(np.mean(diff * diff)))


def compute_improvement_by_channel(
    groundtruth: torch.Tensor,
    da_states: np.ndarray,
    noda_state: torch.Tensor,
    frame_idx: int,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回：
    - da_rmse_per_ch: (C,)
    - noda_rmse_per_ch: (C,)
    - improve_ratio_per_ch: (C,)  # (noda - da) / noda
    """
    # target shape: (C, H, W)
    target = groundtruth[frame_idx + 1].cpu().numpy()
    # da_pred shape: (C, H, W)  (假设 da_states 是 [T, C, H, W])
    da_pred = da_states[frame_idx]
    # noda_pred shape: (C, H, W)
    noda_pred = noda_state.cpu().numpy()

    C = target.shape[0]
    da_rmse = np.zeros((C,), dtype=np.float64)
    noda_rmse = np.zeros((C,), dtype=np.float64)
    improve = np.zeros((C,), dtype=np.float64)

    for c in range(C):
        da_rmse[c] = rmse(da_pred[c], target[c])
        noda_rmse[c] = rmse(noda_pred[c], target[c])
        denom = max(noda_rmse[c], eps)
        improve[c] = (noda_rmse[c] - da_rmse[c]) / denom  # ratio

    return da_rmse, noda_rmse, improve


def plot_comparison(
    groundtruth: torch.Tensor,
    da_states: np.ndarray,
    noda_state: torch.Tensor,
    channel: int,
    frame_idx: int,
    output_path: str,
) -> None:
    """Render 1x5 comparison grid for a specific channel/frame with map background."""
    target = groundtruth[frame_idx + 1, channel].numpy()
    da_field = da_states[frame_idx, channel]
    noda_field = noda_state[channel].numpy()

    da_error = da_field - target
    noda_error = noda_field - target

    vmin = min(target.min(), da_field.min(), noda_field.min())
    vmax = max(target.max(), da_field.max(), noda_field.max())
    err_abs = max(np.abs(da_error).max(), np.abs(noda_error).max())

    titles = ["Ground Truth", "DA", "DA Error", "No DA", "No DA Error"]
    data = [
        (target, "RdBu_r", vmin, vmax),
        (da_field, "RdBu_r", vmin, vmax),
        (da_error, "bwr", -err_abs, err_abs),
        (noda_field, "RdBu_r", vmin, vmax),
        (noda_error, "bwr", -err_abs, err_abs),
    ]

    lon = np.linspace(-180, 180, target.shape[0])
    lat = np.linspace(-90, 90, target.shape[1])

    fig, axes = plt.subplots(
        1,
        len(data),
        figsize=(22, 4.2),
        subplot_kw={"projection": ccrs.Robinson()},
        constrained_layout=False,
    )
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for ax, (arr, cmap, mn, mx), title in zip(axes, data, titles):
        ax.coastlines(linewidth=0.5)
        ax.gridlines(draw_labels=False, color="gray", linestyle=":", alpha=0.3)

        levels = np.linspace(mn, mx, 30)
        colormap = plt.get_cmap(cmap)
        norm = BoundaryNorm(levels, colormap.N)
        mappable = ScalarMappable(norm=norm, cmap=colormap)
        mappable.set_array([])

        ax.set_title(title, fontsize=14)
        ax.contourf(
            lon,
            lat,
            arr.T,
            levels=levels,
            cmap=cmap,
            norm=norm,
            transform=ccrs.PlateCarree(),
            extend="both",
        )
        cbar = fig.colorbar(
            mappable,
            ax=ax,
            orientation="horizontal",
            fraction=0.035,
            pad=0.045,
            aspect=18,
        )
        ticks = np.linspace(levels[0], levels[-1], num=4)
        cbar.set_ticks(ticks)
        cbar.ax.tick_params(labelsize=10)

    fig.suptitle(f"Channel {channel} | Frame {frame_idx + 1}", fontsize=16)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    # Configuration
    start_T = 0
    window_length = 50
    channel = 0
    frame_idx = 21  # zero-based within assimilation outputs

    # Resolve paths
    multi_path = os.path.abspath(os.path.join(CURRENT_DIR, DEFAULT_MULTI_PATH))
    data_path = os.path.abspath(os.path.join(CURRENT_DIR, DEFAULT_DATA_PATH))
    min_path = os.path.abspath(os.path.join(CURRENT_DIR, DEFAULT_MIN_PATH))
    max_path = os.path.abspath(os.path.join(CURRENT_DIR, DEFAULT_MAX_PATH))
    model_dir = os.path.abspath(os.path.join(CURRENT_DIR, DEFAULT_MODEL_DIR))

    # Load data
    da_states = np.load(multi_path)  # expected shape: (T, C, H, W)
    groundtruth, dataset = load_groundtruth(start_T, window_length, data_path, min_path, max_path)
    normalized_gt = dataset.normalize(groundtruth)

    device = set_device()
    forward_model = load_forward_model(model_dir, device)

    # No-DA propagation to align with DA frame (frame_idx corresponds to step frame_idx+1)
    noda_state = run_no_da_forecast(
        normalized_state=normalized_gt[0],
        forward_model=forward_model,
        steps=frame_idx + 1,
        device=device,
        denormalize=dataset.denormalizer(),
    )

    # ========= 新增：按 channel 计算提升百分比 =========
    da_rmse, noda_rmse, improve_ratio = compute_improvement_by_channel(
        groundtruth=groundtruth,
        da_states=da_states,
        noda_state=noda_state,
        frame_idx=frame_idx,
    )

    # 控制台输出
    print(f"\n[Frame {frame_idx + 1}] DA vs No-DA RMSE improvement by channel:")
    for c in range(len(improve_ratio)):
        print(
            f"  ch{c}: DA_RMSE={da_rmse[c]:.6g}, NoDA_RMSE={noda_rmse[c]:.6g}, "
            f"Improve={(improve_ratio[c]*100):.3f}%"
        )

    # ===============================================

    output_path = os.path.join(CURRENT_DIR, "temp_compare.png")
    plot_comparison(
        groundtruth=groundtruth,
        da_states=da_states,
        noda_state=noda_state,
        channel=channel,
        frame_idx=frame_idx,
        output_path=output_path,
    )
    print(f"Saved comparison figure to {output_path}")


if __name__ == "__main__":
    main()
