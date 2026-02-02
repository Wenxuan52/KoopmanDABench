import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless environments
import matplotlib.pyplot as plt

import sys

current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", ".."))
if src_directory not in sys.path:
    sys.path.append(src_directory)

from src.utils.Dataset import ERA5Dataset

# -------------------------
# Config
# -------------------------
MODEL_MAP = {
    "Koopman": "CAE_Koopman",
    "Linear": "CAE_Linear",
    "Weaklinear": "CAE_Weaklinear",
    "MLP": "CAE_MLP",
    "DMD": "DMD",
    "DBF": "DBF",
    "CGKN": "CGKN",
}

CHANNEL_LABELS = [
    "Geopotential",
    "Temperature",
    "Humidity",
    "Wind U-direction",
    "Wind V-direction",
]

# Baseline colors (7) -> exactly for 7 baselines
BASELINE_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#e377c2",
    "#9467bd",
    "#8c564b",
    "#17becf",
]

# Ground Truth emphasized
GT_COLOR = "#ff0000"   # "big red"
GT_ALPHA = 1.0
GT_LW = 1.5

# DA / no-DA styles for baselines
DA_ALPHA = 1.0
DA_LW = 1.0
BG_ALPHA = 0.35
BG_LW = 0.7


# -------------------------
# IO helpers
# -------------------------
def load_predictions(results_root: str, filename: str, subdir: str = "DA") -> dict:
    """
    Load predictions for all models from:
      results_root/<model_name>/ERA5/<subdir>/<filename>
    Return dict keyed by "Koopman"/"Linear"/...
    """
    preds = {}
    for label, model_name in MODEL_MAP.items():
        path = os.path.join(results_root, model_name, "ERA5", subdir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing result: {path}")
        preds[label] = np.load(path)
    return preds


def load_groundtruth(
    data_path: str,
    min_path: str,
    max_path: str,
    start_t: int,
    num_frames: int,
    seq_length: int = 12,
) -> np.ndarray:
    """
    Returns (T, C, H, W)
    """

    dataset = ERA5Dataset(
        data_path=data_path,
        seq_length=seq_length,
        min_path=min_path,
        max_path=max_path,
    )
    raw_data = dataset.data  # expected (T, H, W, C)

    start = start_t + 1
    end = start + num_frames
    if end > raw_data.shape[0]:
        raise ValueError(f"Requested frames [{start}, {end}) exceed dataset length {raw_data.shape[0]}")

    groundtruth = raw_data[start:end]
    return np.transpose(groundtruth, (0, 3, 1, 2))


# -------------------------
# Spectrum helpers
# -------------------------
def _hann2d(h: int, w: int) -> np.ndarray:
    wy = np.hanning(h)
    wx = np.hanning(w)
    return wy[:, None] * wx[None, :]


def isotropic_power_spectrum_2d(field2d: np.ndarray, apply_window: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    field2d: (H, W)
    Return:
      k: integer wavenumber index (1..kmax)
      E(k): isotropic summed power in radial bins
    """
    f = np.asarray(field2d, dtype=np.float64)
    f = f - np.mean(f)

    H, W = f.shape

    if apply_window:
        win = _hann2d(H, W)
        win_norm = np.mean(win**2)
        f = f * win
    else:
        win_norm = 1.0

    F = np.fft.fft2(f)
    N = H * W
    P = (np.abs(F) ** 2) / (N**2 * win_norm)

    # integer wavenumber indices
    kx = np.fft.fftfreq(W) * W
    ky = np.fft.fftfreq(H) * H
    KX, KY = np.meshgrid(kx, ky)
    k_mag = np.sqrt(KX**2 + KY**2).ravel()

    P_flat = P.ravel()
    k_int = np.rint(k_mag).astype(np.int32)

    kmax = int(k_int.max())
    Ek = np.zeros(kmax + 1, dtype=np.float64)
    np.add.at(Ek, k_int, P_flat)

    # drop k=0
    k = np.arange(1, kmax + 1)
    Ek = Ek[1:]
    return k, Ek


def time_mean_spectrum(arr: np.ndarray, ch: int) -> tuple[np.ndarray, np.ndarray]:
    """
    arr: (T, C, H, W)
    return mean spectrum over time for channel ch
    """
    T = arr.shape[0]
    k_ref, E0 = isotropic_power_spectrum_2d(arr[0, ch])
    acc = np.zeros_like(E0, dtype=np.float64)

    for t in range(T):
        k, E = isotropic_power_spectrum_2d(arr[t, ch])
        L = min(len(acc), len(E))
        acc[:L] += E[:L]

    acc /= float(T)
    return k_ref, acc


# -------------------------
# Plot main
# -------------------------
def make_spatial_power_spectrum_comparison(
    data_path: str,
    min_path: str,
    max_path: str,
    results_root: str,
    result_filename: str = "fullobs_direct_era5_multi.npy",
    rollout_filename: str = "fullobs_direct_era5_multi_original.npy",
    out_path: str = "results/Comparison/figures/spatial_power_spectrum_compare.png",
    start_t: int = 0,
):
    # load preds
    preds_da = load_predictions(results_root, result_filename, subdir="DA")
    preds_bg = load_predictions(results_root, rollout_filename, subdir="DA")

    # align time length
    T_da = next(iter(preds_da.values())).shape[0]
    T_bg = next(iter(preds_bg.values())).shape[0]
    T = min(T_da, T_bg)

    # ground truth aligned
    gt = load_groundtruth(
        data_path=data_path,
        min_path=min_path,
        max_path=max_path,
        start_t=start_t,
        num_frames=T,
    )

    # drawing order
    model_keys = list(MODEL_MAP.keys())  # 7 baselines
    assert len(model_keys) == len(BASELINE_COLORS), (
        f"Need {len(model_keys)} baseline colors, but got {len(BASELINE_COLORS)}"
    )

    # legend labels
    legend_labels = ["ground truth"] + [MODEL_MAP[k] for k in model_keys]

    # figure: 1x5
    fig, axes = plt.subplots(1, 5, figsize=(24, 4.8), constrained_layout=True)

    for ch in range(5):
        ax = axes[ch]

        # --- Ground Truth (emphasized) ---
        k_gt, E_gt = time_mean_spectrum(gt, ch)
        ax.loglog(
            k_gt, E_gt,
            color=GT_COLOR,
            alpha=GT_ALPHA,
            lw=GT_LW,
            label=legend_labels[0],
            zorder=10,
        )

        # --- Baselines: DA with legend, BG without legend ---
        for i, mk in enumerate(model_keys):
            color = BASELINE_COLORS[i]

            k_da, E_da = time_mean_spectrum(preds_da[mk], ch)
            ax.loglog(
                k_da, E_da,
                color=color,
                alpha=DA_ALPHA,
                lw=DA_LW,
                label=legend_labels[i + 1],
                zorder=5,
            )

            k_bg, E_bg = time_mean_spectrum(preds_bg[mk], ch)
            ax.loglog(
                k_bg, E_bg,
                color=color,
                alpha=BG_ALPHA,
                lw=BG_LW,
                zorder=1,
            )

        ax.set_title(CHANNEL_LABELS[ch], fontsize=12, fontweight="bold")
        ax.set_xlabel("Wavenumber k")
        if ch == 0:
            ax.set_ylabel("Spatial Power Spectrum")
        ax.grid(True, which="both", ls=":", alpha=0.3)

    # global legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=8,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 1.08),
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    make_spatial_power_spectrum_comparison(
        data_path="../../data/ERA5/ERA5_data/test_seq_state.h5",
        min_path="../../data/ERA5/ERA5_data/min_val.npy",
        max_path="../../data/ERA5/ERA5_data/max_val.npy",
        results_root="../../results",
        result_filename="fullobs_direct_era5_multi.npy",
        rollout_filename="fullobs_direct_era5_multi_original.npy",
        out_path="../../results/Comparison/figures/era5_spectrum_compare.png",
        start_t=0,
    )
