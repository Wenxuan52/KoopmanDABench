import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless environments
import matplotlib.pyplot as plt

current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", ".."))
if src_directory not in sys.path:
    sys.path.append(src_directory)

from src.utils.Dataset import ERA5Dataset

# -------------------------
# Config
# -------------------------
MODEL_MAP = {
    "Koopman": "KKR",
    "Linear": "KAE",
    "Weaklinear": "WAE",
    "MLP": "AE",
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

# Colors / styles
GT_COLOR = "#ff0000"     # red
PRED_COLOR = "#1f77b4"   # blue

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
        raise ValueError(
            f"Requested frames [{start}, {end}) exceed dataset length {raw_data.shape[0]}"
        )

    groundtruth = raw_data[start:end]
    return np.transpose(groundtruth, (0, 3, 1, 2))


# -------------------------
# Spectrum helpers
# -------------------------
def _hann2d(h: int, w: int) -> np.ndarray:
    wy = np.hanning(h)
    wx = np.hanning(w)
    return wy[:, None] * wx[None, :]


def isotropic_power_spectrum_2d(
    field2d: np.ndarray, apply_window: bool = True
) -> tuple[np.ndarray, np.ndarray]:
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
# Plot main (5 x 7)
# -------------------------
def make_spatial_power_spectrum_grid(
    data_path: str,
    min_path: str,
    max_path: str,
    results_root: str,
    da_filename: str = "fullobs_direct_era5_multi.npy",
    noda_filename: str = "fullobs_direct_era5_multi_original.npy",
    da_subdir: str = "DA",
    noda_subdir: str = "DA",
    out_path: str = "results/Comparison/figures/era5_spectrum_grid.png",
    start_t: int = 0,
    # ---- font interfaces ----
    row_label_fontsize: int = 14,
    col_label_fontsize: int = 14,
    tick_fontsize: int = 9,
    sup_fontsize: int = 12,
    legend_fontsize: int = 12,
    # ---- line style interfaces ----
    gt_lw: float = 2.2,
    da_lw: float = 2.0,
    noda_lw: float = 2.0,
):
    # Load predictions
    preds_da = load_predictions(results_root, da_filename, subdir=da_subdir)
    preds_noda = load_predictions(results_root, noda_filename, subdir=noda_subdir)

    # Align time length
    T_da = next(iter(preds_da.values())).shape[0]
    T_noda = next(iter(preds_noda.values())).shape[0]
    T = min(T_da, T_noda)

    # Ground truth aligned
    gt = load_groundtruth(
        data_path=data_path,
        min_path=min_path,
        max_path=max_path,
        start_t=start_t,
        num_frames=T,
    )

    model_keys = list(MODEL_MAP.keys())  # 7 models (columns)
    nrows = len(CHANNEL_LABELS)          # 5 channels (rows)
    ncols = len(model_keys)

    # Figure sizing: tune as you like
    fig_w = 3.4 * ncols
    fig_h = 2.6 * nrows
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_w, fig_h),
        sharex=True,
        sharey="row",   # compare models within each channel row
        constrained_layout=False,
    )

    # Precompute GT spectra per channel (used across the row)
    gt_spec = []
    for ch in range(nrows):
        k_gt, E_gt = time_mean_spectrum(gt, ch)
        gt_spec.append((k_gt, E_gt))

    # For global legend: store handles once
    legend_handles = None
    legend_labels = ["Ground Truth", "DA", "No DA"]

    for r in range(nrows):
        k_gt, E_gt = gt_spec[r]

        for c in range(ncols):
            mk = model_keys[c]
            ax = axes[r, c] if nrows > 1 else axes[c]

            # --- Plot GT ---
            h_gt = ax.loglog(
                k_gt,
                E_gt,
                color=GT_COLOR,
                lw=gt_lw,
                linestyle="-",
                zorder=10,
            )[0]

            # --- Plot DA ---
            k_da, E_da = time_mean_spectrum(preds_da[mk][:T], r)
            h_da = ax.loglog(
                k_da,
                E_da,
                color=PRED_COLOR,
                lw=da_lw,
                linestyle="-",
                zorder=6,
            )[0]

            # --- Plot No DA (dashed) ---
            k_noda, E_noda = time_mean_spectrum(preds_noda[mk][:T], r)
            h_noda = ax.loglog(
                k_noda,
                E_noda,
                color=PRED_COLOR,
                lw=noda_lw,
                linestyle="--",
                zorder=5,
            )[0]

            # Save handles once for a unified legend
            if legend_handles is None:
                legend_handles = [h_gt, h_da, h_noda]

            # Grid / ticks
            ax.grid(True, which="both", ls=":", alpha=0.3)
            ax.tick_params(axis="both", which="both", labelsize=tick_fontsize)

            # Row labels only on first column (bold)
            if c == 0:
                ax.set_ylabel(
                    CHANNEL_LABELS[r],
                    fontsize=row_label_fontsize,
                    fontweight="bold",
                )

            # Column labels only on first row (bold)
            if r == 0:
                ax.set_title(
                    MODEL_MAP[mk],
                    fontsize=col_label_fontsize,
                    fontweight="bold",
                )

    # # Global x/y labels
    # fig.supxlabel("Wavenumber k", fontsize=sup_fontsize)
    # fig.supylabel("Spatial Power Spectrum", fontsize=sup_fontsize)

    # Unified legend at the bottom for all subplots
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=legend_fontsize,
        bbox_to_anchor=(0.5, 0.01),
    )

    # Leave space for legend + suxlabel
    fig.subplots_adjust(left=0.06, right=0.995, top=0.92, bottom=0.08, wspace=0.18, hspace=0.22)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    make_spatial_power_spectrum_grid(
        data_path="../../data/ERA5/ERA5_data/test_seq_state.h5",
        min_path="../../data/ERA5/ERA5_data/min_val.npy",
        max_path="../../data/ERA5/ERA5_data/max_val.npy",
        results_root="../../results",
        da_filename="fullobs_direct_era5_multi.npy",
        noda_filename="fullobs_direct_era5_multi_original.npy",
        da_subdir="DA",
        noda_subdir="DA",
        out_path="../../results/Comparison/figures/era5_spectrum_grid.png",
        start_t=0,
        # you can tune these
        row_label_fontsize=20,
        col_label_fontsize=20,
        tick_fontsize=9,
        sup_fontsize=13,
        legend_fontsize=14,
        gt_lw=2.6,
        da_lw=2.4,
        noda_lw=2.4,
    )
