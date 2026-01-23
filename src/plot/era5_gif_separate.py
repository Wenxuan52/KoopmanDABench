import sys
from datetime import datetime, timedelta

import os

# Pick a writable cache base (prefer SLURM_TMPDIR, else /tmp, else scratch)
TMP_BASE = os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or "/scratch_root/wy524/.cache"
MPLDIR = os.path.join(TMP_BASE, "mpl_config")
CARTOPYDIR = os.path.join(TMP_BASE, "cartopy_data")

os.makedirs(MPLDIR, exist_ok=True)
os.makedirs(CARTOPYDIR, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", MPLDIR)
os.environ.setdefault("CARTOPY_DATA_DIR", CARTOPYDIR)

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
cartopy.config["data_dir"] = os.environ["CARTOPY_DATA_DIR"]
from matplotlib.colors import BoundaryNorm
from matplotlib.animation import FuncAnimation, PillowWriter

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.append(repo_root)

from src.utils.Dataset import ERA5Dataset


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

COL_LABELS = [
    "GroundTruth",
    "Koopman",
    "Linear",
    "Weaklinear",
    "MLP",
    "DMD",
    "DBF",
    "CGKN",
]


def load_groundtruth(
    data_path: str,
    min_path: str,
    max_path: str,
    start_t: int,
    num_frames: int,
    seq_length: int = 12,
) -> np.ndarray:
    dataset = ERA5Dataset(
        data_path=data_path,
        seq_length=seq_length,
        min_path=min_path,
        max_path=max_path,
    )
    raw_data = dataset.data
    start = start_t + 1
    end = start + num_frames
    if end > raw_data.shape[0]:
        raise ValueError(f"Requested frames [{start}, {end}) exceed dataset length {raw_data.shape[0]}")
    groundtruth = raw_data[start:end]
    return np.transpose(groundtruth, (0, 3, 1, 2))


def load_predictions(results_root: str, filename: str) -> dict:
    preds = {}
    for label, model_name in MODEL_MAP.items():
        path = os.path.join(results_root, model_name, "ERA5", "DA", filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing assimilation result: {path}")
        preds[label] = np.load(path)
    return preds


def compute_value_ranges(groundtruth: np.ndarray, preds: dict) -> tuple[list, list, list, list]:
    vmin_main, vmax_main = [], []
    vmin_err, vmax_err = [], []
    for ch in range(5):
        main_values = [groundtruth[:, ch]]
        for pred in preds.values():
            main_values.append(pred[:, ch])
        main_stack = np.stack(main_values, axis=0)
        vmin_main.append(float(np.min(main_stack)))
        vmax_main.append(float(np.max(main_stack)))

        err_values = []
        for pred in preds.values():
            err_values.append(np.abs(pred[:, ch] - groundtruth[:, ch]))
        err_stack = np.stack(err_values, axis=0)
        vmin_err.append(float(np.min(err_stack)))
        vmax_err.append(float(np.max(err_stack)))
    return vmin_main, vmax_main, vmin_err, vmax_err


def make_era5_da_gif_separate(
    data_path: str,
    min_path: str,
    max_path: str,
    results_root: str = "results",
    result_filename: str = "fullobs_direct_era5_multi.npy",
    out_dir: str = "results/Comparison/figures",
    start_t: int = 0,
    start_datetime_str: str = "2018-05-05 00:00",
    hours_per_frame: int = 4,
):
    preds = load_predictions(results_root, result_filename)
    num_frames = next(iter(preds.values())).shape[0]

    groundtruth = load_groundtruth(
        data_path=data_path,
        min_path=min_path,
        max_path=max_path,
        start_t=start_t,
        num_frames=num_frames,
    )

    vmin_main, vmax_main, vmin_err, vmax_err = compute_value_ranges(groundtruth, preds)

    lon = np.linspace(-180, 180, groundtruth.shape[-1])
    lat = np.linspace(-90, 90, groundtruth.shape[-2])
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    cmap_main = plt.cm.RdBu_r
    cmap_err = plt.cm.RdBu_r

    os.makedirs(out_dir, exist_ok=True)
    start_datetime = datetime.strptime(start_datetime_str, "%Y-%m-%d %H:%M")

    def is_error_row(row):
        return row == 1

    for ch in range(5):
        fig, axes = plt.subplots(
            2,
            8,
            figsize=(30, 7),
            subplot_kw={"projection": ccrs.Robinson()},
            constrained_layout=False,
        )
        plt.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.10,
                    wspace=0.18, hspace=0.02)

        def get_levels_norm_cmap(row, col):
            if COL_LABELS[col] == "GroundTruth" and is_error_row(row):
                return None, None, None
            if is_error_row(row):
                levels = np.linspace(vmin_err[ch], vmax_err[ch], 20)
                norm = BoundaryNorm(levels, cmap_err.N, clip=True)
                return levels, norm, cmap_err
            levels = np.linspace(vmin_main[ch], vmax_main[ch], 30)
            norm = BoundaryNorm(levels, cmap_main.N, clip=True)
            return levels, norm, cmap_main

        def draw_frame(t_idx, with_colorbar=False):
            current_time = start_datetime + timedelta(hours=hours_per_frame * t_idx)
            fig.suptitle(f"{CHANNEL_LABELS[ch]} {current_time:%Y-%m-%d %H:%M}", fontsize=26, fontweight="bold", y=0.93)

            for row in range(2):
                for col in range(8):
                    ax = axes[row, col]
                    model_name = COL_LABELS[col]

                    ax.clear()

                    if model_name == "GroundTruth" and is_error_row(row):
                        ax.axis("off")
                        continue

                    ax.set_global()
                    ax.coastlines(linewidth=0.4)
                    ax.gridlines(draw_labels=False, color="gray", linestyle=":", alpha=0.3)

                    levels, norm, cmap = get_levels_norm_cmap(row, col)
                    if levels is None:
                        continue

                    if model_name == "GroundTruth":
                        data = groundtruth[t_idx, ch]
                    else:
                        pred = preds[model_name]
                        if is_error_row(row):
                            data = np.abs(pred[t_idx, ch] - groundtruth[t_idx, ch])
                        else:
                            data = pred[t_idx, ch]

                    plot_data = data.T
                    im = ax.imshow(
                        plot_data,
                        origin="lower",
                        extent=extent,
                        transform=ccrs.PlateCarree(),
                        cmap=cmap,
                        norm=norm,
                        interpolation="nearest",
                    )

                    # if col == 0:
                    #     label = CHANNEL_LABELS[ch] if not is_error_row(row) else "Error"
                    #     ax.text(
                    #         -0.2, 0.5, label,
                    #         transform=ax.transAxes,
                    #         ha="center", va="center",
                    #         fontsize=16, fontweight="bold", rotation=90,
                    #     )

                    if row == 0:
                        ax.set_title(model_name, fontsize=20, fontweight="bold", pad=16)

                    if with_colorbar:
                        cbar = fig.colorbar(
                            im,
                            ax=ax,
                            orientation="horizontal",
                            fraction=0.05,
                            pad=0.04,
                            aspect=24,
                        )
                        ticks = np.linspace(levels[0], levels[-1], num=4)
                        cbar.set_ticks(ticks)
                        cbar.ax.tick_params(labelsize=10)

        def init():
            draw_frame(t_idx=0, with_colorbar=True)
            return []

        def update(t_idx):
            draw_frame(t_idx=t_idx, with_colorbar=False)
            return []

        frames = list(range(num_frames))
        anim = FuncAnimation(
            fig,
            update,
            frames=frames,
            init_func=init,
            blit=False,
            interval=1000,
            repeat=True,
        )

        out_path = os.path.join(out_dir, f"era5_da_fullobs_channel{ch}.gif")
        writer = PillowWriter(fps=1.0)
        anim.save(out_path, writer=writer)

        plt.close(fig)
        print(f"GIF saved to {out_path}")


if __name__ == "__main__":
    make_era5_da_gif_separate(
        data_path="../../data/ERA5/ERA5_data/test_seq_state.h5",
        min_path="../../data/ERA5/ERA5_data/min_val.npy",
        max_path="../../data/ERA5/ERA5_data/max_val.npy",
        results_root="../../results",
        result_filename="fullobs_direct_era5_multi.npy",
        out_dir="../../results/Comparison/figures",
        start_t=0,
        start_datetime_str="2018-01-01 04:00",
        hours_per_frame=4,
    )
