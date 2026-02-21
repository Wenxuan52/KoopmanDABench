"""Aggregate visualization of ERA5 DA metrics across models.

This script loads ``{SAVE_PREFIX}multi_meanstd.npz`` files produced by each model's
assimilation experiment and renders a 5x3 grid comparing per-channel
MSE/RRMSE/SSIM across models.

Update:
- Add SAVE_PREFIX so you can switch among different experiment folders/files.
  Example: SAVE_PREFIX = "fullobs_direct_era5_"
- Also load and print timing statistics from ``{SAVE_PREFIX}time_info.npz``:
  * assimilation_time_mean / assimilation_time_std
  * iteration_count_mean / iteration_count_std
- If SAVE_PREFIX contains "interobs", draw observation frames as black dashed
  vertical lines (lw=1.5) using OBSERVATION_SCHEDULE (e.g., [1,10,20,30,40]).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

# Ensure matplotlib can write cache files in restricted environments
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_config")

# =========================
# Experiment prefix switch
# =========================
# Example: "fullobs_direct_era5_"
# Set "" to use the original paths with no prefix.
SAVE_PREFIX = "interobs_direct_era5_"

# If SAVE_PREFIX contains "interobs", we will draw observation frames (x positions)
# as black dashed lines. These are assimilation-step indices (1-based).
OBSERVATION_SCHEDULE: List[int] = [1, 6, 11, 16, 21, 26]

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
# We will inject SAVE_PREFIX before "multi_meanstd.npz" and "time_info.npz".
MODEL_SPECS: List[Tuple[str, Path]] = [
    ("KKR", Path("results/KKR/ERA5/DA/multi_meanstd.npz")),
    ("KAE", Path("results/KAE/ERA5/DA/multi_meanstd.npz")),
    ("WAE", Path("results/WAE/ERA5/DA/multi_meanstd.npz")),
    ("AE", Path("results/AE/ERA5/DA/multi_meanstd.npz")),
    ("DMD", Path("results/DMD/ERA5/DA/multi_meanstd.npz")),
    ("CGKN", Path("results/CGKN/ERA5/DA/multi_meanstd.npz")),
    ("DBF", Path("results/DBF/ERA5/DA/multi_meanstd.npz")),
]

# Distinct colors for models (plot only)
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


def resolve_prefixed_or_fallback(path_with_prefix: Path, fallback: Path) -> Optional[Path]:
    """Return an existing path: prefer prefixed, otherwise fallback, else None."""
    if path_with_prefix.exists():
        return path_with_prefix
    if fallback.exists():
        print(f"[WARN] Missing prefixed file: {path_with_prefix} (fallback to {fallback})")
        return fallback
    print(f"[WARN] Missing file: {path_with_prefix} (and fallback {fallback} not found)")
    return None


def _safe_npz_get(npz: np.lib.npyio.NpzFile, key: str) -> Any:
    """Safely access a key in npz.

    Some .npz keys may be object arrays (saved via dtype=object). Accessing them
    with allow_pickle=False raises ValueError. We catch and return None.
    """
    if key not in npz.files:
        return None
    try:
        return npz[key]
    except ValueError:
        return None


def _to_float_scalar(x: Any) -> Optional[float]:
    """Convert npz value to a Python float if possible."""
    if x is None:
        return None
    try:
        arr = np.asarray(x).squeeze()
        if arr.size == 0:
            return None
        return float(arr)
    except Exception:
        return None


def load_model_metrics(repo_root: Path, prefix: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Load aggregated metrics for each model.

    Returns a mapping:
      model_name -> {
        "color": str,
        metric_key: ndarray (steps, channels, 2),
        f"{metric_key}_std": ndarray (steps, channels, 2),
      }
    Only DA values (index 0 on the last axis) are used in the visualization.
    """
    model_data: Dict[str, Dict[str, np.ndarray]] = {}
    for (name, rel_path), color in zip(MODEL_SPECS, MODEL_COLORS):
        base_path = repo_root / rel_path
        prefixed_path = with_prefix(base_path, prefix)
        resolved = resolve_prefixed_or_fallback(prefixed_path, base_path)
        if resolved is None:
            continue

        data = np.load(resolved)
        entry: Dict[str, np.ndarray] = {"color": color}

        ok = True
        for key, _ in METRIC_KEYS:
            mean_key = f"{key}_mean"
            std_key = f"{key}_std"
            if mean_key not in data.files or std_key not in data.files:
                print(f"[WARN] {resolved} missing keys: {mean_key} or {std_key}; skip {name}")
                ok = False
                break
            entry[key] = data[mean_key]
            entry[f"{key}_std"] = data[std_key]

        if not ok:
            continue

        model_data[name] = entry
        print(f"Loaded metrics for {name} from {resolved}")

    return model_data


def load_model_time_info(repo_root: Path, model_name: str, prefix: str) -> Optional[Dict[str, float]]:
    """Load per-model time info from {prefix}time_info.npz (preferred), else fallback to unprefixed.

    We ONLY read the scalar keys to avoid object-array pickle issues.
    """
    base_path = repo_root / "results" / model_name / "ERA5" / "DA" / "time_info.npz"
    prefixed_path = with_prefix(base_path, prefix)
    resolved = resolve_prefixed_or_fallback(prefixed_path, base_path)
    if resolved is None:
        return None

    npz = np.load(resolved, allow_pickle=False)

    da_mean = _to_float_scalar(_safe_npz_get(npz, "assimilation_time_mean"))
    da_std = _to_float_scalar(_safe_npz_get(npz, "assimilation_time_std"))
    it_mean = _to_float_scalar(_safe_npz_get(npz, "iteration_count_mean"))
    it_std = _to_float_scalar(_safe_npz_get(npz, "iteration_count_std"))

    out: Dict[str, float] = {}
    if da_mean is not None:
        out["da_time_mean"] = da_mean
    if da_std is not None:
        out["da_time_std"] = da_std
    if it_mean is not None:
        out["iter_mean"] = it_mean
    if it_std is not None:
        out["iter_std"] = it_std

    if not out:
        print(f"[WARN] {resolved} loaded but scalar keys not found. Available keys: {npz.files}")
        return None

    out["_path"] = str(resolved)
    return out


def print_time_summary(repo_root: Path, model_names: List[str], prefix: str) -> None:
    print("\n============================")
    print(" ERA5 DA timing summary")
    print("============================")
    print(f"Prefix: {repr(prefix)}\n")

    rows: List[Tuple[str, Dict[str, float]]] = []
    for name in model_names:
        info = load_model_time_info(repo_root, model_name=name, prefix=prefix)
        if info is None:
            continue
        rows.append((name, info))

    if not rows:
        print("No timing info found for any model.")
        return

    name_w = max(len(r[0]) for r in rows)

    header = (
        f"{'Model':<{name_w}}  "
        f"{'DA time mean (s)':>16}  {'DA time std (s)':>15}  "
        f"{'DA iters mean':>14}  {'DA iters std':>13}"
    )
    print(header)
    print("-" * len(header))

    def fmt_float(x: Optional[float], width: int, decimals: int = 4) -> str:
        if x is None:
            return " " * width
        if isinstance(x, float) and np.isnan(x):
            return " " * width
        return f"{x:{width}.{decimals}f}"

    def fmt_iters(x: Optional[float], width: int) -> str:
        if x is None:
            return " " * width
        if isinstance(x, float) and np.isnan(x):
            return " " * width
        if abs(x - round(x)) < 1e-6:
            return f"{int(round(x)):{width}d}"
        return f"{x:{width}.4f}"

    for name, info in rows:
        da_mean = info.get("da_time_mean")
        da_std = info.get("da_time_std")
        it_mean = info.get("iter_mean")
        it_std = info.get("iter_std")

        print(
            f"{name:<{name_w}}  "
            f"{fmt_float(da_mean, 16)}  {fmt_float(da_std, 15)}  "
            f"{fmt_iters(it_mean, 14)}  {fmt_iters(it_std, 13)}"
        )


def _draw_observation_lines(ax: plt.Axes, steps_len: int, observation_schedule: Sequence[int]) -> None:
    """Draw black dashed vertical lines at observation frames (within [1, steps_len])."""
    for s in observation_schedule:
        try:
            x = int(s)
        except Exception:
            continue
        if 1 <= x <= steps_len:
            ax.axvline(x=x, color="k", linestyle="--", linewidth=1.5, alpha=0.9, zorder=3)


def build_comparison_figure(
    model_data: Dict[str, Dict[str, np.ndarray]],
    output_path: Path,
    observation_schedule: Optional[Sequence[int]] = None,
) -> plt.Figure:
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

            # Plot each model
            steps_len: Optional[int] = None
            for name in model_names:
                mean_arr = model_data[name][metric_key][:, row, 0]
                std_arr = model_data[name][f"{metric_key}_std"][:, row, 0]
                steps = np.arange(1, len(mean_arr) + 1)
                steps_len = len(mean_arr)

                (line,) = ax.plot(steps, mean_arr, color=model_data[name]["color"], label=name, zorder=2)
                ax.fill_between(
                    steps,
                    mean_arr - std_arr,
                    mean_arr + std_arr,
                    color=model_data[name]["color"],
                    alpha=0.15,
                    zorder=1,
                )
                if row == 0 and col == 0:
                    handles.append(line)

            # Draw observation schedule lines (if requested)
            if observation_schedule is not None and steps_len is not None:
                _draw_observation_lines(ax, steps_len=steps_len, observation_schedule=observation_schedule)

            ax.set_title(metric_label if row == 0 else "")
            if col == 0:
                ax.set_ylabel(channel_label)
            if row == len(CHANNEL_LABELS) - 1:
                ax.set_xlabel("Assimilation step")
            ax.grid(True, linestyle="--", alpha=0.4, zorder=0)

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

    fig.tight_layout(rect=[0, 0.06, 1, 1.0])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved comparison figure to {output_path}")
    return fig


def main() -> Optional[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    model_data = load_model_metrics(repo_root, prefix=SAVE_PREFIX)
    if not model_data:
        print("No metrics found; aborting plot generation.")
        return None

    model_names = list(model_data.keys())

    # Print timing info from {prefix}time_info.npz
    print_time_summary(repo_root, model_names, prefix=SAVE_PREFIX)

    # Decide whether to draw observation schedule lines
    obs_schedule_to_draw: Optional[Sequence[int]] = None
    if "interobs" in (SAVE_PREFIX or "").lower():
        obs_schedule_to_draw = OBSERVATION_SCHEDULE
        print(f"\n[Info] interobs detected in SAVE_PREFIX; draw observation lines at: {list(obs_schedule_to_draw)}")
    else:
        print("\n[Info] fullobs (or non-interobs) detected; no observation lines will be drawn.")

    figures_dir = repo_root / "results" / "Comparison" / "figures"
    suffix = SAVE_PREFIX if SAVE_PREFIX else "default_"
    output_path = figures_dir / f"{suffix}era5_DA_comparison.png"

    fig = build_comparison_figure(
        model_data=model_data,
        output_path=output_path,
        observation_schedule=obs_schedule_to_draw,
    )
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    main()
