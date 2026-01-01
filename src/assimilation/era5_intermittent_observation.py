"""Unified ERA5 intermittent observation experiments across multiple models.

This script configures a shared data assimilation setup and sequentially runs
model-specific ERA5 data assimilation entry points. After each run it reloads
the saved assimilation trajectory, compares it against ground truth, prints
channel-wise metrics, and records wall-clock time.
"""
from __future__ import annotations

import contextlib
import importlib
import sys
import os
import time
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch

current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", ".."))
if src_directory not in sys.path:
    sys.path.append(src_directory)

from src.utils.Dataset import ERA5Dataset


def compute_channel_metrics(
    da_states: np.ndarray, groundtruth: np.ndarray
) -> Dict[str, np.ndarray]:
    """Compute per-channel MSE, RRMSE, and SSIM between DA outputs and truth.

    Args:
        da_states: Array of shape (steps, channels, height, width) containing
            denormalized assimilation predictions.
        groundtruth: Array of shape (steps + 1, channels, height, width) with
            denormalized ground-truth states. The first element corresponds to
            the initial state, while metrics are evaluated against elements
            1..steps.

    Returns:
        Dictionary mapping metric name to an array of shape (channels,) holding
        the time-averaged values for each channel.
    """

    targets = groundtruth[1: da_states.shape[0] + 1]
    diff = da_states - targets

    mse = (diff ** 2).mean(axis=(0, 2, 3))
    rrmse = np.sqrt(((diff ** 2).sum(axis=(0, 2, 3))) / ((targets ** 2).sum(axis=(0, 2, 3))))

    per_step_ssim: list[list[float]] = []
    for step in range(da_states.shape[0]):
        step_scores: list[float] = []
        for channel in range(da_states.shape[1]):
            gt_frame = targets[step, channel]
            da_frame = da_states[step, channel]
            data_range = gt_frame.max() - gt_frame.min()
            if data_range == 0:
                step_scores.append(1.0)
                continue
            step_scores.append(
                ssim(
                    gt_frame,
                    da_frame,
                    data_range=float(data_range),
                )
            )
        per_step_ssim.append(step_scores)

    ssim_scores = np.mean(np.asarray(per_step_ssim), axis=0)

    return {"mse": mse, "rrmse": rrmse, "ssim": ssim_scores}


def format_metric_row(name: str, values: Iterable[float]) -> str:
    values_str = ", ".join(f"{v:.6f}" for v in values)
    return f"  {name.upper():<5}: [{values_str}]"


@contextlib.contextmanager
def working_directory(path: Path):
    original_cwd = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(original_cwd)


def main():
    repo_root = Path(__file__).resolve().parents[2]

    # Shared DA configuration
    window_length = 50
    start_T = 0
    observation_schedule = [0, 10, 20, 30, 40]

    base_kwargs = {
        "obs_ratio": 0.15,
        "obs_noise_std": 0.05,
        "observation_schedule": observation_schedule,
        "observation_variance": None,
        "window_length": window_length,
        "num_runs": 5,
        "early_stop_config": (100, 1e-3),
        "start_T": start_T,
    }

    # Prepare ground truth slice once for metric computation
    dataset = ERA5Dataset(
        data_path=repo_root / "data/ERA5/ERA5_data/test_seq_state.h5",
        seq_length=12,
        min_path=repo_root / "data/ERA5/ERA5_data/min_val.npy",
        max_path=repo_root / "data/ERA5/ERA5_data/max_val.npy",
    )
    groundtruth = (
        torch.tensor(dataset.data[start_T : start_T + window_length + 1])
        .permute(0, 3, 1, 2)
        .numpy()
    )

    models: Sequence[Dict[str, object]] = [
        {
            "name": "CAE_Koopman",
            "module": "src.models.CAE_Koopman.ERA5.era5_DA",
            "working_dir": repo_root / "src/models/CAE_Koopman/ERA5",
        },
        {
            "name": "CAE_Linear",
            "module": "src.models.CAE_Linear.ERA5.era5_DA",
            "working_dir": repo_root / "src/models/CAE_Linear/ERA5",
        },
        {
            "name": "CAE_Weaklinear",
            "module": "src.models.CAE_Weaklinear.ERA5.era5_DA",
            "working_dir": repo_root / "src/models/CAE_Weaklinear/ERA5",
        },
        {
            "name": "CAE_MLP",
            "module": "src.models.CAE_MLP.ERA5.era5_DA",
            "working_dir": repo_root / "src/models/CAE_MLP/ERA5",
        },
        {
            "name": "DMD",
            "module": "src.models.DMD.ERA5.era5_DA",
            "working_dir": repo_root / "src/models/DMD/ERA5",
        },
        {
            "name": "discreteCGKN",
            "module": "src.models.discreteCGKN.ERA5.era5_DA",
            "working_dir": repo_root / "src/models/discreteCGKN/ERA5",
            "extra_args": {
                "data_path": "../../../../data/ERA5/ERA5_data/test_seq_state.h5",
                "min_path": "../../../../data/ERA5/ERA5_data/min_val.npy",
                "max_path": "../../../../data/ERA5/ERA5_data/max_val.npy",
                "ckpt_prefix": "stage2",
            },
        },
        {
            "name": "DBF",
            "module": "src.models.DBF.ERA5.era5_DA",
            "working_dir": repo_root / "src/models/DBF/ERA5",
            "extra_args": {"use_rho": True},
        },
    ]

    for model in models:
        name = model["name"]
        print(f"\n===== Running {name} ERA5 DA experiment =====")

        module = importlib.import_module(model["module"])
        run_fn = getattr(module, "run_multi_da_experiment")

        start_time = time.perf_counter()
        with working_directory(model["working_dir"]):
            run_fn(**base_kwargs, **model.get("extra_args", {}))
        elapsed = time.perf_counter() - start_time
        print(f"{name} wall time: {elapsed:.2f}s")

        result_path = repo_root / "results" / name / "ERA5" / "DA" / "multi.npy"
        if not result_path.exists():
            print(f"No results found at {result_path}, skipping metric summary.")
            continue

        da_states = np.load(result_path)
        metrics = compute_channel_metrics(da_states, groundtruth)

        print("Channel-wise metrics (averaged over time):")
        for metric_name, values in metrics.items():
            print(format_metric_row(metric_name, values))


if __name__ == "__main__":
    main()
