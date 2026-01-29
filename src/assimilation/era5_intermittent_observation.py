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
from typing import Dict, Sequence

import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch

current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", ".."))
if src_directory not in sys.path:
    sys.path.append(src_directory)

from src.utils.Dataset import ERA5Dataset

SAVE_PREFIX = "interobs_direct_era5_"

@contextlib.contextmanager
def working_directory(path: Path):
    """Temporarily change the working directory."""

    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def load_groundtruth(window_length: int, start_T: int) -> np.ndarray:
    """Load a ground-truth ERA5 slice in the original value range."""

    repo_root = Path(__file__).resolve().parents[2]
    data_path = repo_root / "data/ERA5/ERA5_data/test_seq_state.h5"
    min_path = repo_root / "data/ERA5/ERA5_data/min_val.npy"
    max_path = repo_root / "data/ERA5/ERA5_data/max_val.npy"

    dataset = ERA5Dataset(
        data_path=str(data_path),
        seq_length=12,
        min_path=str(min_path),
        max_path=str(max_path),
    )

    total_frames = window_length + 1
    raw_data = dataset.data[start_T : start_T + total_frames, ...]
    groundtruth = torch.tensor(raw_data, dtype=torch.float32).permute(0, 3, 1, 2)
    return groundtruth[1:].numpy()  # align with assimilation horizon


def compute_channel_metrics(pred: np.ndarray, target: np.ndarray) -> Sequence[Dict[str, float]]:
    """Compute MSE, RRMSE, and SSIM for each channel over all steps."""

    assert pred.shape == target.shape, "Prediction and target shapes must match"
    num_channels = pred.shape[1]
    channel_metrics: list[Dict[str, float]] = []

    for c in range(num_channels):
        pred_c = pred[:, c]
        target_c = target[:, c]

        mse = float(np.mean((pred_c - target_c) ** 2))
        rrmse = float(np.sqrt(np.sum((pred_c - target_c) ** 2) / np.sum(target_c**2)))

        ssim_scores = []
        for t in range(pred.shape[0]):
            data_range = target_c[t].max() - target_c[t].min()
            if data_range > 0:
                ssim_scores.append(ssim(target_c[t], pred_c[t], data_range=float(data_range)))
            else:
                ssim_scores.append(1.0)

        channel_metrics.append(
            {
                "mse": mse,
                "rrmse": rrmse,
                "ssim": float(np.mean(ssim_scores)),
            }
        )

    return channel_metrics


def summarize_metrics(metrics: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate average metrics across channels."""

    mse_mean = float(np.mean([m["mse"] for m in metrics]))
    rrmse_mean = float(np.mean([m["rrmse"] for m in metrics]))
    ssim_mean = float(np.mean([m["ssim"] for m in metrics]))
    return {"mse": mse_mean, "rrmse": rrmse_mean, "ssim": ssim_mean}


def run_all_models():
    """Run data assimilation for all supported models with shared settings."""

    experiment_config = {
        "obs_ratio": 0.15,
        "obs_noise_std": 0.00,
        "observation_schedule": [0, 10, 20],
        "observation_variance": None,
        "window_length": 30,
        "num_runs": 5,
        "start_T": 0,
    }

    repo_root = Path(__file__).resolve().parents[2]
    models: Dict[str, Dict[str, object]] = {
        "CAE_Koopman": {"module": "src.models.CAE_Koopman.ERA5.era5_DA", "supports_prefix": True},
        "CAE_Linear": {"module": "src.models.CAE_Linear.ERA5.era5_DA", "supports_prefix": True},
        "CAE_Weaklinear": {"module": "src.models.CAE_Weaklinear.ERA5.era5_DA", "supports_prefix": True},
        "CAE_MLP": {"module": "src.models.CAE_MLP.ERA5.era5_DA", "supports_prefix": True},
        "DMD": {"module": "src.models.DMD.ERA5.era5_DA", "supports_prefix": True},
        # "discreteCGKN": {"module": "src.models.discreteCGKN.ERA5.era5_DA", "supports_prefix": True},
        "CGKN": {"module": "src.models.CGKN.ERA5.era5_DA", "supports_prefix": True},
        "DBF": {"module": "src.models.DBF.ERA5.era5_DA", "supports_prefix": True},
    }

    groundtruth = load_groundtruth(
        window_length=experiment_config["window_length"],
        start_T=experiment_config["start_T"],
    )

    for model_name, info in models.items():
        model_dir = repo_root / "src" / "models" / model_name / "ERA5"

        print(f"\n===== Running {model_name} =====")
        if not model_dir.exists():
            print(f"Model directory {model_dir} not found, skipping.")
            continue

        start_time = time.time()
        with working_directory(model_dir):
            module = importlib.import_module(info["module"])
            run_kwargs = dict(experiment_config)
            if info.get("supports_prefix"):
                run_kwargs["save_prefix"] = SAVE_PREFIX
            try:
                time_info = module.run_multi_da_experiment(model_name=model_name, **run_kwargs)

            except Exception as exc:
                print(f"{model_name} run failed: {exc}")
                continue
            if time_info is not None:
                print(f"{model_name} time info: {time_info}")
        elapsed = time.time() - start_time
        print(f"{model_name} total wall time: {elapsed:.2f}s")

        result_dir = repo_root / "results" / model_name / "ERA5" / "DA"
        prefix = SAVE_PREFIX if info.get("supports_prefix") else ""
        multi_path = result_dir / f"{prefix}multi.npy"
        if not multi_path.exists():
            print(f"{multi_path} not found; unable to compute metrics.")
            continue

        da_states = np.load(multi_path)
        if da_states.shape != groundtruth.shape:
            print(
                f"Shape mismatch for {model_name}: predicted {da_states.shape}, "
                f"ground truth {groundtruth.shape}."
            )
            continue

        metrics = compute_channel_metrics(da_states, groundtruth)
        aggregate = summarize_metrics(metrics)

        print("Channel-wise metrics (MSE, RRMSE, SSIM):")
        for idx, channel_metric in enumerate(metrics):
            print(
                f"  Channel {idx}: "
                f"MSE={channel_metric['mse']:.6f}, "
                f"RRMSE={channel_metric['rrmse']:.6f}, "
                f"SSIM={channel_metric['ssim']:.6f}"
            )
        print(
            f"Average: MSE={aggregate['mse']:.6f}, "
            f"RRMSE={aggregate['rrmse']:.6f}, SSIM={aggregate['ssim']:.6f}"
        )


if __name__ == "__main__":
    run_all_models()
