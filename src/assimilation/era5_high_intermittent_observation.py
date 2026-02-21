"""ERA5-High intermittent-observation DA experiments across multiple models."""
from __future__ import annotations

import contextlib
import importlib
import os
import sys
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

from src.utils.Dataset import ERA5HighDataset

SAVE_PREFIX = "interobs_direct_era5_high_"


@contextlib.contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def load_groundtruth(window_length: int, start_T: int) -> np.ndarray:
    repo_root = Path(__file__).resolve().parents[2]
    dataset = ERA5HighDataset(
        data_path=str(repo_root / "data/ERA5_high/raw_data/weatherbench_test.h5"),
        seq_length=12,
        min_path=str(repo_root / "data/ERA5_high/raw_data/era5high_240x121_min.npy"),
        max_path=str(repo_root / "data/ERA5_high/raw_data/era5high_240x121_max.npy"),
    )

    total_frames = window_length + 1
    raw_data = dataset.data[start_T : start_T + total_frames, ...]
    groundtruth = torch.tensor(raw_data, dtype=torch.float32).permute(0, 3, 1, 2)
    return groundtruth[1:].numpy()


def compute_channel_metrics(pred: np.ndarray, target: np.ndarray) -> Sequence[Dict[str, float]]:
    assert pred.shape == target.shape, "Prediction and target shapes must match"
    num_channels = pred.shape[1]
    channel_metrics: list[Dict[str, float]] = []

    for c in range(num_channels):
        pred_c = pred[:, c]
        target_c = target[:, c]
        mse = float(np.mean((pred_c - target_c) ** 2))
        denom = float(np.sum(target_c**2))
        rrmse = float(np.sqrt(np.sum((pred_c - target_c) ** 2) / max(denom, 1e-12)))

        ssim_scores = []
        for t in range(pred.shape[0]):
            data_range = target_c[t].max() - target_c[t].min()
            ssim_scores.append(ssim(target_c[t], pred_c[t], data_range=float(data_range)) if data_range > 0 else 1.0)

        channel_metrics.append({"mse": mse, "rrmse": rrmse, "ssim": float(np.mean(ssim_scores))})

    return channel_metrics


def summarize_metrics(metrics: Sequence[Dict[str, float]]) -> Dict[str, float]:
    return {
        "mse": float(np.mean([m["mse"] for m in metrics])),
        "rrmse": float(np.mean([m["rrmse"] for m in metrics])),
        "ssim": float(np.mean([m["ssim"] for m in metrics])),
    }


def run_all_models():
    experiment_config = {
        "obs_ratio": 0.15,
        "obs_noise_std": 0.00,
        "observation_schedule": [0, 5, 10, 15, 20, 25],
        "observation_variance": None,
        "window_length": 30,
        "num_runs": 5,
        "start_T": 0,
    }

    repo_root = Path(__file__).resolve().parents[2]
    models: Dict[str, Dict[str, object]] = {
        "KKR": {"module": "src.models.KKR.ERA5_High.era5_high_multi_DA", "supports_prefix": True},
        "KAE": {"module": "src.models.KAE.ERA5_High.era5_high_multi_DA", "supports_prefix": True},
        "WAE": {"module": "src.models.WAE.ERA5_High.era5_high_multi_DA", "supports_prefix": True},
        "AE": {"module": "src.models.AE.ERA5_High.era5_high_multi_DA", "supports_prefix": True},
        "DMD": {"module": "src.models.DMD.ERA5_High.era5_high_multi_DA", "supports_prefix": True},
        "CGKN": {"module": "src.models.CGKN.ERA5_High.era5_high_multi_DA", "supports_prefix": True},
        "DBF": {"module": "src.models.DBF.ERA5_High.era5_high_multi_DA", "supports_prefix": True},
    }

    groundtruth = load_groundtruth(window_length=experiment_config["window_length"], start_T=experiment_config["start_T"])

    for model_name, info in models.items():
        model_dir = repo_root / "src" / "models" / model_name / "ERA5_High"
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
                time_info = module.run_multi_continuous_da_experiment(model_name=model_name, **run_kwargs)
            except Exception as exc:
                print(f"{model_name} run failed: {exc}")
                continue
            if time_info is not None:
                print(f"{model_name} time info: {time_info}")
        print(f"{model_name} total wall time: {time.time() - start_time:.2f}s")

        result_dir = repo_root / "results" / model_name / "ERA5_High" / "DA"
        multi_path = result_dir / f"{SAVE_PREFIX if info.get('supports_prefix') else ''}multi.npy"
        if not multi_path.exists():
            print(f"{multi_path} not found; unable to compute metrics.")
            continue

        da_states = np.load(multi_path)
        if da_states.shape != groundtruth.shape:
            print(f"Shape mismatch for {model_name}: predicted {da_states.shape}, ground truth {groundtruth.shape}.")
            continue

        metrics = compute_channel_metrics(da_states, groundtruth)
        aggregate = summarize_metrics(metrics)
        print("Channel-wise metrics (MSE, RRMSE, SSIM):")
        for idx, channel_metric in enumerate(metrics):
            print(f"  Channel {idx}: MSE={channel_metric['mse']:.6f}, RRMSE={channel_metric['rrmse']:.6f}, SSIM={channel_metric['ssim']:.6f}")
        print(f"Average: MSE={aggregate['mse']:.6f}, RRMSE={aggregate['rrmse']:.6f}, SSIM={aggregate['ssim']:.6f}")


if __name__ == "__main__":
    run_all_models()
