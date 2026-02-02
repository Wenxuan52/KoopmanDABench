import argparse
import json
import os
import random
import sys
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import yaml

current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
sys.path.append(src_directory)

from src.utils.Dataset import KolDynamicsDataset
from src.models.DMD.base import TorchDMD


def set_seed(seed: int = 42) -> None:
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Set random seed to {seed}")


def get_memory_usage() -> str:
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        return f"Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB"
    return "CPU mode"


def load_config(config_path: str) -> Dict[str, object]:
    with open(config_path, "r") as file:
        return yaml.safe_load(file) or {}


def _ensure_5d(sequence: torch.Tensor) -> torch.Tensor:
    """Ensure sequence is [B, T, C, H, W]."""
    if sequence.ndim == 3:
        # [T, H, W] -> [1, T, 1, H, W]
        sequence = sequence.unsqueeze(0).unsqueeze(2)
    elif sequence.ndim == 4:
        # [T, C, H, W] or [B, T, H, W]
        if sequence.shape[1] <= 4 and sequence.shape[-1] != sequence.shape[1]:
            # Heuristic: assume [T, C, H, W]
            sequence = sequence.unsqueeze(0)
        else:
            # Assume [B, T, H, W]
            sequence = sequence.unsqueeze(2)
    elif sequence.ndim == 5:
        return sequence
    else:
        raise ValueError(f"Unsupported sequence shape: {sequence.shape}")
    return sequence


def _extract_sequences(dataset: KolDynamicsDataset) -> torch.Tensor:
    """Extract normalized sequences from dataset.

    Preference order:
    1) Use dataset.data if available and normalize using dataset.normalize (consistent with CAE/DMD pipelines).
    2) Fall back to dataset[0] tuple or tensor.
    """
    if hasattr(dataset, "data"):
        data = dataset.data
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        if data.ndim == 4:
            data = data.unsqueeze(2)
        if data.ndim != 5:
            raise ValueError(f"Expected dataset.data with 5 dims [N,T,C,H,W], got {data.shape}")
        normalized = dataset.normalize(data)
        return normalized

    sample = dataset[0]
    if isinstance(sample, tuple):
        sequence = sample[0]
    else:
        sequence = sample
    sequence = sequence.float()
    sequence = _ensure_5d(sequence)
    return sequence


def _build_pairs(sequences: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build (X, Y) snapshot pairs from [B, T, C, H, W] sequences."""
    if sequences.ndim != 5:
        raise ValueError(f"Expected sequences of shape [B,T,C,H,W], got {sequences.shape}")

    x_snapshots = sequences[:, :-1]
    y_snapshots = sequences[:, 1:]
    x_pairs = x_snapshots.reshape(-1, *x_snapshots.shape[2:])
    y_pairs = y_snapshots.reshape(-1, *y_snapshots.shape[2:])
    return x_pairs, y_pairs


def _subsample_pairs(
    x_pairs: torch.Tensor,
    y_pairs: torch.Tensor,
    max_pairs: Optional[int],
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not max_pairs or max_pairs <= 0:
        return x_pairs, y_pairs

    num_pairs = x_pairs.shape[0]
    if num_pairs <= max_pairs:
        return x_pairs, y_pairs

    rng = np.random.default_rng(seed)
    indices = rng.choice(num_pairs, size=max_pairs, replace=False)
    indices = torch.from_numpy(indices)
    x_pairs = x_pairs[indices]
    y_pairs = y_pairs[indices]
    print(f"[INFO] Subsampled pairs: {num_pairs} -> {max_pairs}")
    return x_pairs, y_pairs


def prepare_dmd_data(
    dataset: KolDynamicsDataset,
    max_pairs: Optional[int],
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare DMD data from dataset into [D, num_pairs]."""
    sequences = _extract_sequences(dataset)
    sequences = _ensure_5d(sequences)

    x_pairs, y_pairs = _build_pairs(sequences)
    x_pairs, y_pairs = _subsample_pairs(x_pairs, y_pairs, max_pairs, seed)

    num_pairs, channels, height, width = x_pairs.shape
    d = channels * height * width

    x_flat = x_pairs.reshape(num_pairs, d).transpose(0, 1).contiguous()
    y_flat = y_pairs.reshape(num_pairs, d).transpose(0, 1).contiguous()

    print(f"[INFO] Prepared pairs: {num_pairs}")
    print(f"[INFO] Snapshot shape: (C,H,W)=({channels},{height},{width})")
    print(f"[INFO] Flattened D: {d}")
    return x_flat, y_flat


def save_artifacts(save_folder: str, config: Dict[str, object], metrics: Dict[str, object]) -> None:
    os.makedirs(save_folder, exist_ok=True)
    config_path = os.path.join(save_folder, "config.yaml")
    with open(config_path, "w") as handle:
        yaml.safe_dump(config, handle, default_flow_style=False)
    metrics_path = os.path.join(save_folder, "metrics.json")
    with open(metrics_path, "w") as handle:
        json.dump(metrics, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kolmogorov DMD Trainer")
    parser.add_argument(
        "--config",
        type=str,
        default="../../../../configs/DMD_KOL.yaml",
        help="Path to config YAML",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--max_pairs", type=int, default=None, help="Subsample pairs for memory control")
    parser.add_argument("--data_path", type=str, default=None, help="Override data_path from config")
    return parser.parse_args()


def main() -> None:
    start_time = time.time()

    args = parse_args()
    config = load_config(args.config)

    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    set_seed(seed)
    torch.set_default_dtype(torch.float32)

    max_pairs = args.max_pairs if args.max_pairs is not None else config.get("max_pairs", 0)

    # Device setup
    use_gpu = bool(config.get("use_gpu", True))
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"[INFO] Using GPU: {torch.cuda.get_device_properties(0).name}")
        print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
    else:
        device = torch.device("cpu")
        print(f"[INFO] Using CPU device")
        if use_gpu and not torch.cuda.is_available():
            print("[WARNING] GPU requested but not available, falling back to CPU")

    print(f"[INFO] Memory usage at start: {get_memory_usage()}")
    print("[INFO] Starting Kolmogorov DMD Model Training")
    print(f"[INFO] Configuration: {config}")

    data_path = args.data_path or config.get("data_path") or "../../../../data/kol/kolmogorov_train_data.npy"
    seq_length = int(config.get("seq_length", 5))

    print("\n" + "=" * 50)
    print("Fitting FORWARD MODEL")
    print("=" * 50)

    print("[INFO] Loading dataset...")
    kol_train_dataset = KolDynamicsDataset(
        data_path=data_path,
        seq_length=seq_length,
        mean=None,
        std=None,
    )

    print(f"[INFO] Memory usage after data loading: {get_memory_usage()}")

    print("[INFO] Preparing DMD data...")
    x_train, y_train = prepare_dmd_data(kol_train_dataset, max_pairs=max_pairs, seed=seed)

    print(f"X_train shape: {x_train.shape} on {x_train.device}")
    print(f"y_train shape: {y_train.shape} on {y_train.device}")
    print(f"[INFO] Memory usage after data preparation: {get_memory_usage()}")

    print("[INFO] Initializing DMD model...")
    dmd = TorchDMD(svd_rank=config.get("rank", 0), device=device)

    print(f"[INFO] Memory usage after model initialization: {get_memory_usage()}")

    print("[INFO] Fitting DMD model...")
    dmd.fit(x_train, y_train)

    print(f"[INFO] Memory usage after fitting: {get_memory_usage()}")

    save_folder = config.get("save_folder", "../../../../results/DMD/KMG")
    save_path = os.path.join(save_folder, "dmd_model.pth")
    os.makedirs(save_folder, exist_ok=True)

    print("[INFO] Saving DMD model...")
    dmd.save_dmd(save_path)

    end_time = time.time()
    runtime_sec = end_time - start_time

    metrics = {
        "runtime_sec": runtime_sec,
        "num_pairs": int(x_train.shape[1]),
        "rank": int(config.get("rank", 0)),
        "device": str(device),
        "max_pairs": int(max_pairs) if max_pairs else 0,
    }
    save_artifacts(save_folder, config, metrics)

    print("[INFO] DMD fit successfully completed!")
    print(f"[INFO] Final memory usage: {get_memory_usage()}")
    print(f"[INFO] Total Runtime: {runtime_sec:.2f} seconds")
    print(f"[INFO] Model saved to: {save_path}")


if __name__ == "__main__":
    main()
