"""Unified training entry point for model/dataset combinations.

Example:
    python src/main.py --dataset era5 --model CAE_Koopman
"""
from __future__ import annotations

import argparse
import contextlib
import os
import subprocess
import sys
from pathlib import Path


@contextlib.contextmanager
def working_directory(path: Path):
    """Temporarily switch to ``path`` and then restore previous cwd."""

    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


DATASET_DIR_MAP = {
    "era5": "ERA5",
    "era5_high": "ERA5_High",
    "kmg": "KMG",
    "kol": "KMG",
    "kolmogorov": "KMG",
    "cylinder": "Cylinder",
}

DATASET_TRAINER_MAP = {
    "era5": "era5_trainer.py",
    "era5_high": "era5_high_trainer.py",
    "kmg": "kol_trainer.py",
    "kol": "kol_trainer.py",
    "kolmogorov": "kol_trainer.py",
    "cylinder": "cylinder_trainer.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a model trainer on a selected dataset")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. era5, kmg, cylinder")
    parser.add_argument("--model", required=True, help="Model name under src/models, e.g. CAE_Koopman")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print resolved paths/command without starting training",
    )
    return parser.parse_args()


def resolve_training_target(repo_root: Path, model_name: str, dataset_key: str) -> tuple[Path, str]:
    dataset_dir = DATASET_DIR_MAP.get(dataset_key)
    trainer_name = DATASET_TRAINER_MAP.get(dataset_key)
    if dataset_dir is None or trainer_name is None:
        supported = ", ".join(sorted(set(DATASET_DIR_MAP.keys())))
        raise ValueError(f"Unsupported dataset '{dataset_key}'. Supported: {supported}")

    model_dir = repo_root / "src" / "models" / model_name
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    trainer_dir = model_dir / dataset_dir
    if not trainer_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory '{dataset_dir}' not found for model '{model_name}': {trainer_dir}"
        )

    trainer_path = trainer_dir / trainer_name
    if not trainer_path.exists():
        available = sorted(path.name for path in trainer_dir.glob("*_trainer.py"))
        available_str = ", ".join(available) if available else "<none>"
        raise FileNotFoundError(
            f"Trainer script not found: {trainer_path}. Available in {trainer_dir}: {available_str}"
        )

    return trainer_dir, trainer_name


def main() -> None:
    args = parse_args()
    dataset_key = args.dataset.lower()

    repo_root = Path(__file__).resolve().parents[1]
    trainer_dir, trainer_name = resolve_training_target(repo_root, args.model, dataset_key)

    print(f"[INFO] model    : {args.model}")
    print(f"[INFO] dataset  : {dataset_key}")
    print(f"[INFO] workdir  : {trainer_dir}")
    print(f"[INFO] script   : {trainer_name}")

    if args.dry_run:
        print("[INFO] Dry-run enabled, trainer was not started.")
        return

    with working_directory(trainer_dir):
        cmd = [sys.executable, trainer_name]
        print(f"[INFO] Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
