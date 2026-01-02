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

from src.models.CAE_Koopman.ERA5.era5_DA import run_multi_da_experiment as Koopman_DA
from src.models.CAE_Linear.ERA5.era5_DA import run_multi_da_experiment as LINEAR_DA
from src.models.CAE_Weaklinear.ERA5.era5_DA import run_multi_da_experiment as WEAKLINEAR_DA
from src.models.CAE_MLP.ERA5.era5_DA import run_multi_da_experiment as MLP_DA
from src.models.DMD.ERA5.era5_DA import run_multi_da_experiment as DMD_DA
from src.models.discreteCGKN.ERA5.era5_DA import run_multi_da_experiment as DISCRETECGKN_DA
from src.models.DBF.ERA5.era5_DA import run_multi_da_experiment as DBF_DA


