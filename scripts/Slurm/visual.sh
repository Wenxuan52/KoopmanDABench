#!/bin/bash -l

#SBATCH --job-name=visual
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH -e visual.err
#SBATCH -o visual.out

source /scratch_root/wy524/miniconda3/etc/profile.d/conda.sh
conda activate koopmanda

# =========================
# Fix Matplotlib + Cartopy cache dirs (must be writable)
# =========================
CACHE_BASE="/scratch_root/wy524/.cache/${SLURM_JOB_ID}"
mkdir -p "${CACHE_BASE}/mpl" "${CACHE_BASE}/cartopy"

export MPLCONFIGDIR="${CACHE_BASE}/mpl"
export CARTOPY_DATA_DIR="${CACHE_BASE}/cartopy"

# 可选：某些库也会写 XDG cache/config，稳一点可以加
export XDG_CACHE_HOME="${CACHE_BASE}/xdg_cache"
export XDG_CONFIG_HOME="${CACHE_BASE}/xdg_config"
mkdir -p "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}"

cd /scratch_root/wy524/irp-test-framework/src/plot/
python era5_gif.py
