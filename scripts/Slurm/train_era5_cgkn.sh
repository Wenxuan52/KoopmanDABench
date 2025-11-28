#!/bin/bash -l

#SBATCH --job-name=era5_cgkn
#SBATCH --partition=dgxl_irp
#SBATCH --qos=dgxl_irp_high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH -e era5_cgkn.err
#SBATCH -o era5_cgkn.out

source /scratch_dgxl/wy524/miniconda3/etc/profile.d/conda.sh
conda activate irp-env

cd src/models/CGKN/ERA5/

python era5_train.py
