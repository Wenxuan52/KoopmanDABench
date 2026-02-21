#!/bin/bash -l

#SBATCH --job-name=era5_high_DA
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH -e era5_high_DA.err
#SBATCH -o era5_high_DA.out

source /scratch_root/wy524/miniconda3/etc/profile.d/conda.sh
conda activate koopmanda

# ERA5_High
# DMD / KKR / KAE / WAE / AE / discreteCGKN / CGKN / DBF
cd src/models/DMD/ERA5_High

python era5_high_DA.py

# cd src/assimilation/

# python era5_full_observation.py

# era5_full_observation.py
# era5_intermittent_observation.py