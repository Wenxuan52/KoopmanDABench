#!/bin/bash -l

#SBATCH --job-name=run_DA
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH -e run_DA_full.err
#SBATCH -o run_DA_full.out

source /scratch_root/wy524/miniconda3/etc/profile.d/conda.sh
conda activate koopmanda

# Cylinder / Dam / ERA5
# DMD / CAE_DMD / CAE_Koopman / CAE_Linear / CAE_Weaklinear / CAE_MLP / discreteCGKN / DBF
# cd src/models/discreteCGKN/ERA5

# python direct_DA.py

cd src/assimilation/

python era5_full_observation.py

# era5_full_observation.py
# era5_intermittent_observation.py