#!/bin/bash -l

#SBATCH --job-name=era5_DA
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH -e era5_DA.err
#SBATCH -o era5_DA.out

source /scratch_root/wy524/miniconda3/etc/profile.d/conda.sh
conda activate koopmanda

# Cylinder / Dam / ERA5
# DMD / CAE_Koopman / CAE_Linear / CAE_Weaklinear / CAE_MLP / discreteCGKN / CGKN / DBF
# cd src/models/CGKN/ERA5

# python era5_DA.py

cd src/assimilation/

python era5_intermittent_observation.py

# era5_full_observation.py
# era5_intermittent_observation.py