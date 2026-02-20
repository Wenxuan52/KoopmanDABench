#!/bin/bash -l

#SBATCH --job-name=kol_DA
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH -e kol_DA5.err
#SBATCH -o kol_DA5.out

source /scratch_root/wy524/miniconda3/etc/profile.d/conda.sh
conda activate koopmanda

# Cylinder / ERA5
# DMD / CAE_Koopman / CAE_Linear / CAE_Weaklinear / CAE_MLP / discreteCGKN / CGKN / DBF
cd src/models/DMD/KMG

python kol_DA.py

# cd src/assimilation/

# python era5_full_observation.py

# era5_full_observation.py
# era5_intermittent_observation.py