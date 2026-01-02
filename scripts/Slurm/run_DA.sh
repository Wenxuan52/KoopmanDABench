#!/bin/bash -l

#SBATCH --job-name=run_DA
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=9:00:00
#SBATCH -e run_DA.err
#SBATCH -o run_DA.out

source /scratch_root/wy524/miniconda3/etc/profile.d/conda.sh
conda activate koopmanda

# Cylinder / Dam / ERA5
# DMD / CAE_DMD / CAE_Koopman / CAE_Linear / CAE_Weaklinear / CAE_MLP / discreteCGKN / DBF
# cd src/models/discreteCGKN/ERA5

# python direct_DA.py

# cd src/assimilation/

# python era5_intermittent_observation.py