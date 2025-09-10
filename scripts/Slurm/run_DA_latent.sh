#!/bin/bash -l

#SBATCH --job-name=run_DA_latent
#SBATCH --partition=dgxl_irp
#SBATCH --qos=dgxl_irp_high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:10:00
#SBATCH -e run_DA_latent.err
#SBATCH -o run_DA_latent.out

source /scratch_dgxl/wy524/miniconda3/etc/profile.d/conda.sh
conda activate irp-env

# Cylinder / Dam / ERA5
# DMD / CAE_DMD / CAE_Koopman / CAE_Linear / CAE_Weaklinear / CAE_MLP
cd src/models/DMD/Cylinder

python cylinder_DA_latent.py