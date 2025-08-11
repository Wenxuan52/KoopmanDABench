#!/bin/bash -l

#SBATCH --job-name=run_DA
#SBATCH --partition=dgxl_irp
#SBATCH --qos=dgxl_irp_high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH -e run_DA.err
#SBATCH -o run_DA.out

source /scratch_dgxl/wy524/miniconda3/etc/profile.d/conda.sh
conda activate irp-env

# Cylinder / Dam
cd src/models/CAE_MLP/Dam

python dam_DA.py