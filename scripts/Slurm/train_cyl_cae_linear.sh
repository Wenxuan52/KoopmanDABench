#!/bin/bash -l

#SBATCH --job-name=cylinder_linear
#SBATCH --partition=dgxl_irp
#SBATCH --qos=dgxl_irp_low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH -e cylinder_linear_jointly.err
#SBATCH -o cylinder_linear_jointly.out

source /scratch_dgxl/wy524/miniconda3/etc/profile.d/conda.sh
conda activate irp-env

cd src/models/CAE_Linear
python cylinder_trainer.py
