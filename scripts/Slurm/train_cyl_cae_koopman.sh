#!/bin/bash -l

#SBATCH --job-name=cylinder_cae_k_3loss_model
#SBATCH --partition=dgxl_irp
#SBATCH --qos=dgxl_irp_high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH -e cylinder_cae_k_3loss_model.err
#SBATCH -o cylinder_cae_k_3loss_model.out

source /scratch_dgxl/wy524/miniconda3/etc/profile.d/conda.sh
conda activate irp-env

cd src/models/CAE_Koopman/Cylinder
python cylinder_trainer.py
