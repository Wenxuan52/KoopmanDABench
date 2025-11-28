#!/bin/bash -l

#SBATCH --job-name=dam_cae_mi
#SBATCH --partition=dgxl_irp
#SBATCH --qos=dgxl_irp_high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH -e dam_cae_mi_0gamma.err
#SBATCH -o dam_cae_mi_0gamma.out

source /scratch_dgxl/wy524/miniconda3/etc/profile.d/conda.sh
conda activate irp-env

cd src/models/CAE_MI/Dam
python dam_trainer.py
