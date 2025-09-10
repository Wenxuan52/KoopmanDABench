#!/bin/bash -l

#SBATCH --job-name=dam_cae_k
#SBATCH --partition=dgxl_irp
#SBATCH --qos=dgxl_irp_high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH -e dam_cae_k.err
#SBATCH -o dam_cae_k.out

source /scratch_dgxl/wy524/miniconda3/etc/profile.d/conda.sh
conda activate irp-env

cd src/models/CAE_Koopman/Dam
python dam_trainer.py
