#!/bin/bash -l

##############################
#      Kolmogorov Trainer    #
##############################

#SBATCH --job-name=kol_dmd
#SBATCH --partition=dgxl_irp
#SBATCH --qos=dgxl_irp_low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH -e kol_dmd_trans.err
#SBATCH -o kol_dmd_trans.out

source /scratch_dgxl/wy524/miniconda3/etc/profile.d/conda.sh
conda activate irp-env

cd src/models/DMD
python kol_trainer.py
