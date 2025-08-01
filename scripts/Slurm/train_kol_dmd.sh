#!/bin/bash -l

#SBATCH --job-name=kol_dmd
#SBATCH --partition=dgxl_irp
#SBATCH --qos=dgxl_irp_low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH -e kol_dmd.err
#SBATCH -o kol_dmd.out

source /scratch_dgxl/wy524/miniconda3/etc/profile.d/conda.sh
conda activate irp-env

cd src/models/DMD/KMG
python kol_trainer.py
