#!/bin/bash -l

#SBATCH --job-name=kol_dmd
#SBATCH --partition=root
#SBATCH --qos=epic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH -e kol_dmd_3loss_model.err
#SBATCH -o kol_dmd_3loss_model.out

source /scratch_root/wy524/miniconda3/etc/profile.d/conda.sh
conda activate koopmanda

cd src/models/DMD/KMG
python kol_trainer.py
