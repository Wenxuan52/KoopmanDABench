#!/bin/bash -l

#SBATCH --job-name=kol_linear
#SBATCH --partition=root
#SBATCH --qos=epic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH -e kol_linear_3loss_model.err
#SBATCH -o kol_linear_3loss_model.out

source /scratch_root/wy524/miniconda3/etc/profile.d/conda.sh
conda activate koopmanda

cd src/models/CAE_Linear/KMG
python kol_trainer.py
