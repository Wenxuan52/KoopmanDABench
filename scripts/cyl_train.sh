#!/bin/bash -l

#SBATCH --job-name=cylinder
#SBATCH --partition=root
#SBATCH --qos=epic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH -e cylinder_train.err
#SBATCH -o cylinder_train.out

source /scratch_root/wy524/miniconda3/etc/profile.d/conda.sh
conda activate koopmanda

# Cylinder
# DMD / KKR / KAE / WAE / AE / discreteCGKN / CGKN / DBF
cd src/models/CGKN/Cylinder

python cylinder_trainer.py