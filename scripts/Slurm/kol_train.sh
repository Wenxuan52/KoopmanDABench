#!/bin/bash -l

#SBATCH --job-name=kol
#SBATCH --partition=root
#SBATCH --qos=epic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH -e kol_train.err
#SBATCH -o kol_train.out

source /scratch_root/wy524/miniconda3/etc/profile.d/conda.sh
conda activate koopmanda

# Cylinder / KMG / ERA5
# DMD / CAE_Koopman / CAE_Linear / CAE_Weaklinear / CAE_MLP / discreteCGKN / CGKN / DBF
cd src/models/CGKN/KMG

python kol_trainer.py