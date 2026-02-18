#!/bin/bash -l

#SBATCH --job-name=kol_cgkn
#SBATCH --partition=root
#SBATCH --qos=epic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH -e kol_cgkn.err
#SBATCH -o kol_cgkn.out

source /scratch_root/wy524/miniconda3/etc/profile.d/conda.sh
conda activate koopmanda

cd src/models/CGKN/KMG/

python kol_trainer.py
