#!/bin/bash -l

#SBATCH --job-name=kol_dbf
#SBATCH --partition=root
#SBATCH --qos=epic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH -e kol_dbf.err
#SBATCH -o kol_dbf.out

source /scratch_root/wy524/miniconda3/etc/profile.d/conda.sh
conda activate koopmanda

cd src/models/DBF/KMG/

python kol_trainer.py
