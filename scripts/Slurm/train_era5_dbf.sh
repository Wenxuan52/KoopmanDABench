#!/bin/bash -l

#SBATCH --job-name=era5_dbf
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH -e era5_dbf.err
#SBATCH -o era5_dbf.out

source /scratch_root/wy524/miniconda3/etc/profile.d/conda.sh
conda activate koopmanda

cd src/models/DBF/ERA5/

python era5_trainer.py
