#!/bin/bash -l

#SBATCH --job-name=era5_discretecgkn
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH -e era5_discretecgkn.err
#SBATCH -o era5_discretecgkn.out

source /scratch_root/wy524/miniconda3/etc/profile.d/conda.sh
conda activate koopmanda

cd src/models/discreteCGKN/ERA5/

python era5_train.py --train_stage2
