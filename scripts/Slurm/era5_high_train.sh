#!/bin/bash -l

#SBATCH --job-name=era5_high
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH -e era5_high_train1.err
#SBATCH -o era5_high_train1.out

source /scratch_root/wy524/miniconda3/etc/profile.d/conda.sh
conda activate koopmanda

# ERA5_High
# DMD / CAE_Koopman / CAE_Linear / CAE_Weaklinear / CAE_MLP / discreteCGKN / CGKN / DBF
cd src/models/DBF/ERA5_High

python era5_high_trainer.py