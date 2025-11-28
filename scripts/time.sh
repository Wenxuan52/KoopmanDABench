#!/bin/bash -l

#SBATCH --job-name=time_test
#SBATCH --partition=dgxl_irp
#SBATCH --qos=dgxl_irp_high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH -e time_test.err
#SBATCH -o time_test.out

source /scratch_dgxl/wy524/miniconda3/etc/profile.d/conda.sh
conda activate irp-env

# CAE_Koopman CAE_MI CAE_Linear CAE_MLP
cd src/models/CAE_MLP/ERA5
python era5_trainer.py
