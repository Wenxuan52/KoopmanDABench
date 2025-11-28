#!/bin/bash -l

#SBATCH --job-name=cylinder_dbf
#SBATCH --partition=dgxl_irp
#SBATCH --qos=dgxl_irp_high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -e cylinder_dbf.err
#SBATCH -o cylinder_dbf.out

source /scratch_dgxl/wy524/miniconda3/etc/profile.d/conda.sh
conda activate irp-env

cd src/models/DBF/Cylinder/

python cylinder_trainer.py
