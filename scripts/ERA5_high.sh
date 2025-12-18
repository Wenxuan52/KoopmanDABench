#!/bin/bash -l
#SBATCH --job-name=ERA5_high
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH -e ERA5_high.err
#SBATCH -o ERA5_high.out


source /scratch_root/wy524/miniconda3/etc/profile.d/conda.sh
conda activate koopmanda

cd data/ERA5_high
python process_weatherbench.py