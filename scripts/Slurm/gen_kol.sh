#!/bin/bash -l

#SBATCH --job-name=gen_kol
#SBATCH --partition=dgxl_irp
#SBATCH --qos=dgxl_irp_low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH -e gen_kol_1000.err
#SBATCH -o gen_kol_1000.out

source /scratch_dgxl/wy524/miniconda3/etc/profile.d/conda.sh
conda activate irp-env

cd data/generate_data
python kolmogorov.py --re 1000
