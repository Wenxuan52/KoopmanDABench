#!/bin/bash -l

#SBATCH --job-name=run_ipynb
#SBATCH --partition=dgxl_irp
#SBATCH --qos=dgxl_irp_low
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH -e run_ipynb.err
#SBATCH -o run_ipynb.out

source /scratch_dgxl/wy524/miniconda3/etc/profile.d/conda.sh
conda activate irp-env

cd src/models/CAE_DMD/KMG

jupyter nbconvert --to notebook --execute --inplace kol_DA_randob.ipynb --ExecutePreprocessor.timeout=86400