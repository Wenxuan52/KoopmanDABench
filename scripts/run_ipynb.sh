#!/bin/bash -l

#SBATCH --job-name=run_ipynb
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH -e run_ipynb.err
#SBATCH -o run_ipynb.out

source /scratch_root/wy524/miniconda3/etc/profile.d/conda.sh
conda activate koopmanda

cd docs/

conda run -n koopmanda jupyter nbconvert --to notebook --execute --inplace ERA5_Example.ipynb --ExecutePreprocessor.timeout=86400