#!/bin/bash -l

#SBATCH --job-name=cylinder_dmd
#SBATCH --partition=dgxl_irp
#SBATCH --qos=dgxl_irp_low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH -e cylinder_dmd.err
#SBATCH -o cylinder_dmd.out

# ===== 环境配置 =====

source /scratch_dgxl/wy524/miniconda3/etc/profile.d/conda.sh
conda activate irp-env

# ===== 执行训练脚本 =====
cd src/models/DMD
python cylinder_trainer.py
