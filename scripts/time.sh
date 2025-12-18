#!/bin/bash -l
#SBATCH --job-name=torch_h200_test
#SBATCH --partition=root
#SBATCH --qos=flash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=00:01:00
#SBATCH -e test.err
#SBATCH -o test.out

set -euo pipefail

echo "=== Job info ==="
hostname
date
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo

echo "=== nvidia-smi ==="
nvidia-smi
echo

# 如果你们集群会默认加载 Intel/oneAPI 模块，建议先清一下（没有 module 也不会报错）
module purge 2>/dev/null || true

# 激活 conda（把下面两行改成你服务器上 conda 的实际初始化方式）
source /scratch_root/wy524/miniconda3/etc/profile.d/conda.sh
conda activate koopmanda

echo "=== Python / pip ==="
which python
python -V
python -m pip -V
echo

echo "=== Torch smoke test ==="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    x = torch.randn(2048, 2048, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("GPU matmul ok, y:", y.shape, "mean:", y.mean().item())
else:
    raise SystemExit("CUDA not available in this job allocation")
PY
