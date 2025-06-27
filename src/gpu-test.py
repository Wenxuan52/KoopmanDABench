#!/usr/bin/env python3
import os
import torch

def main():
    # 环境变量
    cuda_env = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    print(f"CUDA_VISIBLE_DEVICES = {cuda_env}")

    # PyTorch 检测
    cuda_avail = torch.cuda.is_available()
    print(f"torch.cuda.is_available() → {cuda_avail}")

    if cuda_avail:
        n_gpu = torch.cuda.device_count()
        print(f"Number of GPUs detected by Torch: {n_gpu}")
        for i in range(n_gpu):
            name = torch.cuda.get_device_name(i)
            cap  = torch.cuda.get_device_capability(i)
            print(f" • Device {i}: {name}, Compute Capability {cap[0]}.{cap[1]}")
    else:
        print("No GPUs detected by PyTorch.")

    try:
        import subprocess
        print("\nRunning `nvidia-smi`:\n")
        print(subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.STDOUT).decode())
    except Exception as e:
        print("\nCould not run nvidia-smi:", e)

if __name__ == "__main__":
    main()
