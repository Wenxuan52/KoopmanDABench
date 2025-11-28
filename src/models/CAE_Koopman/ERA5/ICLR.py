import os
import sys
import torch
import numpy as np
from tqdm import tqdm

from era5_model_FTF import ERA5_C_FORWARD

# Register project root
current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
sys.path.append(src_directory)

from src.utils.Dataset import ERA5Dataset


def rollout_prediction(forward_model: ERA5_C_FORWARD, initial_state: torch.Tensor, n_steps: int) -> torch.Tensor:
    """Roll forward Koopman model for n_steps starting from normalized initial state."""
    preds = []
    with torch.no_grad():
        z = forward_model.K_S(initial_state.unsqueeze(0))
        for _ in range(n_steps):
            z = forward_model.latent_forward(z)
            next_state = forward_model.K_S_preimage(z)
            preds.append(next_state)
    return torch.cat(preds, dim=0)  # (T, C, H, W)


if __name__ == "__main__":
    # ========== Config ==========
    prediction_steps = 50
    model_name = "CAE_Koopman"
    model_dir = f"../../../../results/{model_name}/ERA5/3loss_model"

    # ========== Load dataset ==========
    era5_test_dataset = ERA5Dataset(
        data_path="../../../../data/ERA5/ERA5_data/test_seq_state.h5",
        seq_length=12,
        min_path="../../../../data/ERA5/ERA5_data/min_val.npy",
        max_path="../../../../data/ERA5/ERA5_data/max_val.npy",
    )
    denorm = era5_test_dataset.denormalizer()
    raw_test_data = era5_test_dataset.data  # (N, H, W, C)

    # ========== Initialize model ==========
    forward_model = ERA5_C_FORWARD()
    forward_model.load_state_dict(
        torch.load(os.path.join(model_dir, "forward_model.pt"), weights_only=True, map_location="cpu")
    )
    forward_model.C_forward = torch.load(
        os.path.join(model_dir, "C_forward.pt"), weights_only=True, map_location="cpu"
    )
    forward_model.eval()

    # ========== Rollout from test[0] ==========
    init_state = torch.tensor(raw_test_data[0, ...], dtype=torch.float32).permute(2, 0, 1)
    norm_init = era5_test_dataset.normalize(init_state.unsqueeze(0))[0]

    print(f"Rolling out {prediction_steps} steps from test[0] ...")
    rollout = rollout_prediction(forward_model, norm_init, prediction_steps)
    rollout_denorm = denorm(rollout).cpu().numpy()  # (T, C, H, W)

    # ========== Save ==========
    save_dir = f"../../../../results/{model_name}/figures"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "era5_ICLR.npy")

    np.save(save_path, rollout_denorm)
    print(f"\nSaved rollout result to: {save_path}")
