import argparse
import os
import sys
import time
from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
sys.path.append(src_directory)

from src.models.CAE_Koopman.dabase import CAEKoopmanDABase
from src.models.CAE_Koopman.ERA5.era5_model import ERA5_C_FORWARD
from src.utils.Dataset import ERA5Dataset


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        return "cuda"
    return "cpu"


def load_forward_model(model_dir: str, device: str) -> ERA5_C_FORWARD:
    model = ERA5_C_FORWARD()
    model.load_state_dict(
        torch.load(os.path.join(model_dir, "forward_model.pt"), map_location=device, weights_only=True)
    )
    model.C_forward = torch.load(os.path.join(model_dir, "C_forward.pt"), map_location=device, weights_only=True)
    model.eval()
    model.to(device)
    return model


def build_observation_operator(mask_indices: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    def observation_operator(decoded_state: torch.Tensor) -> torch.Tensor:
        flat = decoded_state.view(decoded_state.shape[0], -1)
        return flat[:, mask_indices]

    return observation_operator


def create_observation_mask(
    observation_ratio: float, state_shape: Tuple[int, int, int], seed: int = 42
) -> torch.Tensor:
    set_seed(seed)
    total_points = int(np.prod(state_shape))
    obs_count = int(total_points * observation_ratio)
    indices = torch.randperm(total_points)[:obs_count]
    return indices


def prepare_observations(
    normalized_frames: torch.Tensor, mask_indices: torch.Tensor
) -> List[torch.Tensor]:
    observations = []
    for frame in normalized_frames:
        flat = frame.view(-1)
        observations.append(flat[mask_indices])
    return observations


def run_iterative_assimilation(
    da_solver: CAEKoopmanDABase,
    observations: List[torch.Tensor],
    initial_latent: torch.Tensor,
    background_cov: torch.Tensor,
    observation_noise: float,
    tol: float = 1e-4,
    max_iter: int = 5,
) -> Tuple[dict, int, float]:
    current_latent = initial_latent.clone()
    runtime_accum = 0.0
    for iteration in range(1, max_iter + 1):
        result = da_solver.run_assimilation(
            observations=observations,
            background_initial_state=current_latent,
            background_covariance=background_cov,
            observation_noise=observation_noise,
            background_is_latent=True,
        )
        runtime_accum += float(result["runtime_seconds"])
        next_latent = result["analysis_latents"][0].to(current_latent.device)
        diff_norm = torch.norm(next_latent - current_latent)
        ref_norm = torch.norm(current_latent) + 1e-8
        if (diff_norm / ref_norm).item() < tol:
            return result, iteration, runtime_accum
        current_latent = next_latent
    return result, max_iter, runtime_accum


def compute_mse_per_step(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2, dim=(1, 2, 3))


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct DA for CAE_Koopman on ERA5.")
    parser.add_argument("--model_dir", type=str, default="../../../../results/CAE_Koopman/ERA5/3loss_model")
    parser.add_argument("--data_path", type=str, default="../../../../data/ERA5/ERA5_data/test_seq_state.h5")
    parser.add_argument("--min_path", type=str, default="../../../../data/ERA5/ERA5_data/min_val.npy")
    parser.add_argument("--max_path", type=str, default="../../../../data/ERA5/ERA5_data/max_val.npy")
    parser.add_argument("--window", type=int, default=50, help="DA window length.")
    parser.add_argument("--observation_ratio", type=float, default=0.15)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--background_cov", type=float, default=1e-3)
    parser.add_argument("--obs_noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="../../../../results/CAE_Koopman/ERA5/DA")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    era5_dataset = ERA5Dataset(
        data_path=args.data_path,
        seq_length=12,
        min_path=args.min_path,
        max_path=args.max_path,
    )
    denorm = era5_dataset.denormalizer()
    raw_data = torch.from_numpy(era5_dataset.data).permute(0, 3, 1, 2)  # (N, C, H, W)

    forward_model = load_forward_model(args.model_dir, device)
    mask_indices = create_observation_mask(args.observation_ratio, raw_data.shape[1:], seed=args.seed)
    observation_operator = build_observation_operator(mask_indices.to(device))

    da_solver = CAEKoopmanDABase(
        dynamics_matrix=forward_model.C_forward,
        decoder=forward_model.K_S_preimage,
        observation_operator=observation_operator,
        encoder=forward_model.K_S,
        device=device,
    )

    # Prepare ground truth and observations
    window_end = 1 + args.window
    gt_frames_physical = raw_data[1 : window_end + 1]  # frames 1..window
    gt_frames_norm = era5_dataset.normalize(gt_frames_physical)
    observations = prepare_observations(gt_frames_norm, mask_indices.to(gt_frames_norm.device))
    observations = [obs.to(device) for obs in observations]

    initial_state_norm = era5_dataset.normalize(raw_data[0].unsqueeze(0))[0].to(device)
    with torch.no_grad():
        z0 = forward_model.K_S(initial_state_norm.unsqueeze(0))
        background_latent = forward_model.latent_forward(z0).squeeze(0)
    background_cov = torch.eye(background_latent.shape[-1], device=device) * args.background_cov

    per_step_mse_runs: List[torch.Tensor] = []
    overall_mse_runs: List[float] = []
    runtimes: List[float] = []
    analysis_sample = None

    os.makedirs(args.save_dir, exist_ok=True)

    for trial in range(args.num_trials):
        start_time = time.perf_counter()
        result, iters, runtime_used = run_iterative_assimilation(
            da_solver=da_solver,
            observations=observations,
            initial_latent=background_latent,
            background_cov=background_cov,
            observation_noise=args.obs_noise,
            tol=1e-4,
            max_iter=5,
        )
        total_runtime = runtime_used + (time.perf_counter() - start_time)
        runtimes.append(total_runtime)

        analysis_states_denorm = denorm(result["analysis_states"]).cpu()
        gt_physical_window = gt_frames_physical[: args.window].cpu()

        per_step_mse = compute_mse_per_step(analysis_states_denorm, gt_physical_window)
        per_step_mse_runs.append(per_step_mse)
        overall_mse_runs.append(per_step_mse.mean().item())

        if analysis_sample is None:
            analysis_sample = analysis_states_denorm[: args.window]

        print(
            f"[Trial {trial + 1}/{args.num_trials}] "
            f"iters={iters}, mean MSE={per_step_mse.mean().item():.6f}, runtime={total_runtime:.3f}s"
        )

    per_step_mse_stack = torch.stack(per_step_mse_runs)  # (num_trials, window)
    per_step_mean = per_step_mse_stack.mean(dim=0).numpy()
    per_step_std = per_step_mse_stack.std(dim=0, unbiased=False).numpy()
    overall_mean = float(np.mean(overall_mse_runs))
    overall_std = float(np.std(overall_mse_runs, ddof=0))
    runtime_mean = float(np.mean(runtimes))
    runtime_std = float(np.std(runtimes, ddof=0))

    np.save(os.path.join(args.save_dir, "multi.npy"), analysis_sample.numpy())
    np.savez(
        os.path.join(args.save_dir, "multi_meanstd.npz"),
        per_step_mean=per_step_mean,
        per_step_std=per_step_std,
        overall_mean=overall_mean,
        overall_std=overall_std,
        runtime_mean=runtime_mean,
        runtime_std=runtime_std,
    )

    print("\n==== Direct DA Summary ====")
    print(f"Overall MSE mean: {overall_mean:.6f}, std: {overall_std:.6f}")
    print(f"Runtime mean: {runtime_mean:.3f}s, std: {runtime_std:.3f}s")
    print(f"Per-step mean/std saved to {os.path.join(args.save_dir, 'multi_meanstd.npz')}")
    print(f"Sample analysis saved to {os.path.join(args.save_dir, 'multi.npy')}")


if __name__ == "__main__":
    main()
