import random
from time import perf_counter
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchda


def set_seed(seed: Optional[int]):
    """Set random seeds for reproducibility."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def set_device() -> str:
    """Return CUDA if available, otherwise CPU."""
    if torch.cuda.device_count() == 0:
        return "cpu"
    torch.set_float32_matmul_precision("high")
    return "cuda"


class UnifiedDynamicSparseObservationHandler:
    """
    Generate and reuse unified sparse observation masks across time steps.

    Supports optional additive Gaussian noise on observed entries to
    emulate noisy measurements.
    """

    def __init__(
        self,
        max_obs_ratio: float = 0.15,
        min_obs_ratio: float = 0.05,
        seed: Optional[int] = 42,
        noise_std: float = 0.0,
        fixed_valid_mask: bool = True,
    ):
        self.max_obs_ratio = max_obs_ratio
        self.min_obs_ratio = min_obs_ratio
        self.seed = seed
        self.noise_std = noise_std
        self.fixed_positions: Optional[torch.Tensor] = None
        self.max_obs_count: int = 0
        self.time_masks: dict[int, dict[str, torch.Tensor | float | int]] = {}
        self.fixed_valid_mask = fixed_valid_mask
        self._shared_valid_indices: Optional[torch.Tensor] = None

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

    def generate_unified_observations(
        self, image_shape: Tuple[int, int, int], time_steps: Sequence[int]
    ) -> int:
        if len(image_shape) != 3:
            raise ValueError(f"Expected 3D image shape (C, H, W), got {image_shape}")

        C, H, W = image_shape
        total_pixels = C * H * W
        self.max_obs_count = int(total_pixels * self.max_obs_ratio)

        # 固定观测位置（大小 = max_obs_count）
        self.fixed_positions = torch.randperm(total_pixels)[: self.max_obs_count]

        actual_ratio = self.max_obs_count / total_pixels
        print(f"Fixed observation setup: {self.max_obs_count} observations ({actual_ratio:.3%} ratio)")
        print(f"Noise level: σ = {self.noise_std:.4f}")

        # 固定有效观测索引（全部有效 = 固定数量）
        if self.fixed_valid_mask:
            self._shared_valid_indices = torch.arange(self.max_obs_count)
            shared_num_valid = self.max_obs_count
            shared_ratio = actual_ratio

        self.time_masks = {}
        for t in time_steps:
            if self.fixed_valid_mask:
                valid_indices = self._shared_valid_indices
                num_valid = shared_num_valid
                obs_ratio = shared_ratio
            else:
                obs_ratio = np.random.uniform(self.min_obs_ratio, self.max_obs_ratio)
                num_valid = min(int(total_pixels * obs_ratio), self.max_obs_count)
                valid_indices = torch.randperm(self.max_obs_count)[:num_valid]

            self.time_masks[int(t)] = {
                "num_valid": int(num_valid),
                "valid_indices": valid_indices,
                "obs_ratio": float(obs_ratio),
            }

        return self.max_obs_count

    def apply_unified_observation(
        self,
        full_image: torch.Tensor,
        time_step_idx: int,
        add_noise: bool = True,
    ) -> torch.Tensor:
        """Apply observation mask to a full image tensor."""
        if time_step_idx not in self.time_masks:
            raise ValueError(f"Time step index {time_step_idx} not found in masks")

        if self.fixed_positions is None:
            raise RuntimeError("Call generate_unified_observations before applying.")

        mask_info = self.time_masks[time_step_idx]
        flat_image = full_image.flatten()
        fixed_obs = flat_image[self.fixed_positions]

        obs_vector = torch.zeros(self.max_obs_count, device=full_image.device)
        valid_indices = mask_info["valid_indices"]
        obs_vector[valid_indices] = fixed_obs[valid_indices]

        if add_noise and self.noise_std > 0:
            obs_vector[valid_indices] += (
                torch.randn_like(obs_vector[valid_indices]) * self.noise_std
            )

        return obs_vector

    def create_block_R_matrix(self, base_variance: Optional[float] = None) -> torch.Tensor:
        """Create a diagonal observation covariance matrix."""
        if base_variance is None:
            base_variance = max(self.noise_std**2, 1e-6)
        return torch.eye(self.max_obs_count) * base_variance


class MLPDAExecutor:
    """
    Shared utilities for running AE-based data assimilation experiments.
    """

    def __init__(
        self,
        forward_model,
        obs_handler: UnifiedDynamicSparseObservationHandler,
        device: str,
        optimizer_kwargs: Optional[dict] = None,
        max_iterations: int = 5000,
        early_stop: Tuple[int, float] = (100, 1e-2),
        algorithm: torchda.Algorithms = torchda.Algorithms.Var4D,
        output_sequence_length: int = 1,
    ):
        self.forward_model = forward_model
        self.obs_handler = obs_handler
        self.device = device
        self.optimizer_kwargs = optimizer_kwargs or {"lr": 0.05}
        self.max_iterations = max_iterations
        self.early_stop = early_stop
        self.algorithm = algorithm
        self.output_sequence_length = output_sequence_length

        self._obs_time_idx = 0

    def _forecast_wrapper(self, z_t: torch.Tensor, time_fw=None, *args):
        """Wrapper for forward latent dynamics."""
        if z_t.ndim == 1:
            z_t = z_t.unsqueeze(0)
        if time_fw is None:
            return self.forward_model.latent_forward(z_t)

        time_steps = int(time_fw.shape[0])
        z_tp = torch.empty(
            (time_steps, z_t.shape[0], z_t.shape[1]), device=z_t.device
        )
        z_current = z_t

        for i in range(time_steps):
            z_tp[i] = z_current
            if i < time_steps - 1:
                z_current = self.forward_model.latent_forward(z_current)

        return z_tp

    def _observation_operator(self, x: torch.Tensor) -> torch.Tensor:
        """Observation operator using the unified sparse handler."""
        x_reconstructed = self.forward_model.K_S_preimage(x)
        sparse_obs = self.obs_handler.apply_unified_observation(
            x_reconstructed.squeeze(), self._obs_time_idx, add_noise=False
        )
        return sparse_obs.unsqueeze(0)

    def _build_case(
        self,
        observations: torch.Tensor,
        background_state: torch.Tensor,
        observation_time_idx: int,
        observation_time_steps: Optional[List[int]] = None,
        gaps: Optional[List[int]] = None,
        B: Optional[torch.Tensor] = None,
        R: Optional[torch.Tensor] = None,
    ):
        """Configure a torchda CaseBuilder for one assimilation step."""
        self._obs_time_idx = observation_time_idx
        if observation_time_steps is None:
            observation_time_steps = [0]
        if gaps is None:
            gaps = [1]

        case_builder = (
            torchda.CaseBuilder()
            .set_observation_time_steps(observation_time_steps)
            .set_gaps(gaps)
            .set_forward_model(self._forecast_wrapper)
            .set_observation_model(self._observation_operator)
            .set_background_covariance_matrix(B)
            .set_observation_covariance_matrix(R)
            .set_observations(observations)
            .set_optimizer_cls(torch.optim.Adam)
            .set_optimizer_args(self.optimizer_kwargs)
            .set_max_iterations(self.max_iterations)
            .set_early_stop(self.early_stop)
            .set_algorithm(self.algorithm)
            .set_device(torchda.Device.GPU if self.device == "cuda" else torchda.Device.CPU)
            .set_output_sequence_length(self.output_sequence_length)
        )
        case_builder.set_background_state(background_state)
        return case_builder

    def assimilate_step(
        self,
        observations: torch.Tensor,
        background_state: torch.Tensor,
        observation_time_idx: int = 0,
        observation_time_steps: Optional[List[int]] = None,
        gaps: Optional[List[int]] = None,
        B: Optional[torch.Tensor] = None,
        R: Optional[torch.Tensor] = None,
    ):
        """
        Run a single-step assimilation and return the assimilated latent state,
        intermediate results, and wall-clock time.
        """
        case_builder = self._build_case(
            observations=observations,
            background_state=background_state,
            observation_time_idx=observation_time_idx,
            observation_time_steps=observation_time_steps,
            gaps=gaps,
            B=B,
            R=R,
        )

        start_time = perf_counter()
        result = case_builder.execute()
        elapsed = perf_counter() - start_time

        assimilated_state = result["assimilated_state"]
        return assimilated_state, result.get("intermediate_results", {}), elapsed

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent state to the physical space."""
        return self.forward_model.K_S_preimage(latent)
