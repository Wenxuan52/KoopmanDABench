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
    """

    def __init__(
        self,
        max_obs_ratio: float = 0.15,
        min_obs_ratio: float = 0.05,
        seed: Optional[int] = 42,
    ):
        self.max_obs_ratio = max_obs_ratio
        self.min_obs_ratio = min_obs_ratio
        self.seed = seed
        self.fixed_positions: Optional[torch.Tensor] = None
        self.max_obs_count: int = 0
        self.time_masks: dict[int, dict[str, torch.Tensor | float | int]] = {}

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

    def generate_unified_observations(
        self, image_shape: Tuple[int, int, int], time_steps: Sequence[int]
    ) -> int:
        """
        Create fixed observation positions and per-time-step valid indices.
        """
        if len(image_shape) != 3:
            raise ValueError(
                f"Expected 3D image shape (C, H, W), got {image_shape}"
            )

        C, H, W = image_shape
        total_pixels = C * H * W
        self.max_obs_count = int(total_pixels * self.max_obs_ratio)

        self.fixed_positions = torch.randperm(total_pixels)[: self.max_obs_count]

        for i, _ in enumerate(time_steps):
            obs_ratio = np.random.uniform(self.min_obs_ratio, self.max_obs_ratio)
            num_valid = int(total_pixels * obs_ratio)
            num_valid = min(num_valid, self.max_obs_count)

            valid_indices = torch.randperm(self.max_obs_count)[:num_valid]
            self.time_masks[i] = {
                "num_valid": num_valid,
                "valid_indices": valid_indices,
                "obs_ratio": obs_ratio,
            }

        return self.max_obs_count

    def apply_unified_observation(
        self, full_image: torch.Tensor, time_step_idx: int
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

        return obs_vector

    def create_block_R_matrix(self, base_variance: float = 1e-3) -> torch.Tensor:
        """Create a diagonal observation covariance matrix."""
        return torch.eye(self.max_obs_count) * base_variance


class KoopmanDAExecutor:
    """
    Shared utilities for running Koopman-based data assimilation experiments.
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

        z_tp = torch.empty(
            (time_fw.shape[0], z_t.shape[0], z_t.shape[1]), device=z_t.device
        )
        current_state = self.forward_model.K_S_preimage(z_t)

        for i in range(int(time_fw.shape[0])):
            z_current = self.forward_model.K_S(current_state)
            z_tp[i] = z_current
            if i < int(time_fw.shape[0]) - 1:
                z_next = self.forward_model.latent_forward(z_current)
                current_state = self.forward_model.K_S_preimage(z_next)

        return z_tp

    def _observation_operator(self, x: torch.Tensor) -> torch.Tensor:
        """Observation operator using the unified sparse handler."""
        x_reconstructed = self.forward_model.K_S_preimage(x)
        sparse_obs = self.obs_handler.apply_unified_observation(
            x_reconstructed.squeeze(), self._obs_time_idx
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
        observation_time_steps = observation_time_steps or [0]
        gaps = gaps or [1]

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
