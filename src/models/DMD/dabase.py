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
    ):
        self.max_obs_ratio = max_obs_ratio
        self.min_obs_ratio = min_obs_ratio
        self.seed = seed
        self.noise_std = noise_std
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

        actual_ratio = self.max_obs_count / total_pixels
        print(
            f"Fixed observation setup: {self.max_obs_count} observations ({actual_ratio:.3%} ratio)"
        )
        print(f"Noise level: σ = {self.noise_std:.4f}")

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

            print(
                f"Time step {i}: {num_valid}/{self.max_obs_count} observations ({obs_ratio:.3%} ratio)"
            )

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


class DMDDAExecutor:
    """
    Shared utilities for running DMD-based data assimilation experiments.
    """

    def __init__(
        self,
        dmd_model,
        obs_handler: UnifiedDynamicSparseObservationHandler,
        device: str,
        image_shape: Tuple[int, int, int],
        optimizer_kwargs: Optional[dict] = None,
        max_iterations: int = 5000,
        early_stop: Tuple[int, float] = (100, 1e-2),
        algorithm: torchda.Algorithms = torchda.Algorithms.Var4D,
        output_sequence_length: int = 1,
    ):
        self.dmd_model = dmd_model
        self.obs_handler = obs_handler
        self.device = device
        self.image_shape = image_shape
        self.optimizer_kwargs = optimizer_kwargs or {"lr": 0.05}
        self.max_iterations = max_iterations
        self.early_stop = early_stop
        self.algorithm = algorithm
        self.output_sequence_length = output_sequence_length

        self._obs_time_idx = 0

    @staticmethod
    def complex_to_real(b_complex: torch.Tensor) -> torch.Tensor:
        return torch.cat([b_complex.real, b_complex.imag], dim=-1)

    @staticmethod
    def real_to_complex(b_real: torch.Tensor) -> torch.Tensor:
        if b_real.ndim == 1:
            b_real = b_real.unsqueeze(0)
        half = b_real.shape[-1] // 2
        return b_real[..., :half] + 1j * b_real[..., half:]

    def latent_forward(self, b_complex: torch.Tensor) -> torch.Tensor:
        if b_complex.ndim == 1:
            b_complex = b_complex.unsqueeze(0)
        eig_diag = torch.diag(self.dmd_model.eigenvalues)
        return (eig_diag @ b_complex.T).T

    def _forecast_wrapper(self, b_real: torch.Tensor, time_fw=None, *args):
        b_real = b_real if b_real.ndim > 1 else b_real.unsqueeze(0)
        b_complex = self.real_to_complex(b_real)

        if time_fw is None:
            b_next = self.latent_forward(b_complex)
            return self.complex_to_real(b_next)

        time_steps = int(time_fw.shape[0])
        b_tp = torch.empty((time_steps, b_real.shape[0], b_real.shape[1]), device=b_real.device)
        b_current = b_complex

        for i in range(time_steps):
            b_tp[i] = self.complex_to_real(b_current)
            if i < time_steps - 1:
                b_current = self.latent_forward(b_current)

        return b_tp

    def _observation_operator(self, b_real: torch.Tensor) -> torch.Tensor:
        b_complex = self.real_to_complex(b_real)
        x_flat = (self.dmd_model.modes @ b_complex.squeeze()).real
        x_reconstructed = x_flat.reshape(self.image_shape)
        sparse_obs = self.obs_handler.apply_unified_observation(
            x_reconstructed, self._obs_time_idx, add_noise=False
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

    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        flat = state.reshape(-1).to(self.device)
        b_complex = torch.linalg.lstsq(
            self.dmd_model.modes, flat.to(torch.complex64)
        ).solution
        return b_complex

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.is_complex():
            b_complex = latent
        else:
            b_complex = self.real_to_complex(latent)
        x_flat = (self.dmd_model.modes @ b_complex.squeeze()).real
        return x_flat.reshape((1, *self.image_shape))
