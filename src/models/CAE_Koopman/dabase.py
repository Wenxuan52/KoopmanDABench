import os
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

current_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", ".."))
sys.path.append(src_directory)


class CAEKoopmanDABase:
    """
    Core data assimilation utilities for CAE_Koopman models.

    The class exposes an analytic latent-space 4D-Var / RTS smoother that
    leverages the linear latent dynamics learned by CAE_Koopman:

        z_{t+1} = A z_t

    Observations are provided in physical space and mapped to latent space
    through a linearization of H ∘ decoder around a background trajectory.
    Model error is ignored, consistent with the strong-constraint setting.
    """

    def __init__(
        self,
        dynamics_matrix: Tensor,
        decoder: torch.nn.Module,
        observation_operator: Callable[[Tensor], Tensor],
        encoder: Optional[torch.nn.Module] = None,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            dynamics_matrix: Learned latent transition matrix A with shape [d_z, d_z].
            decoder: Decoder mapping latent variables back to physical space.
            observation_operator: Callable that extracts observations from decoded physical states.
            encoder: Optional encoder to transform physical background states into latent space.
            device: Torch device where computations will be executed.
        """
        self.dynamics_matrix = dynamics_matrix.detach().to(device)
        self.decoder = decoder.to(device)
        self.observation_operator = observation_operator
        self.encoder = encoder.to(device) if encoder is not None else None
        self.device = device
        self.latent_dim = self.dynamics_matrix.shape[-1]

        self.decoder.eval()
        if self.encoder is not None:
            self.encoder.eval()

    def to(self, device: str) -> "CAEKoopmanDABase":
        self.device = device
        self.dynamics_matrix = self.dynamics_matrix.to(device)
        self.decoder = self.decoder.to(device)
        if self.encoder is not None:
            self.encoder = self.encoder.to(device)
        return self

    def propagate_latent(self, initial_latent: Tensor, steps: int) -> Tensor:
        """
        Deterministically propagate a latent state through the linear dynamics.

        Args:
            initial_latent: z_0 with shape [d_z] or [1, d_z].
            steps: Number of states to generate (including the initial state).

        Returns:
            Tensor of shape [steps, d_z] containing the background trajectory.
        """
        current = initial_latent.to(self.device)
        if current.dim() > 1:
            current = current.squeeze(0)

        latents = []
        for _ in range(steps):
            latents.append(current)
            current = torch.matmul(current, self.dynamics_matrix)
        return torch.stack(latents, dim=0)

    def decode_latents(self, latent_seq: Tensor) -> Tensor:
        """
        Decode a latent trajectory into physical space.
        """
        decoded_states = []
        with torch.no_grad():
            for latent in latent_seq:
                decoded = self.decoder(latent.unsqueeze(0))
                decoded_states.append(decoded.squeeze(0).detach().cpu())
        return torch.stack(decoded_states, dim=0)

    def _forward_observation(self, latent_state: Tensor) -> Tensor:
        """
        Apply decoder and observation operator to a latent vector.
        """
        decoded_state = self.decoder(latent_state.unsqueeze(0))
        observation = self.observation_operator(decoded_state)
        return observation.reshape(-1)

    def _linearize_observations(
        self, background_latents: Tensor, observations: Sequence[Optional[Tensor]]
    ) -> Tuple[List[Optional[Tensor]], List[Optional[Tensor]]]:
        """
        Compute Jacobians H_z,t and baseline observations along the background trajectory.

        Linearization is only performed at time steps where observations are supplied.
        """
        hz_list: List[Optional[Tensor]] = []
        base_observations: List[Optional[Tensor]] = []

        for idx, latent in enumerate(background_latents):
            if observations[idx] is None:
                hz_list.append(None)
                base_observations.append(None)
                continue

            latent_for_jac = latent.detach().to(self.device).requires_grad_(True)

            def obs_fn(z_vector: Tensor) -> Tensor:
                return self._forward_observation(z_vector)

            jacobian = torch.autograd.functional.jacobian(
                obs_fn, latent_for_jac, vectorize=True
            ).detach()
            base_obs = obs_fn(latent_for_jac.detach())

            hz_list.append(jacobian)
            base_observations.append(base_obs.detach())

        return hz_list, base_observations

    @staticmethod
    def _adjust_observations(
        observations: Sequence[Optional[Tensor]],
        base_observations: Sequence[Optional[Tensor]],
        hz_list: Sequence[Optional[Tensor]],
        background_latents: Tensor,
    ) -> List[Optional[Tensor]]:
        """
        Center observations using the linearized observation model:
        y'_t = y_t - H(FD(z_b,t)) + H_z,t z_b,t
        """
        adjusted: List[Optional[Tensor]] = []
        for idx, obs in enumerate(observations):
            if obs is None or hz_list[idx] is None or base_observations[idx] is None:
                adjusted.append(None)
                continue
            centered = (
                obs.to(background_latents.device)
                - base_observations[idx].to(background_latents.device)
                + torch.matmul(hz_list[idx], background_latents[idx])
            )
            adjusted.append(centered)
        return adjusted

    def _prepare_observation_noise(
        self,
        observation_noise: Union[float, Tensor, Sequence[Union[float, Tensor]]],
        time_index: int,
        obs_dim: int,
        min_variance: float = 1e-6,
    ) -> Tensor:
        """
        Convert flexible noise specifications into a full covariance matrix.
        Always enforces a minimum variance to avoid singular innovations.
        """
        if isinstance(observation_noise, (list, tuple)):
            noise_spec = observation_noise[time_index]
        else:
            noise_spec = observation_noise

        if noise_spec is None:
            cov = torch.eye(obs_dim, device=self.device, dtype=self.dynamics_matrix.dtype) * min_variance
        else:
            cov = torch.as_tensor(noise_spec, device=self.device, dtype=self.dynamics_matrix.dtype)
            if cov.dim() == 0:
                cov = torch.eye(obs_dim, device=self.device, dtype=self.dynamics_matrix.dtype) * max(
                    float(cov.item()), min_variance
                )
            elif cov.dim() == 1:
                cov = torch.diag(torch.maximum(cov, torch.full_like(cov, min_variance)))
            else:
                cov = cov.clone()
                cov = cov + torch.eye(obs_dim, device=self.device, dtype=self.dynamics_matrix.dtype) * min_variance
        return cov

    def _kalman_filter(
        self,
        adjusted_observations: Sequence[Optional[Tensor]],
        hz_list: Sequence[Optional[Tensor]],
        observation_noise: Union[float, Tensor, Sequence[Union[float, Tensor]]],
        initial_mean: Tensor,
        initial_covariance: Tensor,
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
        """
        Run the forward Kalman filter for the linear latent model.
        """
        num_steps = len(adjusted_observations)
        identity = torch.eye(self.latent_dim, device=self.device, dtype=self.dynamics_matrix.dtype)

        filter_means: List[Tensor] = []
        filter_covs: List[Tensor] = []
        pred_means: List[Tensor] = []
        pred_covs: List[Tensor] = []

        mu_prev = initial_mean.to(self.device)
        P_prev = initial_covariance.to(self.device)

        for t in range(num_steps):
            if t == 0:
                mu_pred = mu_prev
                P_pred = P_prev
            else:
                mu_pred = torch.matmul(mu_prev, self.dynamics_matrix)
                P_pred = self.dynamics_matrix @ P_prev @ self.dynamics_matrix.t()

            pred_means.append(mu_pred)
            pred_covs.append(P_pred)

            obs = adjusted_observations[t]
            hz_t = hz_list[t]
            if obs is None or hz_t is None:
                mu_post = mu_pred
                P_post = P_pred
            else:
                R_t = self._prepare_observation_noise(observation_noise, t, obs.numel())
                innovation_cov = hz_t @ P_pred @ hz_t.t() + R_t
                jitter = 1e-6
                for _ in range(3):
                    try:
                        inv_innov = torch.linalg.pinv(innovation_cov)
                        break
                    except torch.linalg.LinAlgError:
                        innovation_cov = innovation_cov + jitter * torch.eye(
                            innovation_cov.shape[-1], device=innovation_cov.device, dtype=innovation_cov.dtype
                        )
                        jitter *= 10
                else:
                    inv_innov = torch.linalg.pinv(innovation_cov)
                gain = P_pred @ hz_t.t() @ inv_innov
                innovation = obs - torch.matmul(hz_t, mu_pred)
                mu_post = mu_pred + torch.matmul(gain, innovation)
                P_post = (identity - gain @ hz_t) @ P_pred

            filter_means.append(mu_post)
            filter_covs.append(P_post)
            mu_prev = mu_post
            P_prev = P_post

        return filter_means, filter_covs, pred_means, pred_covs

    def _rts_smoother(
        self,
        filter_means: Sequence[Tensor],
        filter_covs: Sequence[Tensor],
        pred_means: Sequence[Tensor],
        pred_covs: Sequence[Tensor],
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Rauch–Tung–Striebel smoother applied to the filtered estimates.
        """
        num_steps = len(filter_means)
        smoothed_means: List[Tensor] = [None] * num_steps  # type: ignore
        smoothed_covs: List[Tensor] = [None] * num_steps  # type: ignore

        smoothed_means[-1] = filter_means[-1]
        smoothed_covs[-1] = filter_covs[-1]

        for t in reversed(range(num_steps - 1)):
            P_ft = filter_covs[t]
            P_pred_next = pred_covs[t + 1]
            jitter = 1e-6
            for _ in range(3):
                try:
                    inv_p_pred_next = torch.linalg.pinv(P_pred_next)
                    break
                except torch.linalg.LinAlgError:
                    P_pred_next = P_pred_next + jitter * torch.eye(
                        P_pred_next.shape[-1], device=P_pred_next.device, dtype=P_pred_next.dtype
                    )
                    jitter *= 10
            else:
                inv_p_pred_next = torch.linalg.pinv(P_pred_next)
            gain = P_ft @ self.dynamics_matrix.t() @ inv_p_pred_next

            next_mean_diff = smoothed_means[t + 1] - pred_means[t + 1]
            smoothed_means[t] = filter_means[t] + torch.matmul(gain, next_mean_diff)

            next_cov_diff = smoothed_covs[t + 1] - P_pred_next
            smoothed_covs[t] = P_ft + gain @ next_cov_diff @ gain.t()

        return smoothed_means, smoothed_covs

    def run_assimilation(
        self,
        observations: Sequence[Optional[Tensor]],
        background_initial_state: Tensor,
        background_covariance: Tensor,
        observation_noise: Union[float, Tensor, Sequence[Union[float, Tensor]]],
        background_is_latent: bool = False,
    ) -> Dict[str, Union[Tensor, List[Tensor], float, List[Optional[Tensor]]]]:
        """
        Execute analytic latent-space DA over a single window.

        Args:
            observations: Sequence of observation vectors (None for missing frames).
            background_initial_state: Background initial state in physical space (or latent
                space when `background_is_latent=True`).
            background_covariance: Background covariance P_0 in latent space.
            observation_noise: Observation noise specification (shared or per time).
            background_is_latent: Whether `background_initial_state` is already latent.

        Returns:
            Dictionary containing latent/physical analyses, intermediate statistics,
            and runtime in seconds.
        """
        start_time = time.perf_counter()
        if len(observations) == 0:
            raise ValueError("At least one observation entry is required for DA.")

        if background_is_latent:
            z0 = background_initial_state.to(self.device)
        else:
            if self.encoder is None:
                raise ValueError("Encoder must be provided when supplying physical backgrounds.")
            with torch.no_grad():
                z0 = self.encoder(background_initial_state.unsqueeze(0).to(self.device)).squeeze(0)

        background_latents = self.propagate_latent(z0, len(observations))
        background_states = self.decode_latents(background_latents)

        hz_list, base_observations = self._linearize_observations(background_latents, observations)
        adjusted_observations = self._adjust_observations(
            observations, base_observations, hz_list, background_latents
        )

        filter_means, filter_covs, pred_means, pred_covs = self._kalman_filter(
            adjusted_observations, hz_list, observation_noise, z0, background_covariance
        )
        smoothed_means, smoothed_covs = self._rts_smoother(
            filter_means, filter_covs, pred_means, pred_covs
        )

        analysis_latents = torch.stack(smoothed_means, dim=0)
        analysis_states = self.decode_latents(analysis_latents)

        runtime_seconds = time.perf_counter() - start_time

        return {
            "analysis_latents": analysis_latents,
            "analysis_states": analysis_states,
            "filtered_latents": torch.stack(filter_means, dim=0),
            "filtered_covariances": filter_covs,
            "background_latents": background_latents,
            "background_states": background_states,
            "observation_jacobians": hz_list,
            "adjusted_observations": adjusted_observations,
            "smoothing_covariances": smoothed_covs,
            "runtime_seconds": runtime_seconds,
        }