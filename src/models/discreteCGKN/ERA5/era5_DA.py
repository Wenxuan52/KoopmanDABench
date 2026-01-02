"""Discrete CGKN data assimilation on ERA5 with conditional dynamics.

This script mirrors the ERA5 discrete CGKN training artifacts and applies the
NSE-style closed-form predict/update (coefficients conditioned on the held
probe) over a fixed assimilation window. Observation schedule is defined over
steps k=0..window_length-1, where step k corresponds to physical time t=k+1.
At non-scheduled steps, the last observed probe is carried forward without
peeking at ground truth.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

# Ensure project root is importable
current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", "..", "..", ".."))
if src_directory not in sys.path:
    sys.path.append(src_directory)

from src.utils.Dataset import ERA5Dataset  # noqa: E402
from src.models.discreteCGKN.ERA5.era5_model import ERA5Decoder, ERA5Encoder  # noqa: E402
from src.models.discreteCGKN.ERA5.era5_train import DiscreteCGN, ProbeSampler  # noqa: E402
from src.models.CAE_Koopman.ERA5.era5_model_FTF import ERA5_settings  # noqa: E402


def _tensor_summary(name: str, tensor: torch.Tensor) -> str:
    flat = tensor.detach().reshape(-1)
    finite = flat[torch.isfinite(flat)]
    if finite.numel() == 0:
        return f"{name}: all elements are non-finite, shape={tuple(tensor.shape)}"
    return (
        f"{name}: shape={tuple(tensor.shape)}, min={finite.min():.4e}, max={finite.max():.4e},"
        f" mean={finite.mean():.4e}, std={finite.std():.4e}"
    )


def set_seed(seed: int | None):
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)


def set_device(device_str: str | None = None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        return torch.device("cuda")
    return torch.device("cpu")


def parse_schedule(schedule_arg: str | None, window_length: int) -> List[int]:
    if schedule_arg is None or schedule_arg == "":
        return list(range(window_length))
    if isinstance(schedule_arg, str):
        parts = [p for p in schedule_arg.split(",") if p != ""]
        schedule = [int(p) for p in parts]
    else:
        schedule = list(schedule_arg)
    schedule = [s for s in schedule if 0 <= s < window_length]
    if len(schedule) == 0:
        schedule = list(range(window_length))
    return schedule


def safe_denorm(x: torch.Tensor, dataset: ERA5Dataset) -> torch.Tensor:
    """Denormalize while matching the input device to avoid device mismatch."""

    min_v = torch.as_tensor(dataset.min, device=x.device).reshape(1, -1, 1, 1)
    max_v = torch.as_tensor(dataset.max, device=x.device).reshape(1, -1, 1, 1)
    return x * (max_v - min_v + 1e-6) + min_v


def compute_metrics(da_states: torch.Tensor, noda_states: torch.Tensor, groundtruth: torch.Tensor) -> Dict[str, np.ndarray]:
    mse_list: List[np.ndarray] = []
    rrmse_list: List[np.ndarray] = []
    ssim_list: List[np.ndarray] = []

    if da_states.dim() == 5 and da_states.shape[1] == 1:
        da_states = da_states.squeeze(1)
    if noda_states.dim() == 5 and noda_states.shape[1] == 1:
        noda_states = noda_states.squeeze(1)
    if groundtruth.dim() == 5 and groundtruth.shape[1] == 1:
        groundtruth = groundtruth.squeeze(1)

    assert da_states.shape == noda_states.shape == groundtruth.shape, (
        f"Shape mismatch: da {da_states.shape}, noda {noda_states.shape}, gt {groundtruth.shape}"
    )

    T = groundtruth.shape[0]
    for t in range(T):
        gt = groundtruth[t].cpu()
        da = da_states[t].cpu()
        noda = noda_states[t].cpu()

        mse_step = []
        rrmse_step = []
        ssim_step = []
        for c in range(gt.shape[0]):
            diff_da = (da[c] - gt[c]) ** 2
            diff_noda = (noda[c] - gt[c]) ** 2
            mse_step.append((diff_da.mean().item(), diff_noda.mean().item()))
            denom = (gt[c] ** 2).sum().clamp_min(1e-8)
            rrmse_step.append(((diff_da.sum() / denom).sqrt().item(), (diff_noda.sum() / denom).sqrt().item()))
            data_range = (gt[c].max() - gt[c].min()).item()
            if data_range > 0:
                ssim_da = ssim(gt[c].detach().numpy(), da[c].detach().numpy(), data_range=data_range)
                ssim_noda = ssim(gt[c].detach().numpy(), noda[c].detach().numpy(), data_range=data_range)
            else:
                ssim_da = 1.0
                ssim_noda = 1.0
            ssim_step.append((ssim_da, ssim_noda))
        mse_list.append(mse_step)
        rrmse_list.append(rrmse_step)
        ssim_list.append(ssim_step)
    return {"mse": np.array(mse_list), "rrmse": np.array(rrmse_list), "ssim": np.array(ssim_list)}


def decode_mu(decoder: ERA5Decoder, mu: torch.Tensor) -> torch.Tensor:
    """Decode latent mean to a single-frame tensor.

    Args:
        mu: [B, dim_z, 1] latent mean.
    Returns:
        frame: [B, C, H, W]
    """

    mu_seq = mu.transpose(1, 2)  # [B, 1, dim_z]
    frames = decoder(mu_seq)
    return frames[:, 0]


def load_models(device: torch.device, checkpoint_dir: Path, dim_z: int, dim_u1: int, ckpt_prefix: str) -> Tuple[ERA5Encoder, ERA5Decoder, DiscreteCGN]:
    encoder = ERA5Encoder(dim_z=dim_z).to(device)
    decoder = ERA5Decoder(dim_z=dim_z).to(device)
    cgn = DiscreteCGN(dim_u1=dim_u1, dim_z=dim_z).to(device)

    encoder.load_state_dict(torch.load(checkpoint_dir / f"{ckpt_prefix}_encoder.pt", map_location=device, weights_only=True))
    decoder.load_state_dict(torch.load(checkpoint_dir / f"{ckpt_prefix}_decoder.pt", map_location=device, weights_only=True))
    cgn.load_state_dict(torch.load(checkpoint_dir / f"{ckpt_prefix}_cgn.pt", map_location=device, weights_only=True))

    encoder.eval()
    decoder.eval()
    cgn.eval()
    return encoder, decoder, cgn


def prepare_probe_sampler(probe_file: Path, channels: List[int]) -> ProbeSampler:
    coords_np = np.load(probe_file)
    coords: List[Tuple[int, int]] = [tuple(map(int, pair)) for pair in coords_np.tolist()]
    return ProbeSampler(coords, channels)


def apply_filter_step(
    cgn: DiscreteCGN,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    R: torch.Tensor,
    u_cond: torch.Tensor,
    u_obs: torch.Tensor | None,
    do_update: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Predict/update with coefficients conditioned on held u1 and innovation on u1_next.

    Args:
        mu: [B, dim_z, 1]
        R: [B, dim_z, dim_z]
        u_cond: [B, dim_u1] probe used to produce (f,g)
        u_obs: [B, dim_u1] probe observation at next step (None if no update)
        do_update: whether to apply the innovation-based correction
    """

    dim_u1 = cgn.dim_u1
    dim_z = cgn.dim_z
    sigma_u1 = sigma[:dim_u1]
    sigma_z = sigma[dim_u1:]

    def symmetrize_jitter(mat: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        mat = 0.5 * (mat + mat.transpose(1, 2))
        eye = torch.eye(mat.shape[-1], device=mat.device).unsqueeze(0)
        return mat + eps * eye

    def solve_spd(mat: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        """Solve mat @ x = rhs assuming mat is (near) SPD.

        Uses progressively larger jittered Cholesky solves; if they all fail,
        falls back to a diagonal solve to avoid SVD/pseudo-inverse instability
        on ill-conditioned batches.
        """

        for jitter in (1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1):
            mat_spd = symmetrize_jitter(mat, eps=jitter)
            L, info = torch.linalg.cholesky_ex(mat_spd)
            if info.max().item() == 0:
                return torch.cholesky_solve(rhs, L)
        # Diagonal fallback: treat mat as diagonal dominant
        mat_diag = torch.diagonal(mat_spd, dim1=-2, dim2=-1).abs().clamp_min(1e-3)
        return rhs / mat_diag.unsqueeze(-1)

    R = symmetrize_jitter(R, eps=1e-4)

    f1, g1, f2, g2 = cgn.get_cgn_mats(u_cond)

    mu_pred = f2 + torch.bmm(g2, mu)
    R_pred = torch.bmm(torch.bmm(g2, R), g2.transpose(1, 2)) + torch.diag(sigma_z**2).to(R).unsqueeze(0)
    R_pred = symmetrize_jitter(R_pred)

    if not do_update or u_obs is None:
        return mu_pred, R_pred

    # Innovation uses u_obs = u1^{n+1}, coefficients from u_cond = u1^{n}
    innov = u_obs.unsqueeze(-1) - (f1 + torch.bmm(g1, mu))  # [B, dim_u1, 1]
    Dinv = (1.0 / (sigma_u1**2 + 1e-6)).view(1, dim_u1, 1)
    Dinv_g1 = Dinv * g1  # [B, dim_u1, dim_z]
    Gt_Dinv_G = torch.bmm(g1.transpose(1, 2), Dinv_g1)  # [B, dim_z, dim_z]

    eye_z = torch.eye(dim_z, device=R.device).unsqueeze(0)
    R_inv_eye = solve_spd(R, eye_z)  # [B, dim_z, dim_z]
    M = R_inv_eye + Gt_Dinv_G
    M = symmetrize_jitter(M)
    M_inv = solve_spd(M, eye_z)

    def apply_Sinv(rhs: torch.Tensor) -> torch.Tensor:
        """Apply S^{-1} to a RHS (Woodbury), rhs shape [B, dim_u1, k]."""

        Dinv_rhs = Dinv * rhs
        Gt_Dinv_rhs = torch.bmm(g1.transpose(1, 2), Dinv_rhs)
        middle = torch.bmm(M_inv, Gt_Dinv_rhs)
        correction = torch.bmm(Dinv_g1, middle)
        return Dinv_rhs - correction

    # Gain and updates per Eq. (2.6): A = g2 R g1^T, B = S^{-1}(g1 R g2^T)
    R_g1T = torch.bmm(R, g1.transpose(1, 2))  # [B, dim_z, dim_u1]
    g1_R_g2T = torch.bmm(g1, torch.bmm(R, g2.transpose(1, 2)))  # [B, dim_u1, dim_z]
    A = torch.bmm(g2, R_g1T)
    Sinv_innov = apply_Sinv(innov)
    mu_upd = mu_pred + torch.bmm(A, Sinv_innov)

    Bmat = apply_Sinv(g1_R_g2T)
    R_upd = symmetrize_jitter(R_pred - torch.bmm(A, Bmat))
    return mu_upd, R_upd


def run_da_single(
    encoder: ERA5Encoder,
    decoder: ERA5Decoder,
    cgn: DiscreteCGN,
    sigma_hat: torch.Tensor,
    sampler: ProbeSampler,
    frames: torch.Tensor,
    observation_schedule: List[int],
    obs_noise_std: float,
    device: torch.device,
    debug: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run DA over one window. frames shape: [T+1, C, H, W] (normalized)."""

    B = 1
    window_length = frames.shape[0] - 1
    sched_set = set(observation_schedule)
    dim_z = cgn.dim_z
    dim_u1 = sampler.dim_u1
    assert sigma_hat.shape[0] == dim_u1 + dim_z, "sigma_hat dim mismatch"

    frames_b = frames.unsqueeze(0).to(device)

    mu0 = encoder(frames_b[:, :1]).squeeze(1).unsqueeze(-1)  # [1, dim_z, 1]
    R0 = 0.01 * torch.eye(dim_z, device=device).unsqueeze(0)  # [1, dim_z, dim_z]

    # Initial probe from t=0 for conditioning
    u1_hold = sampler.sample(frames_b[:, :1])[:, 0]
    mu = mu0
    R = R0

    preds = []
    noda_preds = []

    # No-DA rollout using model-predicted probes
    u_state = torch.cat([u1_hold, mu.squeeze(-1)], dim=-1)
    for k in range(window_length):
        # DA branch
        has_obs = k in sched_set
        u_cond = u1_hold  # coefficients conditioned on last observed probe (t=k when k=0)
        u_obs = None
        if has_obs:
            u_obs = sampler.sample(frames_b[:, k + 1 : k + 2])[:, 0]
            if obs_noise_std > 0:
                u_obs = u_obs + obs_noise_std * torch.randn_like(u_obs)
        # coefficients use u_cond (held), innovation uses current obs if available
        mu, R = apply_filter_step(
            cgn,
            sigma_hat,
            mu,
            R,
            u_cond=u_cond,
            u_obs=u_obs if has_obs else None,
            do_update=has_obs,
        )
        if has_obs:
            u1_hold = u_obs  # update hold for next step conditioning
        # decode_mu returns [B, C, H, W]; drop the batch dimension for metric alignment
        preds.append(decode_mu(decoder, mu).squeeze(0))

        # No-DA branch: pure rollout
        u_next = cgn(u_state)
        noda_mu = u_next[:, dim_u1:]
        noda_preds.append(decode_mu(decoder, noda_mu.unsqueeze(-1)).squeeze(0))
        u_state = u_next

        if debug:
            if not torch.isfinite(mu).all() or not torch.isfinite(R).all():
                print(f"[WARN] Non-finite values at step {k}")
                print(
                    _tensor_summary("mu", mu.cpu()),
                    _tensor_summary("R", R.cpu()),
                    _tensor_summary("u_cond", u_cond.cpu()),
                    _tensor_summary("u_obs" if has_obs else "u_obs(None)", (u_obs if u_obs is not None else torch.tensor(0)).cpu()),
                    sep="\n",
                )
            elif k == 0:
                print(f"Step {k}: u1_hold {u1_hold.shape}, mu {mu.shape}, R {R.shape}")

    da_stack = torch.stack(preds, dim=0)  # [T, C, H, W]
    noda_stack = torch.stack(noda_preds, dim=0)
    return da_stack, noda_stack


def run_multi_da_experiment(
    obs_ratio: float = 0.15,
    obs_noise_std: float = 0.05,
    observation_schedule: list | str | None = None,
    observation_variance: float | None = None,
    window_length: int = 50,
    num_runs: int = 5,
    early_stop_config: Tuple[int, float] = (100, 1e-3),
    start_T: int = 0,
    model_name: str = "discreteCGKN",
    data_path: str = "../../../../data/ERA5/ERA5_data/test_seq_state.h5",
    min_path: str = "../../../../data/ERA5/ERA5_data/min_val.npy",
    max_path: str = "../../../../data/ERA5/ERA5_data/max_val.npy",
    ckpt_prefix: str = "stage2",
    use_channels: Sequence[int] | None = None,
    device: str | None = None,
    debug: bool = False,
):
    device_t = set_device(device)
    set_seed(42)

    schedule = parse_schedule(observation_schedule, window_length)

    dataset = ERA5Dataset(data_path=data_path, seq_length=window_length, min_path=min_path, max_path=max_path)
    denorm = lambda t: safe_denorm(t, dataset)

    results_dir = Path("../../../../results") / model_name / "ERA5"
    ckpt_dir = results_dir

    probe_file = ckpt_dir / f"{ckpt_prefix}_probe_coords.npy"
    sigma_file = ckpt_dir / f"{ckpt_prefix}_sigma_hat.npy"
    if not sigma_file.exists():
        sigma_file = ckpt_dir / "stage1_sigma_hat.npy"
    sigma_hat = torch.from_numpy(np.load(sigma_file)).to(device_t)
    if not torch.isfinite(sigma_hat).all():
        raise RuntimeError(_tensor_summary("sigma_hat", sigma_hat))

    # load channels/dim_z from saved config if present to avoid mismatch
    config_path = ckpt_dir / f"{ckpt_prefix}_config.json"
    if not config_path.exists():
        config_path = ckpt_dir / "stage1_config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if use_channels is None:
            use_channels = cfg.get("use_channels", None)
        dim_z = cfg.get("dim_z", ERA5_settings["state_feature_dim"][-1])
    else:
        dim_z = ERA5_settings["state_feature_dim"][-1]

    channels = list(range(ERA5_settings["state_feature_dim"][0])) if use_channels is None else list(use_channels)
    sampler = prepare_probe_sampler(probe_file, channels)
    dim_u1 = sampler.dim_u1

    encoder, decoder, cgn = load_models(device_t, ckpt_dir, dim_z=dim_z, dim_u1=dim_u1, ckpt_prefix=ckpt_prefix)

    if observation_variance is not None:
        sigma_hat = sigma_hat.clone()
        sigma_hat[:dim_u1] = observation_variance ** 0.5

    da_runs = []
    noda_runs = []
    gt_runs = []

    for i in range(num_runs):
        idx = min(start_T + i, len(dataset) - 1)
        pre_seq, post_seq = dataset[idx]
        full_seq = torch.cat([pre_seq[:1], post_seq], dim=0)  # [T+1, C, H, W]
        da_pred, noda_pred = run_da_single(
            encoder,
            decoder,
            cgn,
            sigma_hat,
            sampler,
            full_seq,
            schedule,
            obs_noise_std,
            device_t,
            debug=debug,
        )
        gt = full_seq[1 : window_length + 1]
        da_runs.append(denorm(da_pred).cpu())
        noda_runs.append(denorm(noda_pred).cpu())
        gt_runs.append(denorm(gt).cpu())

    da_stack = torch.stack(da_runs)  # [num_runs, T, C, H, W]
    noda_stack = torch.stack(noda_runs)
    gt_stack = torch.stack(gt_runs)

    run_metrics = [compute_metrics(da_stack[i], noda_stack[i], gt_stack[i]) for i in range(num_runs)]
    mse_mean = np.mean([rm["mse"] for rm in run_metrics], axis=0)
    mse_std = np.std([rm["mse"] for rm in run_metrics], axis=0)
    rrmse_mean = np.mean([rm["rrmse"] for rm in run_metrics], axis=0)
    rrmse_std = np.std([rm["rrmse"] for rm in run_metrics], axis=0)
    ssim_mean = np.mean([rm["ssim"] for rm in run_metrics], axis=0)
    ssim_std = np.std([rm["ssim"] for rm in run_metrics], axis=0)

    out_dir = results_dir / "DA"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "multi.npy", da_stack[0].detach().numpy())
    np.save(out_dir / "noda.npy", noda_stack[0].detach().numpy())
    np.savez(
        out_dir / "multi_meanstd.npz",
        mse_mean=mse_mean,
        mse_std=mse_std,
        rrmse_mean=rrmse_mean,
        rrmse_std=rrmse_std,
        ssim_mean=ssim_mean,
        ssim_std=ssim_std,
    )

    print(f"Saved DA trajectory to {out_dir / 'multi.npy'}")
    print(f"Saved metrics to {out_dir / 'multi_meanstd.npz'}")
    for key in ["mse", "rrmse", "ssim"]:
        run_values = [m.mean() for m in [rm[key] for rm in run_metrics]]
        print(
            f"{key.upper()} mean over runs: {float(np.mean(run_values)):.6f}, std: {float(np.std(run_values)):.6f}"
        )

    return run_metrics[0]


def main():
    parser = argparse.ArgumentParser(description="Discrete CGKN ERA5 data assimilation")
    parser.add_argument("--obs_ratio", type=float, default=0.15)
    parser.add_argument("--obs_noise_std", type=float, default=0.05)
    parser.add_argument("--observation_schedule", type=str, default=None, help="Comma-separated indices over window steps")
    parser.add_argument("--observation_variance", type=float, default=None)
    parser.add_argument("--window_length", type=int, default=50)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--start_T", type=int, default=0)
    parser.add_argument("--model_name", type=str, default="discreteCGKN")
    parser.add_argument("--data_path", type=str, default="../../../../data/ERA5/ERA5_data/test_seq_state.h5")
    parser.add_argument("--min_path", type=str, default="../../../../data/ERA5/ERA5_data/min_val.npy")
    parser.add_argument("--max_path", type=str, default="../../../../data/ERA5/ERA5_data/max_val.npy")
    parser.add_argument("--ckpt_prefix", type=str, default="stage2")
    parser.add_argument("--use_channels", type=str, default=None, help="Comma separated channel indices; default all")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    channels = None if args.use_channels is None else [int(c) for c in args.use_channels.split(",") if c != ""]

    run_multi_da_experiment(
        obs_ratio=args.obs_ratio,
        obs_noise_std=args.obs_noise_std,
        observation_schedule=args.observation_schedule,
        observation_variance=args.observation_variance,
        window_length=args.window_length,
        num_runs=args.num_runs,
        start_T=args.start_T,
        model_name=args.model_name,
        data_path=args.data_path,
        min_path=args.min_path,
        max_path=args.max_path,
        ckpt_prefix=args.ckpt_prefix,
        use_channels=channels,
        device=args.device,
        debug=args.debug,
    )


if __name__ == "__main__":
    from src.models.CAE_Koopman.ERA5.era5_model_FTF import ERA5_settings

    main()
