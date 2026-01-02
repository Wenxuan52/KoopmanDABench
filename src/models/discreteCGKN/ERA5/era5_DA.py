"""Discrete CGKN data assimilation on ERA5 with conditional dynamics.

This script mirrors the ERA5 discrete CGKN training artifacts and applies the
NSE-style closed-form predict/update (coefficients conditioned on the held
probe) over a fixed assimilation window. Observation schedule is defined over
steps k=0..window_length-1, where step k corresponds to physical time t=k+1.
At non-scheduled steps, the probe is advanced by the model prediction without
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

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_config")

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


def stabilize_cov(R: torch.Tensor, jitter: float = 1e-6, clamp_min: float = 1e-8, clamp_max: float | None = None) -> torch.Tensor:
    eye = torch.eye(R.shape[-1], device=R.device).unsqueeze(0)
    R = 0.5 * (R + R.transpose(-1, -2)) + jitter * eye
    needs_eig = (
        torch.isnan(R).any()
        or torch.isinf(R).any()
        or (R.diagonal(dim1=-2, dim2=-1) <= 0).any()
    )
    if needs_eig:
        evals, evecs = torch.linalg.eigh(R)
        evals = evals.clamp_min(clamp_min)
        if clamp_max is not None:
            evals = evals.clamp_max(clamp_max)
        R = (evecs * evals.unsqueeze(-2)) @ evecs.transpose(-2, -1)
        R = 0.5 * (R + R.transpose(-1, -2)) + jitter * eye
    return R


def solve_psd(mat: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    mat = 0.5 * (mat + mat.transpose(-1, -2))
    eye = torch.eye(mat.shape[-1], device=mat.device).unsqueeze(0)
    jitters = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
    for jitter in jitters:
        mat_try = mat + jitter * eye
        L, info = torch.linalg.cholesky_ex(mat_try)
        if int(info.max().item()) == 0:
            return torch.cholesky_solve(rhs, L)
    return torch.linalg.solve(mat + 1e-1 * eye, rhs)


def apply_filter_step(
    cgn: DiscreteCGN,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    R: torch.Tensor,
    u_cond: torch.Tensor,
    u_next: torch.Tensor | None,
    do_update: bool,
    obs_noise_std: float,
    mats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Predict/update with coefficients conditioned on held u1 and innovation on u1_next.

    Args:
        mu: [B, dim_z, 1]
        R: [B, dim_z, dim_z]
        u_cond: [B, dim_u1] probe used to produce (f,g)
        u_next: [B, dim_u1] probe at next step (observation or model prediction)
        do_update: whether to apply the innovation-based correction
    """

    dim_u1 = cgn.dim_u1
    dim_z = cgn.dim_z
    sigma_u1 = sigma[:dim_u1]
    sigma_z = sigma[dim_u1:]
    sigma_u1_eff = torch.sqrt(torch.clamp(sigma_u1**2 + obs_noise_std**2, min=0.0))

    R_prior = stabilize_cov(R)
    if mats is None:
        f1, g1, f2, g2 = cgn.get_cgn_mats(u_cond)
    else:
        f1, g1, f2, g2 = mats

    mu_pred = f2 + torch.bmm(g2, mu)
    process_cov = torch.diag_embed(sigma_z**2).to(R_prior).unsqueeze(0)
    R_pred = stabilize_cov(torch.bmm(torch.bmm(g2, R_prior), g2.transpose(1, 2)) + process_cov)

    if (not do_update) or u_next is None:
        return mu_pred, R_pred

    innov = u_next.unsqueeze(-1) - (f1 + torch.bmm(g1, mu))
    S = torch.diag_embed(sigma_u1_eff**2).to(R_prior).unsqueeze(0) + torch.bmm(torch.bmm(g1, R_prior), g1.transpose(1, 2))
    Sinv_innov = solve_psd(S, innov)

    A = torch.bmm(g2, torch.bmm(R_prior, g1.transpose(1, 2)))
    mu_upd = mu_pred + torch.bmm(A, Sinv_innov)

    g1_R_g2T = torch.bmm(g1, torch.bmm(R_prior, g2.transpose(1, 2)))
    Bmat = solve_psd(S, g1_R_g2T)
    R_upd = stabilize_cov(R_pred - torch.bmm(A, Bmat))
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

    frames = torch.nan_to_num(frames, nan=0.0, posinf=0.0, neginf=0.0)
    frames_b = frames.unsqueeze(0).to(device)

    mu = encoder(frames_b[:, :1]).squeeze(1).unsqueeze(-1)  # [1, dim_z, 1]
    R = 0.01 * torch.eye(dim_z, device=device).unsqueeze(0)  # [1, dim_z, dim_z]

    u_cur = sampler.sample(frames_b[:, :1])[:, 0]
    if debug:
        print(
            "[DEBUG] Initial states",
            _tensor_summary("mu", mu),
            _tensor_summary("u_cur", u_cur),
            _tensor_summary("sigma_hat", sigma_hat),
            sep="\n",
        )
    preds: List[torch.Tensor] = []
    noda_preds: List[torch.Tensor] = []

    u_state = torch.cat([u_cur, mu.squeeze(-1)], dim=-1)
    for k in range(window_length):
        has_obs = k in sched_set
        f1, g1, f2, g2 = cgn.get_cgn_mats(u_cur)
        u_pred_next = (f1 + torch.bmm(g1, mu)).squeeze(-1)

        u_obs_next = None
        if has_obs:
            u_obs_next = sampler.sample(frames_b[:, k + 1 : k + 2])[:, 0]
            if obs_noise_std > 0:
                u_obs_next = u_obs_next + obs_noise_std * torch.randn_like(u_obs_next)
        u_next = u_obs_next if has_obs else u_pred_next.detach()

        mu, R = apply_filter_step(
            cgn,
            sigma_hat,
            mu,
            R,
            u_cond=u_cur,
            u_next=u_next,
            do_update=has_obs,
            obs_noise_std=obs_noise_std,
            mats=(f1, g1, f2, g2),
        )
        u_cur = u_next.detach()

        if debug:
            if (not torch.isfinite(mu).all()) or (not torch.isfinite(R).all()):
                print(f"[ERROR] Non-finite filter state at step {k}")
                print(
                    _tensor_summary("mu", mu.cpu()),
                    _tensor_summary("R", R.cpu()),
                    _tensor_summary("u_cond", u_cur.cpu()),
                    _tensor_summary("u_next", u_next.cpu()),
                    sep="\n",
                )
                raise ValueError(f"Non-finite filter state at step {k}")

        da_frame = decode_mu(decoder, mu).squeeze(0)
        da_frame = torch.nan_to_num(da_frame, nan=0.0, posinf=0.0, neginf=0.0)
        preds.append(da_frame)

        u_state = cgn(u_state)
        noda_mu = u_state[:, dim_u1:]
        noda_frame = decode_mu(decoder, noda_mu.unsqueeze(-1)).squeeze(0)
        noda_frame = torch.nan_to_num(noda_frame, nan=0.0, posinf=0.0, neginf=0.0)
        noda_preds.append(noda_frame)

    da_stack = torch.stack(preds, dim=0)
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
    if debug:
        print(
            f"[DEBUG] Dataset loaded: len={len(dataset)}, seq_length={window_length}, "
            f"min finite={np.isfinite(dataset.min).all()}, max finite={np.isfinite(dataset.max).all()}"
        )
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
    if debug:
        print(
            f"[DEBUG] Loaded sigma_hat from {sigma_file}",
            _tensor_summary("sigma_hat", sigma_hat),
            f"Probe file: {probe_file}",
            f"Schedule: {schedule}",
            sep="\n",
        )

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

    da_runs: List[torch.Tensor] = []
    noda_runs: List[torch.Tensor] = []
    gt_runs: List[torch.Tensor] = []

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
        denorm_da = torch.nan_to_num(denorm(da_pred), nan=0.0, posinf=0.0, neginf=0.0)
        denorm_noda = torch.nan_to_num(denorm(noda_pred), nan=0.0, posinf=0.0, neginf=0.0)
        denorm_gt = torch.nan_to_num(denorm(gt), nan=0.0, posinf=0.0, neginf=0.0)
        da_runs.append(denorm_da.cpu())
        noda_runs.append(denorm_noda.cpu())
        gt_runs.append(denorm_gt.cpu())
        if debug:
            print(
                f"[DEBUG] Run {i}:",
                _tensor_summary("denorm_da", denorm_da),
                _tensor_summary("denorm_noda", denorm_noda),
                _tensor_summary("denorm_gt", denorm_gt),
                sep="\n",
            )

    da_stack = torch.stack(da_runs)
    noda_stack = torch.stack(noda_runs)
    gt_stack = torch.stack(gt_runs)

    run_metrics: List[Dict[str, np.ndarray]] = []
    for i in range(num_runs):
        metrics_i = compute_metrics(da_stack[i], noda_stack[i], gt_stack[i])
        for key in metrics_i:
            if not np.isfinite(metrics_i[key]).all():
                if debug:
                    mask = ~np.isfinite(metrics_i[key])
                    bad_idx = np.argwhere(mask)
                    print(f"[WARN] Non-finite {key} entries for run {i}, first positions: {bad_idx[:5]}")
                metrics_i[key] = np.nan_to_num(metrics_i[key], nan=0.0, posinf=0.0, neginf=0.0)
        run_metrics.append(metrics_i)
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
