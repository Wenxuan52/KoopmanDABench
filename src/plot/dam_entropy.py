import os
import glob
import argparse
from signal import Sigmasks

import torch
import numpy as np
import matplotlib.pyplot as plt


def load_entropy_snapshots(data_dir):
    """
    从 data_dir 中加载所有 entropy_epochXXX.pt 文件，
    返回按 epoch 排序的记录列表。
    每个记录包含: epoch, Sigma (np.array), evals (np.array), S, trace, eig_min, eig_max
    """
    pattern = os.path.join(data_dir, "entropy_epoch*.pt")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern}")

    records = []
    for path in files:
        # 如果你的 torch 版本不支持 weights_only=True，就去掉这个参数
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        # epoch 信息从文件中读，如果没有就从文件名解析
        epoch = int(ckpt.get("epoch", _parse_epoch_from_name(path)))
        Sigma = ckpt["Sigma"].numpy()   # [H, H]
        evals = ckpt["evals"].numpy()   # [H]

        # 计算 von Neumann entropy: S = - sum lambda log lambda
        eps = 1e-12
        evals_clamped = np.clip(evals, eps, None)
        S = -np.sum(evals_clamped * np.log(evals_clamped))

        rec = {
            "epoch": epoch,
            "Sigma": Sigma,
            "evals": evals,
            "entropy": S,
            "trace": float(np.trace(Sigma)),
            "eig_min": float(evals.min()),
            "eig_max": float(evals.max()),
        }
        records.append(rec)

    # 按 epoch 排序
    records.sort(key=lambda r: r["epoch"])
    return records


def _parse_epoch_from_name(path):
    # 例如 entropy_epoch010.pt -> 10
    base = os.path.basename(path)
    # 找到数字部分
    digits = "".join(ch for ch in base if ch.isdigit())
    return int(digits)


def plot_cov_heatmaps_grid(records, output_dir):
    """
    生成一个 2x5 的大图并保存所有 Sigma（缩放后的）到一个 npy 文件。
    """
    os.makedirs(output_dir, exist_ok=True)

    epochs_grid = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    epoch_to_record = {r["epoch"]: r for r in records}

    mats = []
    for e in epochs_grid:
        rec = epoch_to_record.get(e, None)
        if rec is not None:
            mats.append(rec["Sigma"])
    if len(mats) == 0:
        print("[WARN] No matching epochs found for heatmap grid.")
        return

    all_vals = np.concatenate([m.flatten() for m in mats])
    vmin_raw = all_vals.min()
    vmax_raw = all_vals.max()

    if vmin_raw < 0 and vmax_raw > 0:
        bound = abs(vmin_raw)
        vmin = -bound
        vmax = bound
    else:
        vmin = vmin_raw
        vmax = vmax_raw

    fig, axes = plt.subplots(2, 5, figsize=(5 * 2.4, 2 * 2.4))

    im = None

    # 用于保存所有缩放后的 Sigma（按 epoch 顺序）
    saved_sigmas = []

    for idx, epoch in enumerate(epochs_grid):
        row = idx // 5
        col = idx % 5
        ax = axes[row, col]

        rec = epoch_to_record.get(epoch, None)
        if rec is None:
            ax.axis("off")
            ax.set_title(f"Epoch {epoch} (missing)", fontsize=10)
            continue

        Sigma = rec["Sigma"]

        # 缩放以便可视化
        save_Sigma = np.abs(Sigma / vmax * 6.6)

        # 加入保存列表
        saved_sigmas.append({"epoch": epoch, "Sigma": save_Sigma})

        im = ax.imshow(
            save_Sigma,
            aspect="equal",
            cmap="Blues",
            origin="upper",
            vmin=0,
            vmax=6.8,
        )

        ax.set_title(f"Epoch {epoch}", fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.subplots_adjust(wspace=0.1, hspace=0.2)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    # cbar.ax.set_ylabel("Covariance", rotation=270, labelpad=12)

    out_path = os.path.join(output_dir, "cov_heatmap1.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved covariance heatmap grid to {out_path}")

    # ===== 新增：保存所有 save_Sigma 到 npy 文件 =====
    save_path = os.path.join(output_dir, "sigmas2.npy")
    np.save(save_path, saved_sigmas)
    print(f"[INFO] Saved all Sigma matrices to {save_path}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../../results/CAE_MI/Dam/entropy_visual1",
        help="Directory containing entropy_epochXXX.pt files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../temp_results",
        help="Directory to save output figures",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Top-K eigenvalues to show in eigen spectra plot",
    )
    args = parser.parse_args()

    records = load_entropy_snapshots(args.data_dir)

    # 1) 2x5 的 covariance heatmap 大图
    plot_cov_heatmaps_grid(records, args.output_dir)

    # 2) eigenvalue 谱（蓝色渐变）
    # plot_eigen_spectra_blue_gradient(records, args.output_dir, top_k=args.top_k)

    print(f"Plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()