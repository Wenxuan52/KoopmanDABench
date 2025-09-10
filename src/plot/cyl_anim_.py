import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from datetime import datetime, timedelta
from matplotlib import gridspec

current_directory = os.getcwd()
src_directory = os.path.abspath(os.path.join(current_directory, "..", ".."))
sys.path.append(src_directory)

from src.utils.Dataset import CylinderDynamicsDataset


def _to_numpy(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def compute_velocity_magnitude(data):
    """Compute velocity magnitude from u,v components."""
    u = data[:, 0, ...]  # u component
    v = data[:, 1, ...]  # v component
    return np.sqrt(u**2 + v**2)


def load_animation_data(model_name, results_path="../../results"):
    """Load animation data for a specific model."""
    data_path = f"{results_path}/{model_name}/DA/cyl_animation_data.npz"
    if not os.path.exists(data_path):
        print(f"Warning: {data_path} not found, returning None")
        return None, None
    
    data = np.load(data_path, allow_pickle=True)
    no_da_data = data['no_da_reconstructions']  # (100, 2, 64, 64)
    da_data = data['da_reconstructions']        # (100, 2, 64, 64)
    
    # Convert to velocity magnitude
    no_da_mag = compute_velocity_magnitude(no_da_data).reshape(-1, 1, 64, 64)
    da_mag = compute_velocity_magnitude(da_data).reshape(-1, 1, 64, 64)
    
    return no_da_mag, da_mag


def compute_row_ranges(datas, time_indices):
    """Return per-channel (vmin, vmax)."""
    vals = []
    for d in datas:
        if d is not None:
            vals.append(d[time_indices])
    if len(vals) == 0:
        return 0, 1
    vals = np.concatenate(vals, axis=0)
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    return vmin, vmax


def make_cylinder_da_gif(
    groundtruth, 
    dmd_no_da, dmd_da,
    cae_dmd_no_da, cae_dmd_da,
    cae_koopman_no_da, cae_koopman_da,
    cae_linear_no_da, cae_linear_da,
    cae_weaklinear_no_da, cae_weaklinear_da,
    cae_mlp_no_da, cae_mlp_da,
    out_path="figures/cylinder_da_comparison_4x7.gif",
    time_indices=None,
    start_timestep=801,
    timesteps_per_frame=1,
    titles=('Groundtruth','DMD','DMD ROM','Koopman ROM','Linear ROM','Weaklinear ROM','MLP ROM'),
    cmap="viridis",
    cmap_err="magma",
    fps=2
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 转换为numpy并准备velocity magnitude
    # groundtruth已经是velocity magnitude格式 [T, 1, H, W]
    if isinstance(groundtruth, torch.Tensor):
        gt = groundtruth.numpy()
    else:
        gt = groundtruth
    
    # 组织数据：每行一种类型，每列一个模型
    no_da_models = [gt, dmd_no_da, cae_dmd_no_da, cae_koopman_no_da, cae_linear_no_da, cae_weaklinear_no_da, cae_mlp_no_da]
    da_models = [gt, dmd_da, cae_dmd_da, cae_koopman_da, cae_linear_da, cae_weaklinear_da, cae_mlp_da]

    T = gt.shape[0]
    if time_indices is None:
        time_indices = list(range(min(100, T)))

    # 计算原始值范围（参考您的代码中的方式）
    all_pred_data = []
    for models in [no_da_models, da_models]:
        for d in models:
            if d is not None:
                all_pred_data.append(d)
    all_pred_data = np.concatenate(all_pred_data, axis=0)
    vmin_pred, vmax_pred = all_pred_data.min(), all_pred_data.max()

    # 计算error范围
    errors = []
    for t in time_indices:
        raw_img = gt[t, 0]  # [H, W]
        # NoDA errors
        for j in range(1, 7):  # 跳过groundtruth
            if no_da_models[j] is not None:
                err = np.abs(no_da_models[j][t, 0] - raw_img)
                errors.append(err)
        # DA errors  
        for j in range(1, 7):  # 跳过groundtruth
            if da_models[j] is not None:
                err = np.abs(da_models[j][t, 0] - raw_img)
                errors.append(err)
    
    if len(errors) > 0:
        errors_all = np.stack(errors)
        vmin_err, vmax_err = errors_all.min(), errors_all.max()
    else:
        vmin_err, vmax_err = 0, 1

    # 网格设置：4行（NoDA + NoDA Error + DA + DA Error），7列
    total_rows = 4
    ncols = 7
    fig = plt.figure(figsize=(ncols * 4.0, total_rows * 3.0))
    gs = gridspec.GridSpec(
        total_rows, ncols + 2,
        width_ratios=[0.6] + [1]*ncols + [0.08],
        wspace=0.12, hspace=0.15, top=0.82
    )

    BIG, MID = 20, 16

    # 建立axes
    axes_img = [[None for _ in range(ncols)] for __ in range(total_rows)]
    axes_label = []
    axes_cbar = []

    row_labels = ["No DA", "No DA Error", "DA", "DA Error"]
    
    for r in range(total_rows):
        # 左侧标签
        ax_label = fig.add_subplot(gs[r, 0])
        ax_label.axis("off")
        
        ax_label.text(0.5, 0.5, row_labels[r],
                      fontsize=BIG, fontweight="bold",
                      va="center", ha="center", rotation=0)
        axes_label.append(ax_label)

        # 图像axes
        for j in range(ncols):
            ax = fig.add_subplot(gs[r, j+1])
            ax.axis("off")
            axes_img[r][j] = ax

        # colorbar axes
        ax_cbar = fig.add_subplot(gs[r, -1])
        axes_cbar.append(ax_cbar)

    # 顶部标题（只在第一行显示）
    for j in range(ncols):
        axes_img[0][j].set_title(titles[j], fontsize=BIG, fontweight="bold", pad=16)

    # 初始化图像
    images = [[None for _ in range(ncols)] for __ in range(total_rows)]
    t0 = time_indices[0]

    for r in range(total_rows):
        is_error_row = (r == 1 or r == 3)  # NoDA Error 或 DA Error
        is_da_row = (r == 2 or r == 3)     # DA 或 DA Error
        
        # 选择数据源
        if is_da_row:
            current_models = da_models
        else:
            current_models = no_da_models
        
        # 选择colormap和范围
        if not is_error_row:
            vmin, vmax = vmin_pred, vmax_pred
            this_cmap = cmap
        else:
            vmin, vmax = vmin_err, vmax_err
            this_cmap = cmap_err

        for j in range(ncols):
            # error行的第一列（groundtruth）不显示
            if is_error_row and j == 0:
                axes_img[r][j].axis("off")
                images[r][j] = None
                continue

            # 检查数据是否存在
            if current_models[j] is None:
                axes_img[r][j].axis("off")
                axes_img[r][j].text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=14)
                images[r][j] = None
                continue

            if not is_error_row:
                # 显示velocity magnitude
                arr = current_models[j][t0, 0]  # [H, W]
            else:
                # 显示误差
                raw_img = gt[t0, 0]
                arr = np.abs(current_models[j][t0, 0] - raw_img)

            images[r][j] = axes_img[r][j].imshow(arr, cmap=this_cmap, vmin=vmin, vmax=vmax)

        # 每行的colorbar
        mappable = None
        for j in range(ncols):
            if images[r][j] is not None:
                mappable = images[r][j]
                break
        if mappable is not None:
            cbar = fig.colorbar(mappable, cax=axes_cbar[r])
            cbar.ax.tick_params(labelsize=MID)
        else:
            axes_cbar[r].axis("off")

    # 时间管理
    def frame_timestep(frame_idx):
        return start_timestep + timesteps_per_frame * frame_idx

    suptitle = fig.suptitle(
        f"Cylinder Flow DA Comparison — Timestep {frame_timestep(0)}",
        fontsize=BIG+4, fontweight="bold", y=0.95
    )

    def update(frame_idx):
        t = time_indices[frame_idx]
        updated = []
        
        for r in range(total_rows):
            is_error_row = (r == 1 or r == 3)
            is_da_row = (r == 2 or r == 3)
            
            # 选择数据源
            if is_da_row:
                current_models = da_models
            else:
                current_models = no_da_models
            
            if not is_error_row:
                vmin, vmax = vmin_pred, vmax_pred
            else:
                vmin, vmax = vmin_err, vmax_err

            for j in range(ncols):
                im = images[r][j]
                if im is None:
                    continue
                
                if current_models[j] is None:
                    continue
                    
                if not is_error_row:
                    arr = current_models[j][t, 0]
                else:
                    raw_img = gt[t, 0]
                    arr = np.abs(current_models[j][t, 0] - raw_img)
                    
                im.set_data(arr)
                im.set_clim(vmin, vmax)
                updated.append(im)

        suptitle.set_text(f"Cylinder Flow DA Comparison — Timestep {frame_timestep(frame_idx)}")
        return updated + [suptitle]

    ani = FuncAnimation(fig, update, frames=len(time_indices), interval=1000, blit=False)
    writer = PillowWriter(fps=fps)
    ani.save(out_path, writer=writer)
    plt.close(fig)
    print(f"[INFO] GIF saved to: {out_path}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fig_save_path = "../../results/Comparison/figures/"
    os.makedirs(fig_save_path, exist_ok=True)
    gif_path = os.path.join(fig_save_path, "cylinder_da_comparison_4x7.gif")

    start_T = 801  # 从801开始，对应保存的动画数据
    prediction_step = 100  # 动画数据有100帧
    val_idx = 3
    
    # Load dataset for groundtruth
    print("[INFO] Loading datasets...")
    cyl_train_dataset = CylinderDynamicsDataset(
        data_path="../../data/cylinder/cylinder_train_data.npy",
        seq_length=12,
        mean=None,
        std=None
    )
    
    cyl_val_dataset = CylinderDynamicsDataset(
        data_path="../../data/cylinder/cylinder_val_data.npy",
        seq_length=12,
        mean=cyl_train_dataset.mean,
        std=cyl_train_dataset.std
    )

    # Prepare groundtruth (从801开始的100帧)
    raw_test_data = cyl_val_dataset.data
    groundtruth = raw_test_data[val_idx, start_T:start_T + prediction_step, ...]
    groundtruth = torch.tensor(groundtruth, dtype=torch.float32)
    groundtruth_data = (groundtruth[:, 0, :, :] ** 2 + groundtruth[:, 1, :, :] ** 2) ** 0.5
    groundtruth_data = groundtruth_data.unsqueeze(1)

    # Load animation data for each model
    print("[INFO] Loading animation data...")
    model_names = ["DMD", "CAE_DMD", "CAE_Koopman", "CAE_Linear", "CAE_Weaklinear", "CAE_MLP"]
    
    no_da_data_list = []
    da_data_list = []
    
    for model_name in model_names:
        no_da_data, da_data = load_animation_data(model_name)
        no_da_data_list.append(no_da_data)
        da_data_list.append(da_data)
        
        if no_da_data is not None:
            print(f"{model_name} - No DA data shape: {no_da_data.shape}")
            print(f"{model_name} - DA data shape: {da_data.shape}")
        else:
            print(f"{model_name} - No animation data found")

    # 打印各结果数组的shape
    print(f"groundtruth_data shape: {groundtruth_data.shape}")
    
    make_cylinder_da_gif(
        groundtruth_data,
        no_da_data_list[0], da_data_list[0],  # DMD
        no_da_data_list[1], da_data_list[1],  # CAE_DMD
        no_da_data_list[2], da_data_list[2],  # CAE_Koopman
        no_da_data_list[3], da_data_list[3],  # CAE_Linear
        no_da_data_list[4], da_data_list[4],  # CAE_Weaklinear
        no_da_data_list[5], da_data_list[5],  # CAE_MLP
        out_path=gif_path,
        start_timestep=801,
        timesteps_per_frame=1,
        cmap="viridis",
        cmap_err="magma",
        fps=10
    )