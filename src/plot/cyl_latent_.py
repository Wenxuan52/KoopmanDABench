#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.ticker import MaxNLocator
import os

def plot_multi_model_latent_comparison():
    
    models = [
        ("CAE_MLP", "VAE", "no_da", "standard"),
        ("CAE_Linear", "KAE", "no_da", "standard"),        
        ("CAE_Koopman", "KKR", "no_da", "rollout"),
        ("CAE_DMD", "PFNN", "da", "standard"),
        ("CAE_Koopman", "Ours", "no_da", "standard")
    ]
    
    # 创建图形：1行5列
    fig = plt.figure(figsize=(25, 5))
    
    # 颜色
    color_gt = 'gray'
    color_blue = 'blue'
    
    def style_axes_3d(ax, title):
        ax.set_title(title, fontsize=32, fontweight='bold', pad=10)
        ax.view_init(elev=20, azim=45)

        # === 轴标签（按需显示 x, y, z）===
        ax.set_xlabel('x', fontsize=26, labelpad=-8)
        ax.set_ylabel('y', fontsize=26, labelpad=-8)
        ax.set_zlabel('z', fontsize=26, labelpad=-8)

        # === 网格开启 ===
        ax.grid(True, alpha=0.4, linestyle='--')

        # 面板透明，更干净
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # 坐标轴线颜色略淡
        ax.xaxis.line.set_color('gray')
        ax.yaxis.line.set_color('gray')
        ax.zaxis.line.set_color('gray')

        # 设定刻度位置（需要有刻度才能保留网格），但隐藏刻度“数值”
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))  # 不强制整数，避免 3D 轴冲突
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=5))

        # 隐藏刻度标签的数字（保留网格与刻度线）
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # 也可以进一步收紧刻度标签的占位空间
        ax.tick_params(axis='both', which='major', pad=0, length=4)

    for idx, (model_name, display_title, data_type, format_type) in enumerate(models):
        # 创建子图
        ax = fig.add_subplot(1, 5, idx + 1, projection='3d')
        
        try:
            if format_type == "rollout":
                results_path = f'../../results/{model_name}/DA/rollout_latent_space_analysis_data.pkl'
                with open(results_path, 'rb') as f:
                    latent_data = pickle.load(f)

                if 'tsne_3d' in latent_data:
                    tsne_3d = latent_data['tsne_3d']
                    gt_3d = tsne_3d.get('ground_truth_3d', tsne_3d.get('ground_truth', None))
                    autoregressive_3d = tsne_3d.get('state_propagation_3d', tsne_3d.get('state_propagation', None))
                else:
                    print(f"Warning: No t-SNE results found for {model_name}, using original latents")
                    gt_3d = latent_data['original_latents']['ground_truth']
                    autoregressive_3d = latent_data['original_latents']['latent_propagation']
            else:
                # 处理标准格式的数据
                results_path = f'../../results/{model_name}/DA/latent_space_analysis_data.pkl'
                with open(results_path, 'rb') as f:
                    latent_data = pickle.load(f)
                
                # 获取3D数据
                tsne_3d = latent_data['tsne_3d']
                gt_3d = tsne_3d['ground_truth_3d']
                
                # 根据数据类型选择自回归数据
                if data_type == 'da':
                    autoregressive_3d = tsne_3d['data_assimilation_3d']
                else:
                    autoregressive_3d = tsne_3d['no_data_assimilation_3d']
            
            # Onestep Prediction（灰色圆）
            if gt_3d is not None:
                ax.scatter(
                    gt_3d[:, 0], gt_3d[:, 1], gt_3d[:, 2],
                    c=color_gt, s=30, alpha=0.4, marker='o',
                    edgecolors='none', label='Onestep Prediction'
                )
            
            # Autoregressive Prediction（蓝色方）
            if autoregressive_3d is not None:
                ax.scatter(
                    autoregressive_3d[:, 0], autoregressive_3d[:, 1], autoregressive_3d[:, 2],
                    c=color_blue, s=40, alpha=0.5, marker='s',
                    edgecolors='none', label='Autoregressive Prediction'
                )
            
            # 应用样式
            style_axes_3d(ax, display_title)
            
            # 如果是 Ours，限制 z 轴范围
            if display_title == "Ours":
                ax.set_zlim(-12, 12)
            
        except FileNotFoundError as e:
            print(f"File not found for {model_name}: {e}")
            ax.text2D(0.5, 0.5, "Data Not Available",
                      transform=ax.transAxes, ha='center', va='center',
                      fontsize=16, color='red')
            style_axes_3d(ax, display_title)
        except KeyError as e:
            print(f"Key error for {model_name}: {e}")
            ax.text2D(0.5, 0.5, f"Missing key: {str(e)[:20]}",
                      transform=ax.transAxes, ha='center', va='center',
                      fontsize=12, color='red')
            style_axes_3d(ax, display_title)
        except Exception as e:
            print(f"Error for {model_name}: {e}")
            ax.text2D(0.5, 0.5, f"Error: {str(e)[:20]}",
                      transform=ax.transAxes, ha='center', va='center',
                      fontsize=12, color='red')
            style_axes_3d(ax, display_title)
    
    # 统一图例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_gt,
                   markersize=14, alpha=0.4, label='Onestep Prediction'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color_blue,
                   markersize=14, alpha=0.5, label='Autoregressive Prediction')
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               bbox_to_anchor=(0.5, -0.05), ncol=2,
               fontsize=28, frameon=True, fancybox=True, shadow=True)
    
    # 布局&保存
    plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.08, wspace=0.15)
    save_path = '../../results/Comparison/figures/multi_model_latent_comparison_3d.pdf'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.show()
    print(f"Figure saved to: {save_path}")

if __name__ == "__main__":
    plot_multi_model_latent_comparison()
