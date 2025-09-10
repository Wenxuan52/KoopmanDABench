#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def plot_multi_model_latent_comparison():

    models = [
        ("DMD", "DMD"),
        ("CAE_DMD", "DMD ROM"),
        ("CAE_Koopman", "Koopman ROM"),
        ("CAE_Linear", "Linear ROM"),
        ("CAE_Weaklinear", "Weaklinear ROM"),
        ("CAE_MLP", "MLP ROM")
    ]

    fig = plt.figure(figsize=(20, 25), constrained_layout=False)
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1])

    colors_gt = colors_no_da = colors_da = None

    def style_axes_2d(ax, title):
        ax.set_title(title, fontsize=24, fontweight='bold', pad=12)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for spine in ('left', 'bottom'):
            ax.spines[spine].set_linewidth(1.2)
            ax.spines[spine].set_color('gray')
        ax.tick_params(axis='both', which='major', labelsize=16, width=1.2, length=4)
        ax.set_aspect('equal', adjustable='datalim')

    def style_axes_3d(ax, title):
        ax.set_title(title, fontsize=24, fontweight='bold', pad=1)
        ax.view_init(elev=20, azim=45)
        ax.set_xlabel('', fontsize=18)
        ax.set_ylabel('', fontsize=18)
        ax.set_zlabel('', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.grid(True, alpha=0.5, linestyle='--')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

    for idx, (model_name, model_display_name) in enumerate(models):
        row, col = divmod(idx, 3)
        results_path = f'../../results/{model_name}/DA/latent_space_analysis_data.pkl'

        try:
            with open(results_path, 'rb') as f:
                latent_data = pickle.load(f)
            tsne_2d = latent_data['tsne_2d']
            tsne_3d = latent_data['tsne_3d']
            gt_2d, no_da_2d, da_2d = tsne_2d['ground_truth_2d'], tsne_2d['no_data_assimilation_2d'], tsne_2d['data_assimilation_2d']
            gt_3d, no_da_3d, da_3d = tsne_3d['ground_truth_3d'], tsne_3d['no_data_assimilation_3d'], tsne_3d['data_assimilation_3d']

            if colors_gt is None:
                colors_gt = plt.cm.Greys(np.linspace(0.35, 0.60, len(gt_2d)))
                colors_no_da = plt.cm.Blues(np.linspace(0.35, 0.95, len(no_da_2d)))
                colors_da = plt.cm.Reds(np.linspace(0.35, 0.95, len(da_2d)))

            # 2D
            ax2d = fig.add_subplot(gs[row, col])
            for i in range(len(gt_2d)):
                ax2d.scatter(gt_2d[i, 0], gt_2d[i, 1], c=[colors_gt[i]], s=50, alpha=0.55, marker='o')
            for i in range(len(no_da_2d)):
                ax2d.scatter(no_da_2d[i, 0], no_da_2d[i, 1], c=[colors_no_da[i]], s=50, alpha=0.8, marker='s')
            for i in range(len(da_2d)):
                ax2d.scatter(da_2d[i, 0], da_2d[i, 1], c=[colors_da[i]], s=50, alpha=0.8, marker='^')
            style_axes_2d(ax2d, model_display_name)

            # 3D
            ax3d = fig.add_subplot(gs[row+2, col], projection='3d')
            for i in range(len(gt_3d)):
                ax3d.scatter(gt_3d[i, 0], gt_3d[i, 1], gt_3d[i, 2], c=[colors_gt[i]], s=30, alpha=0.7, marker='o')
            for i in range(len(no_da_3d)):
                ax3d.scatter(no_da_3d[i, 0], no_da_3d[i, 1], no_da_3d[i, 2], c=[colors_no_da[i]], s=40, alpha=0.8, marker='s')
            for i in range(len(da_3d)):
                ax3d.scatter(da_3d[i, 0], da_3d[i, 1], da_3d[i, 2], c=[colors_da[i]], s=50, alpha=0.8, marker='^')
            style_axes_3d(ax3d, model_display_name)

        except FileNotFoundError:
            ax2d = fig.add_subplot(gs[row, col])
            ax2d.text(0.5, 0.5, "Data Not Available", ha='center', va='center', fontsize=18)
            ax3d = fig.add_subplot(gs[row+2, col], projection='3d')
            ax3d.text(0.5, 0.5, 0.5, "Data Not Available", ha='center', va='center', fontsize=18)

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=14, label='One step'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=14, label='No Data Assimilation'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=14, label='4D-Var')
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.07), ncol=3, fontsize=25, frameon=True)

    plt.subplots_adjust(hspace=0.25, wspace=0.18, bottom=0.12, top=0.95)
    save_path = '../../results/Comparison/figures/multi_model_latent_comparison.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pil_kwargs={"optimize": True})




if __name__ == "__main__":
    plot_multi_model_latent_comparison()