import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fig_save_path = '../../results/Comparison/figures/'

img1 = mpimg.imread('../../results/DMD/DA/latent_space_3d_tsne.png')
img2 = mpimg.imread('../../results/CAE_DMD/DA/latent_space_3d_tsne.png')
img3 = mpimg.imread('../../results/CAE_Koopman/DA/latent_space_3d_tsne.png')
img4 = mpimg.imread('../../results/CAE_Linear/DA/latent_space_3d_tsne.png')
img5 = mpimg.imread('../../results/CAE_Weaklinear/DA/latent_space_3d_tsne.png')
img6 = mpimg.imread('../../results/CAE_MLP/DA/latent_space_3d_tsne.png')

crop = 480 # 90
pad = 5
size = 6

titles = ['DMD', 'DMD ROM', 'Koopman ROM', 'Linear ROM', 'Weaklinear ROM', 'MLP ROM']

fig, axes = plt.subplots(2, 3, figsize=(8, 5))

axes[0, 0].imshow(img1[crop:, :])
axes[0, 0].axis('off')
axes[0, 0].set_title(titles[0], fontsize=size, pad=pad, fontweight='bold')

axes[0, 1].imshow(img2[crop:, :])
axes[0, 1].axis('off')
axes[0, 1].set_title(titles[1], fontsize=size, pad=pad, fontweight='bold')

axes[0, 2].imshow(img3[crop:, :])
axes[0, 2].axis('off')
axes[0, 2].set_title(titles[2], fontsize=size, pad=pad, fontweight='bold')

axes[1, 0].imshow(img4[crop:, :])
axes[1, 0].axis('off')
axes[1, 0].set_title(titles[3], fontsize=size, pad=pad, fontweight='bold')

axes[1, 1].imshow(img5[crop:, :])
axes[1, 1].axis('off')
axes[1, 1].set_title(titles[4], fontsize=size, pad=pad, fontweight='bold')

axes[1, 2].imshow(img6[crop:, :])
axes[1, 2].axis('off')
axes[1, 2].set_title(titles[5], fontsize=size, pad=pad, fontweight='bold')

plt.subplots_adjust(wspace=0.05, hspace=0.09)

plt.savefig(f'{fig_save_path}cyl_4dvar_latent_3d.png', dpi=300, bbox_inches='tight')