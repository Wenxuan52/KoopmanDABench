import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fig_save_path = '../../results/Comparison/figures/'

img1 = mpimg.imread('../../results/DMD/DA/kol_DMD.png')
img2 = mpimg.imread('../../results/CAE_DMD/DA/kol_CAE_DMD.png')
img3 = mpimg.imread('../../results/CAE_Linear/DA/kol_CAE_Linear.png')
img4 = mpimg.imread('../../results/CAE_Weaklinear/DA/kol_CAE_Weaklinear.png')

fig, axes = plt.subplots(1, 4, figsize=(10, 4))

axes[0].imshow(img1)
axes[1].imshow(img2)
axes[2].imshow(img3)
axes[3].imshow(img4)

for ax in axes:
    ax.axis('off')

plt.subplots_adjust(wspace=0.10)

plt.savefig(f'{fig_save_path}kol_4dvar_integrate.png', dpi=300, bbox_inches='tight')