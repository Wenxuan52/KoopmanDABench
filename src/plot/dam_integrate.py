import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fig_save_path = '../../results/Comparison/figures/'

img1 = mpimg.imread('../../results/DMD/DA/dam_DMD.png')
img2 = mpimg.imread('../../results/CAE_DMD/DA/dam_CAE_DMD.png')
img3 = mpimg.imread('../../results/CAE_Koopman/DA/dam_CAE_Koopman.png')
img4 = mpimg.imread('../../results/CAE_Linear/DA/dam_CAE_Linear.png')
img5 = mpimg.imread('../../results/CAE_Weaklinear/DA/dam_CAE_Weaklinear.png')
img6 = mpimg.imread('../../results/CAE_MLP/DA/dam_CAE_MLP.png')

# fig, axes = plt.subplots(1, 6, figsize=(14, 10))

# axes[0].imshow(img1)
# axes[1].imshow(img2)
# axes[2].imshow(img3)
# axes[3].imshow(img4)
# axes[4].imshow(img5)
# axes[5].imshow(img6)

fig, axes = plt.subplots(2, 3, figsize=(10, 9))

axes[0, 0].imshow(img1)
axes[0, 0].axis('off')
axes[0, 1].imshow(img2)
axes[0, 1].axis('off')
axes[0, 2].imshow(img3)
axes[0, 2].axis('off')
axes[1, 0].imshow(img4)
axes[1, 0].axis('off')
axes[1, 1].imshow(img5)
axes[1, 1].axis('off')
axes[1, 2].imshow(img6)
axes[1, 2].axis('off')

plt.subplots_adjust(wspace=0.05, hspace=0.05)

plt.savefig(f'{fig_save_path}dam_4dvar_integrate.png', dpi=300, bbox_inches='tight')