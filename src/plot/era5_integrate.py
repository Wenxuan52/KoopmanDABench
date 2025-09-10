import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

fig_save_path = '../../results/Comparison/figures/'

# 读取图片
img1 = mpimg.imread('../../results/Comparison/figures/era5_Geopotential_4dvar_comparison.png')
img2 = mpimg.imread('../../results/Comparison/figures/era5_Humidity_4dvar_comparison.png')
img3 = mpimg.imread('../../results/Comparison/figures/era5_Temperature_4dvar_comparison.png')
img4 = mpimg.imread('../../results/Comparison/figures/era5_U_wind_4dvar_comparison.png')
img5 = mpimg.imread('../../results/Comparison/figures/era5_V_wind_4dvar_comparison.png')

# 定义一个裁剪函数，默认裁掉底部 20%
def crop_bottom(img, ratio=0.2):
    h = img.shape[0]
    crop_h = int(h * (1 - ratio))
    return img[:crop_h, :, :]

crop = 0.0

# 对前四张图裁剪
img1_cropped = crop_bottom(img1, crop)
img2_cropped = crop_bottom(img2, crop)
img3_cropped = crop_bottom(img3, crop)
img4_cropped = crop_bottom(img4, crop)

# 不裁剪最后一张
img5_cropped = img5  

# 拼图
fig, axes = plt.subplots(5, 1, figsize=(11, 15))

images = [img1_cropped, img2_cropped, img3_cropped, img4_cropped, img5_cropped]

for ax, img in zip(axes, images):
    ax.imshow(img)
    ax.axis('off')

plt.subplots_adjust(wspace=0.05, hspace=0.05)

plt.savefig(f'era5_integrate.png', dpi=100, bbox_inches='tight')
plt.close()
