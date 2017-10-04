#import helper functions
from helper_to_read_files import *
from helper_to_select_roi import *
from helper_to_get_metrics import *
from helper_to_noise_algorithm import *


###Run algorithms on sharp edge image
sharp_edge_image = create_sharp_edge_image(N=768,K=1024)

print(sharp_edge_image.shape)

gauss_edge = gaussian_smoothing_filter(sharp_edge_image)
median_edge = median_smoothing_filter(sharp_edge_image)
cnn_edge = cnn_filter(sharp_edge_image[0])



fig,ax = plt.subplots(figsize=(12,8))
ax.plot(sharp_edge_image[0][370:395,500],"r>",label="Original Image")
ax.plot(median_edge[0][370:395,500],"k*",label="Median Image")
ax.plot(gauss_edge[0][370:395,500],"g.",label="Gauss Image")
ax.plot(cnn_edge[370:395,500],"b<",label="CNN Image")
ax.legend()

plt.show()

# fig,axes = plt.subplots(ncols=2,nrows=2,figsize=(12,8))
# ax = axes.ravel()

# ax[0].imshow(sharp_edge_image[0],cmap="gray")
# ax[0].set_title("Original Image")
# ax[1].imshow(cnn_edge,cmap="gray")
# ax[1].set_title("CNN Filter Image")
# ax[2].imshow(median_edge[0],cmap="gray")
# ax[2].set_title("Median Filter Image")
# ax[3].imshow(gauss_edge[0],cmap="gray")
# ax[3].set_title("Gaussian Image")

# plt.show()


####################
### Add Random Noise
#####################

# noise_100 = add_random_noise(lo_images_roi,noise_level=0.01)
# patch_nps = power_spectral_analysis(noise_100)
# radial_profile_100 = nps_radial_profile(patch_nps)

# noise_500 = add_random_noise(lo_images_roi,noise_level=5)
# patch_nps = power_spectral_analysis(noise_500)
# radial_profile_500 = nps_radial_profile(patch_nps)

# noise_1000 = add_random_noise(lo_images_roi,noise_level=15)
# patch_nps = power_spectral_analysis(noise_1000)
# radial_profile_1000 = nps_radial_profile(patch_nps)

# noise_5000 = add_random_noise(lo_images_roi,noise_level=30)
# patch_nps = power_spectral_analysis(noise_5000)
# radial_profile_5000 = nps_radial_profile(patch_nps)

# noise_10000 = add_random_noise(lo_images_roi,noise_level=40)
# patch_nps = power_spectral_analysis(noise_10000)
# radial_profile_10000 = nps_radial_profile(patch_nps)

# fig,axes = plt.subplots(ncols=5,nrows=1,figsize=(12,8))
# ax = axes.ravel()

# ax[0].imshow(noise_100[0],cmap="gray")
# ax[0].set_title("S&P Noise 0.01")
# ax[1].imshow(noise_500[0],cmap="gray")
# ax[1].set_title("S&P Noise 5")
# ax[2].imshow(noise_1000[0],cmap="gray")
# ax[2].set_title("S&P Noise 15")
# ax[3].imshow(noise_5000[0],cmap="gray")
# ax[3].set_title("S&P Noise 30")
# ax[4].imshow(noise_10000[0],cmap="gray")
# ax[4].set_title("S&P Noise 40")

# fig,axes = plt.subplots(figsize=(12,8))
# axes.plot(radial_profile_100,"b*",label="Noise 0.01")
# axes.plot(radial_profile_500,"g^",label="Noise 5")
# axes.plot(radial_profile_1000,"ro",label="Noise 15")
# axes.plot(radial_profile_5000,"k.",label="Noise 30")
# axes.plot(radial_profile_10000,"c>",label="Noise 40")
# axes.legend(loc="best")
# axes.set_yscale("log", nonposy='clip')
# axes.set_xlim(left=0)
# axes.set_xlabel("Spatial Frequency (cycles/mm)")
# axes.set_ylabel("NPS (mm^2) ")

# plt.tight_layout()
#plt.show()