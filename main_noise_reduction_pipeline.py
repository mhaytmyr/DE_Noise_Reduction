#import helper functions
from helper_to_read_files import *
from helper_to_select_roi import *
from helper_to_get_metrics import *
from helper_to_noise_algorithm import *


#first read files
lo_images_combined, hi_images_combined= read_files()
print(lo_images_combined.shape, hi_images_combined.shape)

lo_images_roi, hi_images_roi = choose_region_of_interest(lo_images_combined,hi_images_combined)
print(lo_images_roi.shape, hi_images_roi.shape)


###Add random noise
# de_images_roi = add_random_noise(de_images_roi)

### Run NPS on original image
de_images_roi = apply_log_subtraction(lo_images_roi,hi_images_roi)
patch_nps = power_spectral_analysis(de_images_roi)
img_radial_profile = nps_radial_profile(patch_nps)
original_de_images = apply_log_subtraction(lo_images_combined[0],hi_images_combined[0])


### Run auto-encoder
de_images = apply_log_subtraction(lo_images_roi,hi_images_roi)
de_images_noise = cnn_filter(de_images)
patch_nps = power_spectral_analysis(de_images_noise)
cnn_radial_profile = nps_radial_profile(patch_nps)
# cnn_de_images = cnn_filter(apply_log_subtraction(lo_images_combined[0:1],hi_images_combined[0:1]))

#### Run gaussin filter and get NPs profile
## Noise reduction is applied to High image only
hi_images_noise = gaussian_smoothing_filter(hi_images_roi)
de_images_roi = apply_log_subtraction(lo_images_roi,hi_images_noise)
patch_nps = power_spectral_analysis(de_images_roi)
gauss_radial_profile = nps_radial_profile(patch_nps)
gauss_de_images = apply_log_subtraction(lo_images_combined[0:1],gaussian_smoothing_filter(hi_images_combined[0:1]))


# #### Run median filter and get NPs profile
hi_images_noise = median_smoothing_filter(hi_images_roi)
de_images_roi = apply_log_subtraction(lo_images_roi,hi_images_noise)
patch_nps = power_spectral_analysis(de_images_roi)
median_radial_profile = nps_radial_profile(patch_nps)
median_de_images = apply_log_subtraction(lo_images_combined[0:1],median_smoothing_filter(hi_images_combined[0:1]))

### Run noise clipping algorithm
hi_images_noise = noise_clipping_filter(lo_images_roi,hi_images_roi)
de_images_roi = apply_log_subtraction(lo_images_roi,hi_images_noise)
patch_nps = power_spectral_analysis(de_images_roi)
noc_radial_profile = nps_radial_profile(patch_nps)
noc_de_images = apply_log_subtraction(lo_images_combined[0:1],
    noise_clipping_filter(lo_images_combined[0:1],hi_images_combined[0:1]))

# #de_roi_std = variance_analysis(de_images_roi)

fig,axes = plt.subplots(ncols=2,figsize=(12,8))
ax = axes.ravel()

ax[0].imshow(de_images_roi[0],cmap="gray")
ax[1].plot(img_radial_profile,"b*",label="Original Image")
ax[1].plot(gauss_radial_profile,"g^",label="Gaussian Image")
ax[1].plot(median_radial_profile,"ro",label="Median Image")
ax[1].plot(noc_radial_profile,"k.",label="NOC Image")
ax[1].plot(cnn_radial_profile,"c>",label="CNN Image")
ax[1].legend(loc="best")
ax[1].set_yscale("log", nonposy='clip')
ax[1].set_xlim(left=5)

plt.show()

fig,axes = plt.subplots(ncols=2,nrows=2,figsize=(15,12))
ax = axes.ravel()

ax[0].imshow(original_de_images[0],cmap="gray")
ax[0].set_title("Original Image")
# ax[1].imshow(cnn_de_images[0,:,:],cmap="gray")
# ax[1].set_title("CNN Filter Image")
ax[2].imshow(median_de_images[0],cmap="gray")
ax[2].set_title("Median Filter Image")
ax[3].imshow(noc_de_images[0],cmap="gray")
ax[3].set_title("Noise Clip Image")

plt.show()


# for i in range(de_images_roi.shape[0]):
#     de_img = de_images_roi[i,:,:]

#     plt.figure(figsize=(12,8))
#     plt.imshow(de_img,cmap="gray")
#     plt.show(0)
#     time.sleep(2)
