#import helper functions
from helper_to_read_files import *
from helper_to_select_roi import *
from helper_to_get_metrics import *


#first read files
de_images_combined = read_files()
print(de_images_combined.shape)
de_images_roi = choose_region_of_interest(de_images_combined)
print(de_images_roi.shape)

#### TODO: Call noise reduction algorithm
## noise_reduction_algo1, noise_reduction_algo2 etc.

#de_roi_std = variance_analysis(de_images_roi)

patch_nps = power_spectral_analysis(de_images_roi)

radial_profile = nps_radial_profile(patch_nps)

fig,axes = plt.subplots(ncols=2,figsize=(12,8))
ax = axes.ravel()

ax[0].imshow(patch_nps,cmap="gray")
ax[1].plot(radial_profile,"b*")
# ax[1].set_yscale("log", nonposy='clip')
plt.show()


# for i in range(de_images_roi.shape[0]):
#     de_img = de_images_roi[i,:,:]

#     plt.figure(figsize=(12,8))
#     plt.imshow(de_img,cmap="gray")
#     plt.show(0)
#     time.sleep(2)
