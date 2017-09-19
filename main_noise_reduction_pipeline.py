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

de_roi_std = variance_analysis(de_images_roi)

# for i in range(de_images_roi.shape[0]):
#     de_img = de_images_roi[i,:,:]

#     plt.figure(figsize=(12,8))
#     plt.imshow(de_img,cmap="gray")
#     plt.show(0)
#     time.sleep(2)
