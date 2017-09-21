import numpy as np
import cv2
from skimage.filters import threshold_otsu, threshold_local


from tkinter import Tk
from tkinter.filedialog import askdirectory, askopenfilename, askopenfilenames

from helper_to_read_files import apply_log_subtraction

'''
This is interactive helper function to choose region of interest.
'''

#Following function is only for visualization purposes
def sigmoid_normalization(img,min_new=0,max_new=2**16-1,power=-1):
    min_old, max_old = min(img.flat), max(img.flat)
    alpha = np.std(img.flat)
    beta = np.mean(img.flat)

    img_norm = (max_new-min_new)*(1+np.exp(-(img-beta)/alpha))**power+min_new
    return img_norm

#open files and select region of interest
def choose_region_of_interest(lo_images,hi_images):
    M = lo_images.shape[0]

    #choose single image for display
    img = apply_log_subtraction(lo_images[0],hi_images[0])

    block_size = 65
    adaptive_thresh = threshold_local(img, block_size, offset=0)
    binary_adaptive = img*(img > adaptive_thresh)

    #now call cv2
    roi_region = cv2.selectROI("Select Soft Region",binary_adaptive)
    cv2.destroyAllWindows()

    lo_images_roi, hi_images_roi = [],[]
    for i in range(M):
        img_iter = lo_images[i,:,:].copy()
        roi_img = img_iter[roi_region[1]:roi_region[1]+roi_region[3], roi_region[0]:roi_region[0]+roi_region[2]]
        lo_images_roi.append(roi_img)

        img_iter = hi_images[i,:,:].copy()
        roi_img = img_iter[roi_region[1]:roi_region[1]+roi_region[3], roi_region[0]:roi_region[0]+roi_region[2]]
        hi_images_roi.append(roi_img)

    lo_images_roi = np.array(lo_images_roi)
    hi_images_roi = np.array(hi_images_roi)

    return lo_images_roi, hi_images_roi

