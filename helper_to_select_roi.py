import numpy as np
import cv2
from skimage.filters import threshold_otsu, threshold_local


from tkinter import Tk
from tkinter.filedialog import askdirectory, askopenfilename, askopenfilenames

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
def choose_region_of_interest(combined_images):
    #choose single image for display
    img = combined_images[0,:,:]

    block_size = 65
    adaptive_thresh = threshold_local(img, block_size, offset=0)
    binary_adaptive = img*(img > adaptive_thresh)

    #now call cv2
    roi_region = cv2.selectROI("Select Soft Region",binary_adaptive)
    cv2.destroyAllWindows()

    reduced_images = []
    for i in range(combined_images.shape[0]):
        img_iter = combined_images[i,:]

        roi_img = img_iter[roi_region[1]:roi_region[1]+roi_region[3], roi_region[0]:roi_region[0]+roi_region[2]]
        reduced_images.append(roi_img)

    return np.array(reduced_images)

