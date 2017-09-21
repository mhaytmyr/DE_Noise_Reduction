import numpy as np
import cv2
from scipy import signal
from helper_to_read_files import apply_log_subtraction


'''
This file contains noise reduction filters.
1. Simple smoothing filter.
    Composed of gaussian filter with FWHM = 4.71 pixel. ROI = 19 pixel
2. Median filter
    Composed of median filter on 5x5 area

3. Noise clipping method based on the
'''

def gaussian_smoothing_filter(de_images_roi,window=5,sigma=1):
    M = de_images_roi.shape[0]
    de_images_gauss = de_images_roi.copy()

    for i in range(M):
        de_images_gauss[i,:,:] = cv2.GaussianBlur(de_images_roi[i,:,:],(window,window),sigma)

    return de_images_gauss

def median_smoothing_filter(de_images_roi,window=5):
    M = de_images_roi.shape[0]
    de_images_median = de_images_roi.copy()

    for i in range(M):
        de_images_median[i,:,:] = cv2.medianBlur(de_images_roi[i,:,:],window)

    return de_images_median

import time

def noise_clipping_filter(lo_images_roi,hi_images_roi):
    M,N,K = de_images_roi.shape
    de_images_clip = lo_images_roi.copy()

    for i in range(M):
        #first compute low and high background
        lo_bkg = signal.medfilt2d(lo_images_roi[i,:,:].astype(np.float32))
        hi_bkg = signal.medfilt2d(hi_images_roi[i,:,:].astype(np.float32))

        #then subtract each of images from background
        lo_contrast = (lo_images_roi[i,:,:]-lo_bkg)
        hi_contrast = (hi_images_roi[i,:,:]-hi_bkg)

        #loot at the ratio between hi_contrast and lo_contrast
        #convert hi_pixel values min(h_pixel,0.72) and max(h_pixel,0.44)
        plt.plot(lo_contrast.ravel(), hi_contrast.ravel(),"r*")
        plt.xlabel("Low contrast")
        plt.ylabel("Low contrast")
        plt.show(0)
        time.sleep(3)


def add_random_noise(de_images_roi):
    M,N,K = de_images_roi.shape
    de_images_noise = de_images_roi.copy()

    noise = np.zeros((N,K))

    for i in range(M):
        de_images_noise[i,:,:] = cv2.randu(noise,0,10000)

    return de_images_noise

