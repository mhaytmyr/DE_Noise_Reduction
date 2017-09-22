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
import matplotlib.pyplot as plt

def noise_clipping_filter(lo_images_roi,hi_images_roi,apply_log=True):
    M = lo_images_roi.shape[0]
    de_images_clip = lo_images_roi.copy()

    #slopes from the paper
    up_slope = 0.44 #0.72
    down_slope = 0.72 #0.44

    #only return high images, as only the yahve been modified
    high_images = []
    for i in range(M):

        if apply_log:
            lo_bkg = signal.medfilt2d(np.log(lo_images_roi[i,:,:]).astype(np.float32))
            hi_bkg = signal.medfilt2d(np.log(hi_images_roi[i,:,:]).astype(np.float32))

            #then subtract each of images from background
            lo_contrast = (np.log(lo_images_roi[i,:,:])-lo_bkg)
            hi_contrast = (np.log(hi_images_roi[i,:,:])-hi_bkg)

            #convert zeros to one
            lo_contrast[lo_contrast==0] =1
            #hi_contrast[hi_contrast==0] =1

            #now define ratio
            ratio = hi_contrast/lo_contrast

            clip_hi_contrast = hi_contrast.copy()
            #now clip slopes
            clip_hi_contrast[ratio>up_slope] = up_slope*lo_contrast[ratio>up_slope]
            clip_hi_contrast[ratio<down_slope] = down_slope*lo_contrast[ratio<down_slope]

            #now convert back images to normal
            clip_hi_tmp = (clip_hi_contrast.copy()+hi_bkg)
            clip_hi_norm = (2**16-1)*clip_hi_tmp.copy()/(clip_hi_tmp.max()-clip_hi_tmp.min())
            # clip_hi_contrast = np.exp(clip_hi_contrast)

            high_images.append(clip_hi_norm)

        else:
            #first compute low and high background
            lo_bkg = signal.medfilt2d(lo_images_roi[i,:,:].astype(np.float32))
            hi_bkg = signal.medfilt2d(hi_images_roi[i,:,:].astype(np.float32))

            #then subtract each of images from background
            lo_contrast = (lo_images_roi[i,:,:]-lo_bkg)
            hi_contrast = (hi_images_roi[i,:,:]-hi_bkg)

            #convert zeros to one
            lo_contrast[lo_contrast==0] =1
            hi_contrast[hi_contrast==0] =1

        #loot at the ratio between hi_contrast and lo_contrast
        #convert hi_pixel values min(h_pixel,0.72) and max(h_pixel,0.44)
        # plt.plot(lo_contrast.ravel(),clip_hi_constrast.ravel(),"b.")
        # plt.xlabel("Low contrast")
        # plt.ylabel("Low contrast")
        # plt.xlim(-0.1,0.1)
        # plt.ylim(-0.1,0.1)
        # plt.show()
        # fig,axes = plt.subplots(ncols=2)
        # ax = axes.ravel()
        # ax[0].imshow(hi_images_roi[-1],label="original",cmap="gray")
        # ax[1].imshow(high_images[-1],label="clipped",cmap="gray")
        # plt.show()

    return np.array(high_images)

def add_random_noise(de_images_roi):
    M,N,K = de_images_roi.shape
    de_images_noise = de_images_roi.copy()

    noise = np.zeros((N,K))

    for i in range(M):
        de_images_noise[i,:,:] = cv2.randu(noise,0,10000)

    return de_images_noise

