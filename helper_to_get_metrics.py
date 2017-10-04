import numpy as np
import matplotlib.pyplot as plt
import dicom

from scipy import signal

'''
This function contains two metrics to access performance of noise reduction techniques.
Based on the Warp and Dobbins (2003). There are metrics

1. Variance analysis
    The residual noise in each ROI is measured as the standard deviation of the residual noise distribution.
    The residual noise distribution is estimated by subtracting a single image from the ensemble average of N=66 images.
2. Power spectral analysis
    The regional two-dimensional noise power spectrum (2D-NPS). The regional NPS was used to evaluate noise
    characteristics in the lung, mediastinum, and subdiaphragm. Regions of interest (ROI)
    containing 384 2 pixels (7.68 cm^2) were chosen to compute the 2D-NPS using the periodogram technique of Welch.

'''

def variance_analysis(patch_of_roi):

    mean_intensity_all_images = patch_of_roi.mean()
    residual_noise = patch_of_roi - mean_intensity_all_images
    standard_dev = residual_noise.std()

    return standard_dev

#Image Plane Pixel Spacing
Ux= 0.388*(100/150); # x-pixel in mm
Uy = 0.388*(100/150);# y-pixel in mm
#can also be accessed. dicom.read_fil("file.dcm").ImagePlanePixelSpacing

def power_spectral_analysis(patch_of_roi):
    #lineraize image
    #following is important to remove errors ins log
    patch_roi_norm = np.log(patch_of_roi.copy().clip(patch_of_roi<=0))

    N,row,col = patch_of_roi.shape
    Size = row*col

    patch_nps = np.zeros((row,col))
    #initially remove mean from each selected roi
    for i in range(N):

        patch_roi_norm[i,:,:] = patch_of_roi[i,:,:]-patch_of_roi[i,:,:].mean()

        #now perform fourier transform for each patch
        f = np.fft.fft2(patch_roi_norm[i,:,:])
        fshift = np.fft.fftshift(f)

        #from the paper
        patch_nps += abs(fshift**2)*(Ux*Uy/Size)
        # patch_nps += 20*np.log(np.abs(fshift)*(Ux*Uy/Size))

        #here is how you do inverse transform
        # rows, cols = img.shape
        # crow,ccol = rows/2 , cols/2

        # # create a mask first, center square is 1, remaining all zeros
        # mask = np.zeros((rows,cols,2),np.uint8)
        # mask[crow-30:crow+30, ccol-30:ccol+30] = 1

        # # apply mask and inverse DFT
        # fshift = dft_shift*mask
        # f_ishift = np.fft.ifftshift(fshift)
        # img_back = cv2.idft(f_ishift)

    #finally divide by the number of patches
    patch_nps /= N

    #normalize by maximim
    # patch_nps /= 10e-10

    return patch_nps

def nps_radial_profile(patch_nps, center=None):

    y, x = np.indices((patch_nps.shape))
    y = y*Uy
    x = x*Ux

    if not center:
        center = np.array([np.floor((x.max()-x.min())/2.0), np.floor((y.max()-y.min())/2.0)])

    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), patch_nps.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / nr

    return radial_profile