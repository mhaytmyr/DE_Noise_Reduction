import numpy as np
import matplotlib.pyplot as plt
import dicom

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


def power_spectral_analysis(patch_of_roi):
