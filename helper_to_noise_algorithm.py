import numpy as np
import cv2

'''
This file contains noise reduction filters.
1. Simple smoothing filter.
    Composed of gaussian filter with FWHM = 4.71 pixel. ROI = 19 pixel
2. Median filter
    Composed of median filter on 5x5 area
'''