import dicom, os, cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askdirectory, askopenfilename, askopenfilenames

'''
This function will return all images as one ndarray and apply region of interest.

'''

def apply_log_subtraction(low_files,high_files,weight=0.5):
    M = low_files.shape[0]
    de_img_norm = low_files.copy()

    #if there is only one file
    if len(low_files.shape)==2:
        de_img = np.exp(-(np.log(high_files)-weight*np.log(low_files)))
        de_img_norm = (2**16-1)*(de_img-de_img.min())/(de_img.max()-de_img.min())
    else:
        for i in range(M):
            de_img = np.exp(-(np.log(high_files[i,:,:])-weight*np.log(low_files[i,:,:])))
            de_img_norm[i,:,:] = (2**16-1)*(de_img-de_img.min())/(de_img.max()-de_img.min())

    return de_img_norm

def read_files():
    Tk().withdraw()
    low_dir = askdirectory(title="Low kVp directory")
    print(low_dir)
    high_dir = askdirectory(title="High kVp directory")
    print(high_dir)

    low_files = sorted(os.listdir(low_dir))
    high_files = sorted(os.listdir(high_dir))

    low_img_files,high_img_files = [],[]

    #make sure number of high energy and low energy files are equal
    assert low_files.__len__()==high_files.__len__()

    for i in range(low_files.__len__()):
        #read files and revert them
        low_img = dicom.read_file(low_dir+"/"+low_files[i]).pixel_array
        low_img = cv2.bitwise_not(low_img)

        high_img = dicom.read_file(high_dir+"/"+high_files[i]).pixel_array
        high_img = cv2.bitwise_not(high_img)
        print(low_files[i],high_files[i])

        # de_img = apply_log_subtraction(low_img,high_img)
        # combined_images.append(de_img)
        low_img_files.append(low_img)
        high_img_files.append(high_img)

    return np.array(low_img_files), np.array(high_img_files)

