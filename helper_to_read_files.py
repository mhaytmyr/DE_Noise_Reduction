import dicom, os, cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askdirectory, askopenfilename, askopenfilenames

'''
This function will return all images as one ndarray and apply region of interest.

'''


def apply_log_subtraction(low,high,weight=0.5):
    de_img = np.exp(-(np.log(high)-weight*np.log(low)))
    de_img_norm = (2**16-1)*(de_img-de_img.min())/(de_img.max()-de_img.min())

    return de_img_norm

def read_files():
    Tk().withdraw()
    low_dir = askdirectory(title="Low kVp directory")
    print(low_dir)
    high_dir = askdirectory(title="High kVp directory")
    print(high_dir)

    low_files = sorted(os.listdir(low_dir))
    high_files = sorted(os.listdir(high_dir))

    combined_images = []

    #make sure number of high energy and low energy files are equal
    assert low_files.__len__()==high_files.__len__()

    for i in range(low_files.__len__()):
        #read files and revert them
        low_img = dicom.read_file(low_dir+"/"+low_files[i]).pixel_array
        low_img = cv2.bitwise_not(low_img)

        high_img = dicom.read_file(high_dir+"/"+high_files[i]).pixel_array
        high_img = cv2.bitwise_not(high_img)
        print(low_files[i],high_files[i])

        de_img = apply_log_subtraction(low_img,high_img)
        combined_images.append(de_img)

    return np.array(combined_images)

