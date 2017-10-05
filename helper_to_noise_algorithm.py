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

#from the paper FWHM = 4.71 ---> sigma = 4.71/sqrt(8*ln(2))
def gaussian_smoothing_filter(de_images_roi,window=5,sigma=2):
    M = de_images_roi.shape[0]
    de_images_gauss = de_images_roi.copy()

    if len(de_images_roi.shape)==2:
        de_images_gauss = cv2.GaussianBlur(de_images_roi,(window,window),sigma)
    else:
        for i in range(M):
            de_images_gauss[i,:,:] = cv2.GaussianBlur(de_images_roi[i,:,:],(window,window),sigma)

    return de_images_gauss

def median_smoothing_filter(de_images_roi,window=5):
    M = de_images_roi.shape[0]
    de_images_median = de_images_roi.copy()

    for i in range(M):
        de_images_median[i,:,:] = cv2.medianBlur(de_images_roi[i,:,:],window)

    return de_images_median


def correlated_noise_reduction(lo_images_roi,hi_images_roi):
    M = lo_images_roi.shape[0]
    w_T, w_B = 0.44, 0.72
    # p_T, p_B = 3.5, 3.5 # for lung, 2.0 for abdomenon and soft tissue
    p_T, p_B = 2.0, 2.0 # 2.0 for abdomenon and soft tissue

    soft_images_kcnr, bone_images_kcnr = [],[]
    for i in range(M):

        #get soft and bone images using weights
        soft_tmp = -np.log(hi_images_roi[i])+w_T*np.log(lo_images_roi[i])
        bone_tmp = np.log(hi_images_roi[i])-w_B*np.log(lo_images_roi[i])

        #apply high-pass-filter
        soft_hps = gaussian_smoothing_filter(soft_tmp)
        bone_hps = gaussian_smoothing_filter(bone_tmp)

        bone_cnr = bone_tmp+soft_hps/p_B
        soft_cnr = soft_tmp+bone_hps/p_T

        soft = (2**16-1)*soft_cnr/(soft_cnr.max()-soft_cnr.min())
        bone = (2**16-1)*bone_cnr/(bone_cnr.max()-bone_cnr.min())

        soft_images_kcnr.append(soft)
        bone_images_kcnr.append(bone)

    # return np.array(soft_images_kcnr), np.array(bone_images_kcnr)
    return np.array(soft_images_kcnr)

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
            lo_contrast[lo_contrast==0] = 1
            #hi_contrast[hi_contrast==0] =1

            #now define ratio
            ratio = hi_contrast/lo_contrast

            clip_hi_contrast = hi_contrast.copy()
            #now clip slopes
            clip_hi_contrast[ratio>up_slope] = up_slope*lo_contrast[ratio>up_slope]
            clip_hi_contrast[ratio<down_slope] = down_slope*lo_contrast[ratio<down_slope]

            #now convert back images to normal
            clip_hi_tmp = (clip_hi_contrast.copy()+hi_bkg)
            clip_hi_norm = (2**16-1)*(clip_hi_tmp)/(clip_hi_tmp.max()-clip_hi_tmp.min())
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
        # plt.plot(lo_contrast.ravel(),clip_hi_contrast.ravel(),"b.")
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

import keras as K
from keras.backend import tf as ktf
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D, UpSampling2D, Conv2D
from keras.layers import Lambda, Input
from keras.models import Model


def auto_encoder_model(de_image_roi,down_kernel=3,up_kernel=3):
    H,W = de_image_roi.shape
    input_img = Input(shape=(H,W,1))

    x = Conv2D(64, (down_kernel, down_kernel), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    print("Pool1 shape ",x.shape)

    # x = Conv2D(64, (down_kernel, down_kernel), activation='relu', padding='same')(x)
    # x = MaxPooling2D((2, 2), padding='same')(x)
    # print("Pool2 shape ",x.shape)

    # x = Conv2D(32, (up_kernel, up_kernel), activation='relu', padding='same')(x)
    # x = MaxPooling2D((2, 2), padding='same')(x)
    # print("Pool3 shape ",x.shape)

    x = Conv2D(1, (up_kernel,up_kernel), activation='relu', padding='same')(x)
    decoded = UpSampling2D((2,2))(x)
    print("Pool4 shape ",decoded.shape)

    decoded = Lambda(lambda image: ktf.image.resize_images(image, (H,W)))(x)

    # sgd = K.optimizers.SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
    sgd = K.optimizers.RMSprop(lr=0.002, rho=0.9, epsilon=1e-08, decay=0.95)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=sgd, loss='mean_absolute_error')

    X = de_image_roi.copy()/(2**16-1)
    # X = np.log(img.copy()+1)

    hist = autoencoder.fit(X[np.newaxis,:,:,np.newaxis], X[np.newaxis,:,:,np.newaxis],
                batch_size=10, epochs=100, verbose=1)
    X_filtered = autoencoder.predict(X[np.newaxis,:,:,np.newaxis])*(2**16-1)

    return hist,X_filtered[0,:,:,0]


def cnn_filter(de_images_roi):
    N = de_images_roi.shape[0]
    de_images_noise = de_images_roi.copy()

    if len(de_images_roi.shape)==2:
        _, filtered = auto_encoder_model(de_images_roi)
        de_images_noise = filtered

    else:
        for i in range(N):
            print("Running image ",i)
            _, filtered = auto_encoder_model(de_images_roi[i,:,:])
            de_images_noise[i,:,:] = filtered

    return de_images_noise

def add_random_noise(de_images_roi,noise_level):
    M,N,K = de_images_roi.shape
    de_images_noise = de_images_roi.copy()

    noise = np.zeros((N,K))

    for i in range(M):
        # de_images_noise[i,:,:] = cv2.randu(noise,0,noise_level)
        # cv2.randn(noise,0,noise_level)
        cv2.randn(noise,0,noise_level)
        de_images_noise[i,:,:] += noise.astype(np.uint16)

    return de_images_noise

def create_sharp_edge_image(N,K):
    sharp_image = np.zeros((N,K))

    sharp_image[N//2:, :] = (2**16-1)

    return np.array([sharp_image,sharp_image],np.uint16)

