import numpy as np
import dicom
from PIL import ImageTk as pil_imagetk
from PIL import Image as pil_image

from tkinter import Tk
from tkinter.filedialog import askdirectory, askopenfilename, askopenfilenames
from skimage.filters import threshold_otsu, threshold_local
import cv2

from tkinter import *

#Load low energy and high energy images
high_img = dicom.read_file("/home/maksat/Desktop/DualEnergyProject/120_kVp/DS-1-120.dcm").pixel_array
high_img = cv2.bitwise_not(high_img)

low_img = dicom.read_file("/home/maksat/Desktop/DualEnergyProject/60_kVp/DS-1-60.dcm").pixel_array
low_img = cv2.bitwise_not(low_img)


class ACNR_GUI(Tk):
    def __init__(self,root, low_files, high_files):
        self.window = root
        self.high_img = high_files
        self.low_img = low_files
        self.w_T = 0.5
        self.w_B = 0.5
        self.img_width = 3*low_files.shape[0]//4
        self.img_height = 3*low_files.shape[1]//4

        self.__init_panel()

    def __init_panel(self):
        self.frame = Frame(self.window, bd=2, relief=SUNKEN)
        
        #get bone and soft tissue images at a time
        soft_tmp = -np.log(self.high_img)+self.w_T*np.log(self.low_img)
        bone_tmp = np.log(self.high_img)-self.w_B*np.log(self.low_img)

        self.soft = (2**8-1)*(soft_tmp-soft_tmp.min())/(soft_tmp.max()-soft_tmp.min())
        self.bone = (2**8-1)*(bone_tmp-bone_tmp.min())/(bone_tmp.max()-bone_tmp.min())

        self.soft_img_obj = pil_imagetk.PhotoImage(
            pil_image.fromarray(self.soft).resize((self.img_height,self.img_width),pil_image.ANTIALIAS))
        self.bone_img_obj = pil_imagetk.PhotoImage(
            pil_image.fromarray(self.bone).resize((self.img_height,self.img_width),pil_image.ANTIALIAS))


        self.panel1 = Label(self.frame,image=self.soft_img_obj)
        self.panel1.image = self.soft_img_obj ##you have to keep this other image will not show
        self.panel1.grid(row=0,column=0,sticky=N)

        self.panel2 = Label(self.frame,image=self.bone_img_obj)
        self.panel2.image = self.bone_img_obj ##you have to keep this otherwise image will not show
        self.panel2.grid(row=0,column=1,sticky=N)


        self.frame.pack()

    def update_tissue_weight(self,val):
        self.w_T = float(val)
        self.print_weights()
        self.update_soft_image()
        
    def update_bone_weight(self,val):
        self.w_B = float(val)
        self.print_weights()
        self.update_bone_image()
    def print_weights(self):
        print("Tissue weight: ",self.w_T,"Bone weight: ",self.w_B)

    def update_bone_image(self):
        bone_tmp = np.log(self.high_img)-self.w_B*np.log(self.low_img)

        self.bone = (2**8-1)*(bone_tmp-bone_tmp.min())/(bone_tmp.max()-bone_tmp.min())
        self.bone_img_obj = pil_imagetk.PhotoImage(
            pil_image.fromarray(self.bone).resize((self.img_height,self.img_width),pil_image.ANTIALIAS))
        self.panel2.configure(image=self.bone_img_obj)
        self.panel2.image = self.bone_img_obj

    def update_soft_image(self):
        soft_tmp = -np.log(self.high_img)+self.w_T*np.log(self.low_img)

        self.soft = (2**8-1)*(soft_tmp-soft_tmp.min())/(soft_tmp.max()-soft_tmp.min())
        self.soft_img_obj = pil_imagetk.PhotoImage(
            pil_image.fromarray(self.soft).resize((self.img_height,self.img_width),pil_image.ANTIALIAS))
        self.panel1.configure(image=self.soft_img_obj)
        self.panel1.image = self.soft_img_obj


root = Tk()
root.title("Dual Images")
root.geometry("1600x768")
root.configure(background='grey')
root.grid()
#initilize class to store widget positions
slider = ACNR_GUI(root,low_img,high_img)
w1 = Scale(slider.frame, from_=0.1, to=1, resolution=0.01, label="Soft tissue weight",
    orient = HORIZONTAL, tickinterval=1, length = 500, command=slider.update_tissue_weight)
w1.grid(row=1,column=0)
w2 = Scale(slider.frame, from_=-1, to=1.5, resolution=0.01, label="Bone weight",
    orient = HORIZONTAL, tickinterval=1, length = 500, command=slider.update_bone_weight)
w2.grid(row=1,column=1)

root.mainloop()
