# Read in: Alan's segmented cell images and their corresponding black masks.
# Output: White masked, centred, size normalized to 112*112 cell images.
#         * Ready to put into CNN (ResNet/VGG/etc.)
#         * Ready to train for FCN (ResNet + HeatMap)


# Load packages.
import cv2, os
import numpy as np
import glob,shutil
from PIL import Image
from keras import backend as K
K.set_image_dim_ordering('tf')
from numpy import max
import tkinter as tk
import tkinter.filedialog as tkFileDialog

######################################################## STEP 1 #######################################################
# Combine Cell Image with Mask, padding to a certain size (eg. 112*112)
# Set the initial directory.
# Use GUI to select the directory contains initial data from Alan and the directory to save the processed images.
root = tk.Tk()
root.directory = tkFileDialog.askdirectory()
readpath = root.directory
root.destroy()

root = tk.Tk()
root.directory = tkFileDialog.askdirectory()
savepath = root.directory
root.destroy()

# Example:
# readpath = '/data/edong/PycharmProjects/projpy/InitialIMG'
# savepath = "/data/edong/PycharmProjects/projpy/Colour_Mask_Padded_White_Samller_InterIMG"


# Return the maximum width and height of all images under the readpath (Cancer & Normal), also the number of images.
# This step is to get an overview of the biggest cell image of the current dataset. To help decide whehter 112 is appropriate.
def getImgDimension(readpath):
    listdir = ['Cancer', 'Normal']
    wlist = []
    hlist = []
    for dir in listdir:
        os.chdir(readpath + '\\' + dir)
        print('Loaded the images of dataset-' + '{}\n'.format(dir))
        listimgs = glob.glob('*.tif')
        for i in listimgs:
            #file, ext = os.path.splitext(i)
            img = Image.open(i)
            width, height = img.size
            wlist.append(width)
            hlist.append(height)

    maxw = max(wlist)
    maxh = max(hlist)
    numCells = len(wlist)

    return maxw, maxh, numCells

getImgDimension(readpath)

# TODO: Set up a tk GUI for storing an entered value. Currently, the default wanted image size after the first step processing is 112*112.

# Combine Cell Image with Mask, padding to 112 (No resize involved to preserve the cell images)
# Here under any directory, I have two subfolders to store images: "Cancer" and "Normal".
# Example: All the images from readpath "Cancer" will be processed and stored into savepath "Cancer".
def mask_resize_img(path1,dst_dir, pad_size):
    listdir = os.listdir(path1)
    for dataset in listdir:
        all_black_after_crop = []
        os.chdir(path1 + '\\' + dataset)
        print('Loaded the images of dataset-' + '{}\n'.format(dataset))
        lisoftiff = glob.glob('*.tif')
        everyother = lisoftiff[::2]
        for img in everyother:
            os.chdir(path1 + '\\' + dataset)
            img_name, ext = os.path.splitext(img)
            img_read = cv2.imread(img)
            mask_read = cv2.imread(img_name+'m.tif', 0)
            masked_img = cv2.bitwise_and(img_read, img_read, mask=mask_read)
            r, c, ch = img_read.shape
            imgsub = pad_img_white(r,c,masked_img, pad_size)
            os.chdir(dst_dir + '\\' + dataset)
            cv2.imwrite(img, imgsub)

mask_resize_img(readpath, savepath, pad_size= 112)

# Helper function to pad the surroundings of an image to a certain size (eg. 112*112), after masked.
# Cutoff Extra Background and Pad Insufficient Background, on both margins of a horizontal or a vertical direction.
def pad_img_white(r,c, masked_img, pad_size):
    imgsub = np.zeros([pad_size, pad_size, 3])
    if (r >= pad_size and c >= pad_size):
        imgsub = masked_img[int(r / 2) - int(pad_size / 2): int(r / 2) + int(pad_size / 2), int(c / 2) - int(pad_size / 2): int(c / 2) + int(pad_size / 2)]
    elif (r > pad_size and c <= pad_size):
        imgcrop = masked_img[int(r / 2) - int(pad_size / 2): int(r / 2) + int(pad_size / 2), :]
        imgsub[:, int(pad_size / 2 - c / 2): int(pad_size / 2 - c / 2) + c] = imgcrop
    elif (c > pad_size and r <= pad_size):
        imgcrop = masked_img[:, int(c / 2) - int(pad_size / 2): int(c / 2) + int(pad_size / 2)]
        imgsub[int(pad_size / 2 - r / 2): int(pad_size / 2 - r / 2) + r, :] = imgcrop
    elif (r < pad_size and c < pad_size):
        imgsub[int(pad_size / 2 - r / 2): int(pad_size / 2 - r / 2) + r, int(pad_size / 2 - c / 2): int(pad_size / 2 - c / 2) + c] = masked_img

    img_mean = np.mean(imgsub, axis=2)
    for m in range(112):
        for l in range(112):
            if (img_mean[m, l] == 0):
                imgsub[m, l, :] = 255

    return imgsub



############################################################## Step 2 #################################################
# After the big cropping, some cells would be cut into half or appear all blank if they are not completely centred,
# and we will do more processing for those cells.


######## Step One: Re-crop the half cells, to make them centre.
# Deal with peripheral cells Manually
# Read in each of the half cells, and crop with the below 8 img_cen options. Currently, manually.
# TODO: Generate an algorithm that can identify the names of the half-cells. For each of the name on list, read-in image,
# TODO: perform a location detection algorithm (probably Fast-RCNN), locate the half cell pixel, and re-crop with one
# TODO: of the 8 img_cen options to make the new crop 112*112 centred.

img = cv2.imread('/home/edong/Desktop/Peripheral Cells/Normal/UD02-3490_Thionin_Normal_FEU_00040_1_40x.tif')
# img_cen = img[56:168, 0:112] # left middle
#img_cen = img[0:112, 56:168] # up middle
#img_cen = img[112:224, 56:168] # down middle
#img_cen = img[56:168, 112:224] # right middle

# img_cen = img[0:112, 0:112] # up left corner
# img_cen = img[112:224, 0:112] # down left corner
# img_cen = img[112:224, 112:224] # down right corner
img_cen = img[0:112, 112:224] # up right corner

# Normal Half Cells Name List
# E01213032006E_2_FEU_00054_1_40x
# C037140324A03_2_FEU_00004_1_40x
# C037140324A03_2_FEU_00001_1_40x
# C037140324A03_2_FEU_00019_1_40x


# Cancer Half Cells Name List
# 01-2309-_Thionin_Cancer_FEU_00000_1_40x
# 97-37535B_Thionin_Cancer_FEU_00155_1_40x
# 98-32735C_Thionin_Cancer_FEU_00083_1_40x

# Returns the name of the all-blank images after cropping, so that we can re-crop them.
def find_all_white(savepath):
    all_white_list=[]
    listdir = os.listdir(savepath)
    for dataset in listdir:
        os.chdir(savepath + '/' + dataset)
        print('Loaded the images of dataset-' + '{}\n'.format(dataset))
        lisoftiff = glob.glob('*.tif')
        for img in lisoftiff:
            img_read = cv2.imread(img)
            f = np.asarray(img_read)
            m = np.sum(f)
            if (m == 9596160):  # 224*224*255
                all_white_list.append(img)
    return all_white_list

find_all_white(savepath)

# Now re-crop also with the above 8 img_cen options. Manually.
################################################### Experiment #########################################################
# # Display an image
# import cv2, os
# img = cv2.imread('/data/edong/PycharmProjects/projpy/InitialIMG/Cancer/00-3734A_Thionin_Cancer_FEU_00000_1_40x.tif')
# mask = cv2.imread('/data/edong/PycharmProjects/projpy/InitialIMG/Cancer/00-3734A_Thionin_Cancer_FEU_00000_1_40xm.tif', 0)
# res = cv2.bitwise_and(img, img, mask=mask)
# os.chdir('C:\\Users\\edong\\Desktop')
# cv2.imwrite('masked.tiff', res)
# cv2.imshow('masked_img', res)
# cv2.waitKey(0)
# cv2.imshow('img', img)
# # cv2.imshow('mask', mask)
# cv2.waitKey(0)
#
# ### Extra Large images replacement #####
# # Padd any images to 224*224 for ResNet
# img = cv2.imread('K:\\edong\\PycharmProjects\\projpy\\InitialIMG\\Normal\\01UD-787_Thionin_Normal_FEU_00000_1_40x.tif')
# mask = cv2.imread('K:\\edong\\PycharmProjects\\projpy\\InitialIMG\\Normal\\01UD-787_Thionin_Normal_FEU_00000_1_40xm.tif', 0)
# res = cv2.bitwise_and(img, img, mask=mask)
# r,c,ch=img.shape
# imgsub = np.zeros([224,224,3])
# if(r>224 and c>224):
#     imgsub = res[int(r/2) - 112 : int(r/2) + 112, int(c/2) - 112 : int(c/2) + 112]
# if(r > 224 and c<224):
#     imgcrop = res[int(r/2) - 112 : int(r/2) + 112, :]
#     imgsub[:, int(224 / 2 - c / 2): int(224 / 2 - c / 2) + c] = imgcrop
# if(c > 224 and r<224):
#     imgcrop = res[:, int(c / 2) - 112: int(c / 2) + 112]
#     imgsub[int(224 / 2 - r / 2): int(224 / 2 - r / 2) + r, :] = imgcrop
# if(r<224 and c<224):
#     imgsub[int(224 / 2 - r / 2): int(224 / 2 - r / 2) + r, int(224 / 2 - c / 2): int(224 / 2 - c / 2) + c] = res
#
# imgsub = pad_img(r,c,res,224)
#
# cv2.imshow('croped', imgsub)
# cv2.waitKey(0)
# os.chdir('C:\\Users\\edong\\Desktop')
# cv2.imwrite('UD02-3490_Thionin_Normal_FEU_00026_1_40x.tiff', imgsub)
# cv2.imwrite('UD02-3490_Thionin_Normal_FEU_00026_1_40xm.tiff', mask)
#
# ###Try to find the centre implementation #
# img = cv2.imread('/data/edong/PycharmProjects/projpy/Colour_Mask_Padded_InterIMG/Cancer/00-13464A_Thionin_Cancer_FEU_00004_1_40x.tif')
# img_cen = img[112-56:112+56, 112-56:112+56]
# img_mean = np.mean(img_cen, axis=2)
#
# # Invert Black Mask to White Mask.
# for m in range(112):
#     for l in range(112):
#         if (img_mean[m,l] == 0):
#             img_cen[m,l,:] = 255
#
# cv2.imshow('img_cen', img_cen)
# cv2.waitKey(0)
#
# os.chdir('/home/edong/Desktop')
# cv2.imwrite('img.tiff', img_cen)
#
# # Experiment with all white
# img = cv2.imread('/data/edong/PycharmProjects/projpy/Colour_Mask_Padded_White_Samller_InterIMG/Normal/C040140922A04_1_FEU_00045_1_40x.tif')
# img = np.asarray(img)
#
#
# img_mean = np.mean(img_cen, axis=2)
# for m in range(112):
#     for l in range(112):
#         if (img_mean[m,l] == 0):
#             img_cen[m,l,:] = 255
# cv2.imshow('img_cen', img_cen)
# cv2.waitKey(0)
#
# os.chdir('/home/edong/Desktop/Peripheral Cells/Normalok')
# cv2.imwrite('UD02-3490_Thionin_Normal_FEU_00040_1_40x.tif', img_cen)
#
#
# ############################## Useless since I no longer train on these separate folders
# # Save as PNG format for Image Data Generator
# train_data_dir = '/data/edong/PycharmProjects/projpy/Colour_Mask_Padded_White_Samller_InterIMG_Train_Val_Splited/Train'
# validation_data_dir = '/data/edong/PycharmProjects/projpy/Colour_Mask_Padded_White_Samller_InterIMG_Train_Val_Splited/Val'
#
# def save_tif_as_png(readpath):
#     listdir = os.listdir(readpath)
#     for dataset in listdir:
#         if dataset == 'Cancer':
#             datasave = 'CancerPNG'
#         else: datasave = 'NormalPNG'
#
#         os.chdir(readpath + '/' + dataset)
#         print('Loaded the images of dataset-' + '{}\n'.format(dataset))
#         lisoftiff = glob.glob('*.tif')
#         for img in lisoftiff:
#             os.chdir(readpath + '/' + dataset)
#             img_name, ext = os.path.splitext(img)
#             img_read = cv2.imread(img)
#             os.chdir(readpath + '/' + datasave)
#             cv2.imwrite(img_name + '.png', img_read)
#
# save_tif_as_png(train_data_dir)
# save_tif_as_png(validation_data_dir)


###################################### Helper Functions (might be useful in the future) ###############################
# Copy a list of .tif files to the destination direction.

# def CopyForOverWrite(list,dst_dir):
#     for tiffile in list:
#         shutil.copy(tiffile, dst_dir)