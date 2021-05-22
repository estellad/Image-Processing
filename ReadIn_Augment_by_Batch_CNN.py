from __future__ import division
import os, cv2
import numpy as np
import glob,shutil
from keras.utils import np_utils
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau
K.set_image_dim_ordering('tf')

# %%
pathcantrain = '/data/edong/PycharmProjects/projpy/Colour_Mask_Padded_White_Samller_InterIMG/Cancer'
pathnortrain = '/data/edong/PycharmProjects/projpy/Colour_Mask_Padded_White_Samller_InterIMG/Normal'


# Step 4
# No augmentation, just read in.
def get_data (path1, path2):
    X = []
    numlist = []
    namelist = []
    os.chdir(path1)
    print('Loaded the images of the folder-' + '{}\n'.format('Cancer'))
    images = glob.glob('*.tif')
    numlist.append(len(images))
    for input_img in images:
        if input_img is not None:
            input= cv2.imread(input_img)
            X.append(input)
            namelist.append(input_img)


    os.chdir(path2)
    print('Loaded the images of the folder-' + '{}\n'.format('Normal'))
    images = glob.glob('*.tif')
    numlist.append(len(images))
    for input_img in images:
        if input_img is not None:
            input = cv2.imread(input_img)
            X.append(input)
            namelist.append(input_img)

    X = np.array(X)
    return X, numlist, namelist

X, numlist, namelist= get_data(pathcantrain, pathnortrain)
X = X.astype('float32')
X = X * 1. / 255.

print(X.shape)  # (8603, 512, 512, 3)
# numlist #  [5737, 5712] Cancer, Normal
namelist = np.asarray(namelist)

Y = numlist[0]* ['Ca'] + numlist[1] * ['No']
Y_m = np.asarray(Y)

encoder = LabelEncoder()
encoder.fit(Y)
Y = np_utils.to_categorical(encoder.transform(Y))

# Shuffle the dataset
shuffled_index = shuffle(list(range(11573)), random_state=2)
# Split the dataset
name_train = np.asarray([namelist[n] for n in [shuffled_index[x] for x in list(range(9837))]])
name_val =   np.asarray([namelist[n] for n in [shuffled_index[x] for x in list(range(9837, 11573))]])
X_train = np.asarray([X[n] for n in [shuffled_index[x] for x in list(range(9837))]])
X_val = np.asarray([X[n] for n in [shuffled_index[x] for x in list(range(9837, 11573))]])
y_train = np.asarray([Y[n] for n in [shuffled_index[x] for x in list(range(9837))]])
y_val = np.asarray([Y[n] for n in [shuffled_index[x] for x in list(range(9837, 11573))]])

################################# Separate X, Y data into folders with the above shuffle #########################
# def CopyForOverWrite(list,dst_dir):
#     for tiffile in list:
#         shutil.copy(tiffile, dst_dir)
#
# import pandas as pd
# df = pd.DataFrame({'x':name_train, 'y':y_train})
# subdf = df[df['y'] == 'Ca']
# train_cancer_list = subdf['x'].as_matrix()
# subdf1 = df[df['y'] == 'No']
# train_normal_list = subdf1['x'].as_matrix()
#
# dir_train_can = '/data/edong/PycharmProjects/projpy/Colour_Mask_Padded_White_Samller_InterIMG/Train/Cancer'
# dir_train_nor = '/data/edong/PycharmProjects/projpy/Colour_Mask_Padded_White_Samller_InterIMG/Train/Normal'
# os.chdir(pathcantrain)
# CopyForOverWrite(train_cancer_list, dir_train_can)
# os.chdir(pathnortrain)
# CopyForOverWrite(train_normal_list, dir_train_nor)
#
# import pandas as pd
# df = pd.DataFrame({'x':name_val, 'y':y_val})
# subdf3 = df[df['y'] == 'Ca']
# val_cancer_list = subdf3['x'].as_matrix()
# subdf4 = df[df['y'] == 'No']
# val_normal_list = subdf4['x'].as_matrix()
#
# dir_val_can = '/data/edong/PycharmProjects/projpy/Colour_Mask_Padded_White_Samller_InterIMG/Val/Cancer'
# dir_val_nor = '/data/edong/PycharmProjects/projpy/Colour_Mask_Padded_White_Samller_InterIMG/Val/Normal'
# os.chdir(pathcantrain)
# CopyForOverWrite(val_cancer_list, dir_val_can)
# os.chdir(pathnortrain)
# CopyForOverWrite(val_normal_list, dir_val_nor)
############################################ Augmentation Way of Read In ########################################
# In total 11573 cell images.
nb_train_samples = 9837
nb_validation_samples = 1736
epochs = 100
batch_size = 128
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.00000000001)

# train_data_dir = '/data/edong/PycharmProjects/projpy/Colour_Mask_Padded_White_Samller_InterIMG_Train_Val_Splited/Train'
# validation_data_dir = '/data/edong/PycharmProjects/projpy/Colour_Mask_Padded_White_Samller_InterIMG_Train_Val_Splited/Val'

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range = 0.1,
    height_shift_range= 0.1,
    horizontal_flip=True,
    vertical_flip=True)

val_datagen = ImageDataGenerator(
    rescale=1. / 255,

    )
train_datagen.fit(X_train)
val_datagen.fit(X_val)

train_generator = train_datagen.flow(
    X_train, y_train,
    batch_size=batch_size)

validation_generator = val_datagen.flow(
    X_val, y_val,
    batch_size=batch_size)

hist = model.fit_generator(
    train_generator,
    steps_per_epoch= nb_train_samples//batch_size ,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks= [reduce_lr],
    validation_steps=nb_validation_samples//batch_size)


################################### Save Weights ############################################################
import os
os.chdir('/data/edong/PycharmProjects/projpy/Weights')
model.save('HFRN50_Ep100_Shift_lr1e-11_86.57%.h5')




# ############################ For Acc and Confusion Matrix ###################################################
# val_generator = val_datagen.flow_from_directory(
#         validation_data_dir,
#         target_size=(112, 112),
#         batch_size=batch_size,
#         class_mode=None,  # only data, no labels
#         shuffle=False)  # keep data in same order as labels
# y_prd = model.predict_generator(val_generator, 14)
#
# from sklearn.metrics import confusion_matrix
# y_true = np.array([0] * 870 + [1] * 866)
# y_pred = np.argmax(y_prd, axis=1)
# confusion_matrix(y_true, y_pred)
#

################################# Useless Experiment with width shift number #################################
# from matplotlib import pyplot
#
# # create a grid of 3x3 images
# for i in range(0, 9):
#     pyplot.subplot(330 + 1 + i)
#     pyplot.imshow(X_train[i], cmap=pyplot.get_cmap('gray'))
# # show the original images
# pyplot.show()
#
#
#
# ############################## Useless Experiment with Image Generator #####################################
# exp_datagen = ImageDataGenerator(
#     width_shift_range = 0.1)#,
#     #height_shift_range= 0.1)
#
# exp_datagen.fit(X_train)
#
# # configure batch size and retrieve one batch of images
# for X_batch, y_batch in exp_datagen.flow(X_train, y_train, batch_size=9):
#     # create a grid of 3x3 images
#     for i in range(0, 9):
#         pyplot.subplot(330 + 1 + i)
#         pyplot.imshow(X_batch[i], cmap=pyplot.get_cmap('gray'))
#     pyplot.show()# show the plot
#     break
#
#
