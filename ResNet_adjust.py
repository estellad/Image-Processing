## ResNet 50
import keras

# Original Res Block - <Deep Residual Learning for Image Recognition> (2015 winner)
# Original: Xl -> Weight -> BN -> ReLu -> Weight -> BN -> addition -> ReLu -> Xl+1
'''ResNet50 model for Keras.
# Reference:
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
Adapted from code contributed by BigMoyan.
'''
from __future__ import print_function

import numpy as np
import warnings
from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D  # Keras default padding = "valid", no padding and some edges were dropped.
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.preprocessing import image
import keras.backend as K
K.set_image_dim_ordering('tf')
from keras.engine.topology import get_source_inputs
from collections import Counter
from keras.callbacks import ReduceLROnPlateau

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


def identity_block(input_tensor, kernel_size, filters, stage, block):

    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding="same", name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x

# def identity_blockREG(input_tensor, kernel_size, filters, stage, block):
#     filters1, filters2, filters3 = filters
#     if K.image_data_format() == 'channels_last':
#         bn_axis = 3
#     else:
#         bn_axis = 1
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'
#
#     x = Conv2D(filters1, (1, 1),
#                kernel_initializer="he_normal",
#                kernel_regularizer=l2(0.0001),
#                name=conv_name_base + '2a')(input_tensor)
#     x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
#     x = Activation('relu')(x)
#
#     x = Conv2D(filters2, kernel_size,
#                padding="same",
#                kernel_initializer="he_normal",
#                kernel_regularizer=l2(0.0001),
#                name=conv_name_base + '2b')(x)
#     x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
#     x = Activation('relu')(x)
#
#     x = Conv2D(filters3, (1, 1),
#                kernel_initializer="he_normal",
#                kernel_regularizer=l2(0.0001),
#                name=conv_name_base + '2c')(x)
#     x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
#
#     x = layers.add([x, input_tensor])
#     x = Activation('relu')(x)
#     return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

# def conv_blockREG(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
#     """conv_block is the block that has a conv layer at shortcut
#     # Arguments
#         input_tensor: input tensor
#         kernel_size: defualt 3, the kernel size of middle conv layer at main path
#         filters: list of integers, the filterss of 3 conv layer at main path
#         stage: integer, current stage label, used for generating layer names
#         block: 'a','b'..., current block label, used for generating layer names
#     # Returns
#         Output tensor for the block.
#     Note that from stage 3, the first conv layer at main path is with strides=(2,2)
#     And the shortcut should have strides=(2,2) as well
#     """
#     filters1, filters2, filters3 = filters
#     if K.image_data_format() == 'channels_last':
#         bn_axis = 3
#     else:
#         bn_axis = 1
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'
#
#     x = Conv2D(filters1, (1, 1), strides=strides,
#                kernel_initializer="he_normal",
#                kernel_regularizer=l2(0.0001),
#                name=conv_name_base + '2a')(input_tensor)
#     x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
#     x = Activation('relu')(x)
#
#     x = Conv2D(filters2, kernel_size, padding='same',
#                kernel_initializer="he_normal",
#                kernel_regularizer=l2(0.0001),
#                name=conv_name_base + '2b')(x)
#     x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
#     x = Activation('relu')(x)
#
#     x = Conv2D(filters3, (1, 1),
#                kernel_initializer="he_normal",
#                kernel_regularizer=l2(0.0001),
#                name=conv_name_base + '2c')(x)
#     x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
#
#     shortcut = Conv2D(filters3, (1, 1), strides=strides,
#                       kernel_initializer="he_normal",
#                       kernel_regularizer=l2(0.0001),
#                       name=conv_name_base + '1')(input_tensor)
#     shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
#
#     x = layers.add([x, shortcut])
#     x = Activation('relu')(x)
#     return x


def ResNet50(include_top=True, weights=None,
             input_tensor=None, input_shape=None,
             pooling=None,
             classes=2):

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random -> he_normal?) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Without determine proper input shape, input_shape = input_shape

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    # x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    # x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    #
    # x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    # x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    # x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    # x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    #
    # x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    #
    # x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    # x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    # x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = conv_block(x, 3, [32, 32, 128], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [32, 32, 128], stage=2, block='b')
    x = identity_block(x, 3, [32, 32, 128], stage=2, block='c')

    x = conv_block(x, 3, [64, 64, 256], stage=3, block='a')
    x = identity_block(x, 3, [64, 64, 256], stage=3, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=3, block='c')
    x = identity_block(x, 3, [64, 64, 256], stage=3, block='d')

    x = conv_block(x, 3, [128, 128, 512], stage=4, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=4, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=4, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=4, block='d')
    x = identity_block(x, 3, [128, 128, 512], stage=4, block='e')
    x = identity_block(x, 3, [128, 128, 512], stage=4, block='f')

    x = conv_block(x, 3, [256, 256, 1024], stage=5, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=5, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=5, block='c')

    x = AveragePooling2D((4, 4),  name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc2')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50')

    return model

input_shape = (112,112,3)
model = ResNet50(include_top=True, input_shape = input_shape)
#sgd = SGD(lr=0.01, decay=2e-4, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

# if __name__ == '__main__':


# img_path = '/data/edong/PycharmProjects/projpy/CancerOriginal/00-3734A_Thionin_Cancer_FEU_00000_1_40x.tif'
# img = image.load_img(img_path, target_size=(224, 224))
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
# print('Input image shape:', x.shape)
#
# Test a image:
# preds = model.predict(x)
# print('Predicted:', decode_predictions(preds))


############################################# Decrease Learning Rate & Class Weight ##################################################
# def scheduler(epoch):
#     if epoch == 10:
#         model.optimizer.lr.assign(model.optimizer.lr/10)
#     if epoch == 60:
#         model.optimizer.lr.assign(model.optimizer.lr/10)
#     if epoch == 80:
#         model.optimizer.lr.assign(model.optimizer.lr/10)
#     if epoch == 100:
#         model.optimizer.lr.assign(model.optimizer.lr/10)
#     return K.eval(model.optimizer.lr)
#
#
# def get_class_weights(y):
#     counter = Counter(y)
#     majority = max(counter.values())
#     return  {cls: float(majority/count) for cls, count in counter.items()}
# # class {0:5, 1:1, 2:1.25} #
# # class_weight = {0: 3.1823043266893536, 1: 1.0}
# class_weight = get_class_weights(y)



# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
#                               patience=5, min_lr=0.00001)
# hist = model.fit(X_train, y_train, epochs = 80, batch_size = 128, callbacks=[reduce_lr], validation_data=(X_val, y_val))




# Read h5 file ###################
# import h5py
# filename = 'HFRN50_Ep160_lr1e-9_87.56%.h5'
# f = h5py.File(filename, 'r')
#
# # List all groups
# print("Keys: %s" % f.keys())
# a_group_key = list(f.keys())[0]
#
# # Get the data
# data = list(f[a_group_key])