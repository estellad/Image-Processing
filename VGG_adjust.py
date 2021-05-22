from __future__ import division

from keras.applications import VGG16, ResNet50
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, RMSprop, adam
import numpy as np
import os
# os.environ("THEANO_FLAGS")="FAST_RUN,device=gpu,floatX=float32"

# Read h5 file for pre-trained VGG16 weights
import h5py
filename = 'masknaug.h5'
f = h5py.File(filename, 'w')
# List all groups
print("Keys: %s" % f.keys())
a_group_key = f.keys()[0]
# Get the data
data = list(f[a_group_key])


# #Write HDF5
# #!/usr/bin/env python
# import h5py
# # Create random data
# import numpy as np
# data_matrix = np.random.uniform(-1, 1, size=(10, 3))
# # Write data to HDF5
# data_file = h5py.File('file.hdf5', 'w')
# data_file.create_dataset('group_name', data=data_matrix)
# data_file.close()

 ##############################################################################################
# Transfer Learning from a pre-trained VGG-16/Keras-Resnet
model1 = Sequential()
print('Starting training')
# model1 = application_vgg16(include_top = TRUE, weights = ,
#   input_tensor = NULL, input_shape = NULL, pooling = NULL,
#   classes = 1000)
from keras.models import Model
from keras.layers import Input
from keras import applications
num_classes = 2


base_model = applications.VGG16(include_top=False, weights='imagenet')
input = Input(shape=(1, 96, 96),name = 'image_input')
vgg_output = base_model(input)

top_model = Flatten()(vgg_output)
top_model = Dense(64, activation='relu')(top_model)
top_model = Dropout(0.5)(top_model)
top_model = Dense(64, activation='relu')(top_model)
top_model = Dropout(0.5)(top_model)
top_model = Dense(num_classes, activation='softmax', name='prediction_layer')(top_model)

model = Model(inputs=input, outputs=top_model)

# first: train only the top 2 vgg block layers, and the FC layer
# i.e. freeze all convolutional (InceptionV3) layers
layers = base_model.layers[:11]
for layer in layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=1, validation_data=(X_val, y_val))



###############################################################################################
# Train VGG from scratch
from keras. regularizers import l2
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers.advanced_activations import ELU


class VGG16_Builder(object):
    @staticmethod
    def build(input_shape = None, num_classes = 2, weight_initializer = None):
        """
        Builds a custom VGG with H.
        Args:
            input_shape:  One of none or channel last(rows, cols, channels)
            num_classes: The number of outcome for this classification problem, default is binary 2.
            weight_initializer: The weight initializer, he_normal or default glorot_uniform
        Returns:
            VGG16 model without compile.
        """
        if weight_initializer is None:
            weight_initializer = 'glorot_uniform'
        else:
            weight_initializer = 'he_normal'

        print('Creating the VGG16 model')
        model = Sequential()
        model.add(Conv2D(64, (3, 3), kernel_initializer= weight_initializer, padding='same', activation='relu',
                         input_shape=(input_shape)))
        model.add(Conv2D(64, (3, 3), kernel_initializer= weight_initializer, padding='same', activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Conv2D(128, (3, 3), kernel_initializer= weight_initializer, padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), kernel_initializer= weight_initializer, padding='same', activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Conv2D(256, (3, 3), kernel_initializer= weight_initializer, padding='same', activation='relu'))
        model.add(Conv2D(256, (3, 3), kernel_initializer= weight_initializer, padding='same', activation='relu'))
        model.add(Conv2D(256, (3, 3), kernel_initializer= weight_initializer, padding='same', activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Conv2D(512, (3, 3), kernel_initializer= weight_initializer, padding='same', activation='relu'))
        model.add(Conv2D(512, (3, 3), kernel_initializer= weight_initializer, padding='same', activation='relu'))
        model.add(Conv2D(512, (3, 3), kernel_initializer= weight_initializer, padding='same', activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Conv2D(512, (3, 3), kernel_initializer= weight_initializer, padding='same', activation='relu'))
        model.add(Conv2D(512, (3, 3), kernel_initializer= weight_initializer, padding='same', activation='relu'))
        model.add(Conv2D(512, (3, 3), kernel_initializer= weight_initializer, padding='same', activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(4096, kernel_initializer= weight_initializer, kernel_regularizer=l2(0.0001), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, kernel_initializer= weight_initializer, kernel_regularizer=l2(0.0001), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        return model


from keras.optimizers import RMSprop, Adam
#rmspropmo = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# adammo = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(X_train, y_train, batch_size=48, epochs=20, verbose=1, validation_data=(X_val, y_val))

# Save Weights
os.chdir( 'C:\\Users\\edong\\PycharmProjects\\proj.py\\Weights')
model.save('MaskedRelu.h5')

# model.save('model.hdf5')
# loaded_model=load_model('model.hdf5')
# # #################################################################################################################
# # # Transfer Learn - The Model
# # model_1 = VGG16(weights='imagenet')
# # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# # model_1.compile(optimizer=sgd, loss='categorical_crossentropy')
# # out = model_1.fit(X_train, y_train)
# # print(np.argmax(out))
# #
# # # On Test
# # model_1.predict(X_val, y_val)
# # #################################################################################
# # model_2 = ResNet50(weights='imagenet')
# # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# # model_2.compile(optimizer=sgd, loss='categorical_crossentropy')
# # model_2.fit(X_train, y_train)
# #
# # # On Test
# # model_2.predict(X_val, y_val)