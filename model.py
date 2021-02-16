import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *


def unet(input_size=(128, 128, 1)):

    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(BatchNormalization(trainable=True)(inputs))
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(BatchNormalization(trainable=True)(conv1))
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(BatchNormalization(trainable=True)(pool1))
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(BatchNormalization(trainable=True)(conv2))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(BatchNormalization(trainable=True)(pool2))
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(BatchNormalization(trainable=True)(conv3))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(BatchNormalization(trainable=True)(pool3))
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(BatchNormalization(trainable=True)(conv4))
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(BatchNormalization(trainable=True)(pool4))
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(BatchNormalization(trainable=True)(conv5))
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(BatchNormalization(trainable=True)(merge6))
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(BatchNormalization(trainable=True)(conv6))

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(BatchNormalization(trainable=True)(merge7))
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(BatchNormalization(trainable=True)(conv7))

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(BatchNormalization(trainable=True)(merge8))
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(BatchNormalization(trainable=True)(conv8))

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(BatchNormalization(trainable=True)(merge9))
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(BatchNormalization(trainable=True)(conv9))
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(BatchNormalization(trainable=True)(conv9))
    conv10 = Conv2D(1, 1, activation='sigmoid')(BatchNormalization(trainable=True)(conv9))

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    return model
