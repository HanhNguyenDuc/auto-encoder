import tensorflow as tf
from keras.layers import *
from keras.models import *
from keras.optimizers import *
import os
import cv2
from keras.callbacks import ModelCheckpoint
import numpy as np

TRAIN_DATA_DIR = 'small_vt_cards/train'

checkpoint = ModelCheckpoint('weight.hdf5', monitor = 'val_acc', save_best_only = True, mode = 'max')

IMG_SHAPE = (88, 312, 3)
x_train = []
y_train = []
i = 0

def model_autoencoding():
    input_img = Input(shape = (IMG_SHAPE))
    x = Conv2D(8, (3, 3), activation = 'relu', padding = 'same')(input_img)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation = 'relu', padding = 'same')(x)
    x = MaxPooling2D((2, 2), padding = 'same')(x)
    x = Conv2D(32, (3, 3), activation = 'relu', padding = 'same')(x)
    x = MaxPooling2D((2, 2), padding = 'same')(x)
    x = Conv2D(64, (3, 3), activation ='relu', padding = 'same')(x)
    encoded = MaxPooling2D((2, 2), padding = 'same')(x)

    x = Conv2D(64, (3, 3), padding = 'same', activation = 'relu')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding = 'same', activation = 'relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation = 'relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation = 'relu', padding = 'same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation = 'sigmoid', padding = 'same')(x)

    return Model(inputs = input_img, outputs = decoded)

model = model_autoencoding()
model.summary()
model.load_weights('weight.hdf5')

# model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# model.fit(x_train, y_train, epochs = 100, batch_size = 32, validation_split = 0.1, callbacks = [checkpoint], verbose = 2)
i = 0

for file in os.listdir(os.path.join(TRAIN_DATA_DIR, 'inputs')):
    img = cv2.imread(os.path.join(TRAIN_DATA_DIR, 'inputs/' + file))
    img = img / 255
    img = cv2.resize(img, (312, 88), interpolation = cv2.INTER_AREA)
    y_img = model.predict(np.array([img]))
    print(y_img.shape)
    cv2.imwrite('model_output/' + file, y_img[0] * 255)
    print('{} images has loaded!'.format(i))
    i+=1