from keras.layers import *
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import cv2
import matplotlib.pyplot as plt

((x_train, _), (x_test, _))  = mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255

x_train = np.expand_dims(x_train, axis = 3)
x_test = np.expand_dims(x_test, axis = 3)
noise_train = np.random.normal(loc=0, scale=0.5, size=x_train.shape)
x_train_noise = x_train + noise_train
noise_test = np.random.normal(loc=0, scale = 0.5, size=x_test.shape)
x_test_noise = x_test + noise_test
IMG_SHAPE = x_train.shape[1:]
# print(y_train.shape)



def model_autoencoder():
    input_img = Input(shape = (IMG_SHAPE))
    x = Conv2D(8, (3, 3), activation = 'relu', padding = 'same')(input_img)
    x = MaxPooling2D((2, 2), padding = 'same')(x)
    x = Conv2D(16, (3, 3), activation = 'relu', padding = 'same')(x)
    x = MaxPooling2D((2, 2), padding = 'same')(x)
    x = Conv2D(32, (3, 3), activation ='relu', padding = 'same')(x)
    encoded = MaxPooling2D((2, 2), padding = 'same')(x)

    x = Conv2D(32, (3, 3), padding = 'same', activation = 'relu')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), padding = 'same', activation = 'relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation = 'relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation = 'sigmoid', padding = 'same')(x)

    return Model(inputs = input_img, outputs = decoded)

model = model_autoencoder()
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.summary()

model.fit(x_train_noise, x_train, epochs = 50, shuffle = True)

decoded_imgs = model.predict(x_test)

n = 21
plt.figure(figsize=(20, 4))
for i in range(11, n):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test_noise[i - 10].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i- 10].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

loss, acc = model.evaluate(x_test_noise, x_test)

print(loss, acc) 

#accuracy 81%, features that are extracted are very similar to origin data (without noise)

