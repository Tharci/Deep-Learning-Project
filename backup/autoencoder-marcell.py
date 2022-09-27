
#### LOAD DATASET ####

import cv2
from pathlib import Path
import numpy as np


IMAGES_PATH = '../Deep-Learning-Project-Data/cyberpunk'


def read_images(path):
    files = [x for x in path.iterdir() if x.is_file()]

    l = []
    for path in files:
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        l.append(img)
    l = np.array(l)
    return l


def train_test_split(data):
    np.random.shuffle(data)
    train_size = int(len(data) * .8)
    return data[:train_size], data[train_size:]


images = read_images(Path(IMAGES_PATH)) / 255.
x_test, x_train = train_test_split(images)

print(images.shape)


#### TRAIN AUTOENCODER ####

from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, AveragePooling2D
from keras import Sequential
import keras
from matplotlib import pyplot as plt


def show_imgs(x_test, n=10):
    sz = x_test.shape[1]
    plt.figure(figsize=(100, 50))
    for i in range(n):
        ax = plt.subplot(1, n, i+1)
        plt.imshow(x_test[i].reshape(sz, sz, 3))
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


autoencoder = Sequential([
    Conv2D(input_shape=(360, 360, 3), filters=16, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), padding='same'),
    Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'),
    AveragePooling2D(pool_size=(3, 3), padding='same'),
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
    AveragePooling2D(name='compressed', pool_size=(4, 4), padding='same'),

    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
    UpSampling2D((4, 4)),
    Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'),
    UpSampling2D((3, 3)),
    Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'),
    UpSampling2D((2, 2)),
    Conv2D(filters=3, kernel_size=(5, 5), padding='same', activation='sigmoid'),
])

print(autoencoder.summary())


autoencoder.compile(optimizer=keras.optimizers.SGD(learning_rate=1, momentum=0.9), loss='mse')
autoencoder.fit(x_train, x_train, epochs=3, batch_size=100,
                shuffle=True, validation_data=(x_test, x_test), verbose=1)
               #shuffle=True, validation_data=(x_test, x_test), verbose=1)


decoded_imgs = autoencoder.predict(x_test)
print("input (upper row)")
show_imgs(x_test)
print("decoded (bottom row)")
show_imgs(decoded_imgs)