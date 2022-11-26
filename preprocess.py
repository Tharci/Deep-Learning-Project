import tensorflow as tf
import numpy as np
import math


def create_dataflows(images_path, img_size, batch_size):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        # shear_range=0.2,
        # zoom_range=zoom_range,
        horizontal_flip=True,
        validation_split=0.2,
        # featurewise_center=True
    )

    train_ds = train_datagen.flow_from_directory(
        images_path,
        target_size=img_size,
        batch_size=batch_size,
        color_mode='rgb',
        class_mode=None,
        subset='training'
    )

    test_ds = train_datagen.flow_from_directory(
        images_path,
        target_size=img_size,
        batch_size=batch_size,
        color_mode='rgb',
        class_mode=None,
        subset='validation'
    )

    return train_ds, test_ds


def create_dataflows_grayscale(images_path, img_size, batch_size):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        # shear_range=0.2,
        # zoom_range=zoom_range,
        horizontal_flip=True,
        validation_split=0.2,
        # featurewise_center=True
    )

    train_ds = train_datagen.flow_from_directory(
        images_path,
        target_size=img_size,
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode=None,
        subset='training'
    )

    test_ds = train_datagen.flow_from_directory(
        images_path,
        target_size=img_size,
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode=None,
        subset='validation'
    )

    return train_ds, test_ds


def crop_img(img, crop_size):
    y = np.random.randint(0, img.shape[0] - crop_size[0])
    x = np.random.randint(0, img.shape[1] - crop_size[1])
    return img[y:y+crop_size[0], x:x+crop_size[1]]


# shifts img from [-1,1] to [0,1]
def decenter_img(img):
    return img / 2 + 0.5


# shifts img from [0,1] to [-1,1]
def center_img(img):
    return (img - 0.5) * 2


def slice_img(img, slice_size):
    padded_size = (math.ceil(img.shape[0] / float(slice_size[0])) * slice_size[0],
                math.ceil(img.shape[1] / float(slice_size[1])) * slice_size[1],
                img.shape[2])

    padded_img = np.zeros(padded_size)
    padded_img[0:img.shape[0], 0:img.shape[1]] = img
    M, N = slice_size
    slices = np.array(
            [padded_img[x:x+M, y:y+N] for x in range(0, padded_img.shape[0], M)
                                      for y in range(0, padded_img.shape[1], N)]
    )
    return slices


def deslice_img(slices, img_size):
    slice_size = slices[0].shape
    img = np.zeros(img_size)

    y, x = 0, 0
    for s in slices:
        y_to = min(y+slice_size[0], img.shape[0])
        x_to = min(x+slice_size[1], img.shape[1])
        img[y:y_to, x:x_to] = s[0:y_to-y, 0:x_to-x]
        if x + slice_size[1] >= img.shape[1]:
            x = 0
            y += slice_size[0]
        else:
            x += slice_size[1]

    return img






