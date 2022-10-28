import tensorflow as tf
import numpy as np


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


def crop_img(img, crop_size):
    y = np.random.randint(0, img.shape[0] - crop_size[0])
    x = np.random.randint(0, img.shape[1] - crop_size[1])
    return img[y:y+crop_size[0], x:x+crop_size[1]]


# TODO






