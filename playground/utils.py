import tensorflow
from keras import Model
from keras.layers import Conv2D, BatchNormalization, Activation, Conv2DTranspose, Input
from preprocess import decenter_img


#def ssim_loss(y_true, y_pred):
#    return 1 - tensorflow.reduce_mean(tensorflow.image.ssim_multiscale(y_true, y_pred, 1.0, filter_size=3))


def my_conv(x, filters, kernel_size=3, strides=(2, 2), padding='same', kernel_initializer='he_normal', name=None):
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer, name=name)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def my_convTrans(x, filters, kernel_size=3, strides=(2, 2), padding='same', kernel_initializer='he_normal', name=None):
    x = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer, name=name)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def split_autoencoder(autoencoder):
    # Get encoder
    encoder = Model(inputs=autoencoder.layers[0].input, outputs=autoencoder.get_layer('encoded').output)
    decoder_input = Input(shape=(12, 12, 32))
    decoder = Model(inputs=autoencoder.get_layer('decoder_input').input, outputs=autoencoder.layers[-1].output)

    return encoder, decoder


def create_learning_scheduler(steps_per_epoch, initial_learning=0.01, decay_rate=0.8):
    # decay_rate=0.8 - training seems to be faster for ~20 epochs
    return tensorflow.keras.optimizers.schedules.ExponentialDecay(
        initial_learning,
        decay_steps=steps_per_epoch,
        decay_rate=decay_rate,
        staircase=True)


class LearningRateLoggingCallback(tensorflow.keras.callbacks.Callback):
    # prints learning rate during training
    # can be added to 'autoencoder.fit', argument 'callbacks'
    def __init__(self, steps_per_epoch):
        super().__init__()
        self.steps_per_epoch = steps_per_epoch
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        print(f' Learning rate = {lr(epoch*self.steps_per_epoch)}')