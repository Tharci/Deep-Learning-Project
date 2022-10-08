def create_improved_baseline_model_simple(image_width, image_height):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(7, 7), strides=(2, 2), kernel_initializer='he_normal',
                     padding='same', input_shape=(image_width, image_height, 1)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(.2))

    model.add(Conv2D(32, kernel_size=(3, 3), strides=(2, 2), kernel_initializer='he_normal', padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), kernel_initializer='he_normal', padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(.2))

    model.add(Conv2D(32, kernel_size=(3, 3), strides=(2, 2), kernel_initializer='he_normal', padding='valid'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(name='encoded', dtype='float16'))

    model.add(Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer='he_normal'))
    model.add(Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(Conv2DTranspose(16, kernel_size=(7, 7), strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(Conv2DTranspose(1, kernel_size=(3, 3), strides=(2, 2), activation='sigmoid', padding='same', name='decoded'))

    print(("shape of encoded", K.int_shape(model.get_layer('encoded').output)))
    print(("shape of decoded", K.int_shape(model.get_layer('decoded').output)))

    return model


'''
100 epochs:

106/106 [==============================] - 63s 592ms/step - loss: 0.0305 - val_loss: 0.0239
Epoch 2/30
106/106 [==============================] - 63s 593ms/step - loss: 0.0290 - val_loss: 0.0225
Epoch 3/30
106/106 [==============================] - 65s 605ms/step - loss: 0.0295 - val_loss: 0.0209
Epoch 4/30
106/106 [==============================] - 68s 640ms/step - loss: 0.0293 - val_loss: 0.0204
Epoch 5/30
106/106 [==============================] - 63s 590ms/step - loss: 0.0288 - val_loss: 0.0215
Epoch 6/30
106/106 [==============================] - 63s 598ms/step - loss: 0.0318 - val_loss: 0.0234
'''