def create_improved_baseline_model_12x_comp(image_width, image_height):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(7, 7), strides=(2, 2), kernel_initializer='he_normal',
                     padding='same', input_shape=(image_width, image_height, 3)))
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
    model.add(Conv2DTranspose(3, kernel_size=(3, 3), strides=(2, 2), activation='sigmoid', padding='same', name='decoded'))

    print(("shape of encoded", K.int_shape(model.get_layer('encoded').output)))
    print(("shape of decoded", K.int_shape(model.get_layer('decoded').output)))

    return model


autoencoder = create_improved_baseline_model_12x_comp(image_width, image_height)
autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=0.002), loss=ssim_loss)
autoencoder.fit(train_ds_prep,
                validation_data = test_ds_prep,
                steps_per_epoch = train_ds.n // train_ds.batch_size,
                validation_steps = test_ds.n // test_ds.batch_size,
                epochs=100,
                verbose=1)
