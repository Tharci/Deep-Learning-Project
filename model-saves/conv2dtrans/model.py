def create_model(image_width, image_height):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(2, 2), activation='relu', kernel_initializer='he_normal', input_shape=(image_width, image_height, 1)))
    model.add(Conv2D(16, kernel_size=(3, 3), strides=(2, 2), activation='relu', kernel_initializer='he_normal'))
    model.add(Conv2D(16, kernel_size=(3, 3), strides=(2, 2), activation='relu', kernel_initializer='he_normal', name='encoded'))
    model.add(Conv2DTranspose(16, kernel_size=(3,3), strides=(2, 2), activation='relu', kernel_initializer='he_normal'))
    model.add(Conv2DTranspose(16, kernel_size=(3,3), strides=(2, 2), activation='relu', kernel_initializer='he_normal'))
    model.add(Conv2DTranspose(32, kernel_size=(3,3), strides=(2, 2), activation='relu', kernel_initializer='he_normal'))
    model.add(ZeroPadding2D(((1, 0), (1, 0))))
    model.add(Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same', name='decoded'))

    print(("shape of encoded", K.int_shape(model.get_layer('encoded').output)))
    print(("shape of decoded", K.int_shape(model.get_layer('decoded').output)))

    return model

autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005), loss=ssim_loss)

'''
After 70 epochs:
Epoch 48/50
106/106 [==============================] - 43s 411ms/step - loss: 0.0077 - val_loss: 0.0076
Epoch 49/50
106/106 [==============================] - 43s 411ms/step - loss: 0.0082 - val_loss: 0.0088
Epoch 50/50
106/106 [==============================] - 45s 429ms/step - loss: 0.0079 - val_loss: 0.0078


After 170 epochs:
Epoch 98/100
106/106 [==============================] - 44s 422ms/step - loss: 0.0066 - val_loss: 0.0071
Epoch 99/100
106/106 [==============================] - 45s 425ms/step - loss: 0.0067 - val_loss: 0.0066
Epoch 100/100
106/106 [==============================] - 45s 424ms/step - loss: 0.0067 - val_loss: 0.0068

Findings:
    use 200x200 cropped images -> computational and memory eff.
'''