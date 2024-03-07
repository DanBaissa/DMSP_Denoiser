# model.py
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam


def create_model(conv_size):
    model = Sequential()
    model.add(Conv2D(32, (conv_size, conv_size), activation='relu', padding='same', input_shape=(None, None, 1)))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(32, (conv_size, conv_size), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(32, (conv_size, conv_size), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(32, (conv_size, conv_size), activation='relu', padding='same'))

    # Decoding layers (upsampling and convolution)
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (conv_size, conv_size), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (conv_size, conv_size), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (conv_size, conv_size), activation='relu', padding='same'))
    model.add(Conv2D(1, (conv_size, conv_size), activation='relu', padding='same'))

    # Compile the model with the SGD optimizer and momentum
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    return model
