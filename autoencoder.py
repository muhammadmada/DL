
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape
from keras.models import Model

class Autoencoder:
    def __init__(self, input_shape, encoding_dim):
        self.input_shape = input_shape
        self.encoding_dim = encoding_dim
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.autoencoder = self.build_autoencoder()

    def build_encoder(self):
        input_layer = Input(shape=self.input_shape)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2,2), padding="same")(x)
        encoder = Model(inputs=input_layer, outputs=encoded)
        return encoder

    def build_decoder(self):
        encoded_input = Input(shape=self.encoding_dim,)
        x = Reshape((256, 1, 1))(encoded_input)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
        decoder = Model(inputs=encoded_input, outputs=decoded)
        return decoder


    def build_autoencoder(self):
        input_layer = Input(shape=self.input_shape,)
        encoded = self.encoder(input_layer)
        decoded = self.decoder(encoded)
        autoencoder = Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        return autoencoder
