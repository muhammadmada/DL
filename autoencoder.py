import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Flatten, UpSampling2D, Dense, Reshape
from keras.models import Model

class Autoencoder:
    def __init__(self, input_shape, encoding_dim, learning_rate):
        self.input_shape = input_shape
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        self.optimizer_type = keras.optimizers.Adam(learning_rate=learning_rate)
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.autoencoder = self.build_autoencoder()

    def build_encoder(self):
        input_layer = Input(shape=self.input_shape)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        #x = MaxPooling2D((2, 2), padding='same')(x)
        #x = Flatten()(x)
        #encoded = Dense(512, activation='relu')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)
        encoder = Model(inputs=input_layer, outputs=encoded)
        return encoder

    def build_decoder(self):
        encoded_input = Input(shape=self.encoding_dim)
        # x = Dense(512, activation='relu')(encoded_input)
        # x = Reshape((32, 32, 512))(x)
        x = Conv2DTranspose(256, (3, 3), activation='relu', strides=(2, 2), padding='same')(encoded_input)
        x = Conv2DTranspose(128, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
        x = Conv2DTranspose(64, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
        decoded = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)
        decoder = Model(inputs=encoded_input, outputs=decoded)
        return decoder

    def build_autoencoder(self):
        input_layer = Input(shape=self.input_shape)
        encoded = self.encoder(input_layer)
        decoded = self.decoder(encoded)
        autoencoder = Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer=self.optimizer_type, loss='mean_squared_error')
        return autoencoder