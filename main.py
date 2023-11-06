from preprocess import preprocess_images
from Loader import load_dataset_styler
from autoencoder import Autoencoder
from Callbacks import CustomCallback
import numpy as np
from keras.optimizers import Adam

styler_path = r'/mnt/d/Documents/Coolyeah/DL/images/'
exts = {".jpg", ".jpeg", ".png"}

X_train_styler = load_dataset_styler(styler_path, exts)

image_paths = [path for _, class_images in X_train_styler for path in class_images]

X_train_styler = preprocess_images(image_paths)
X_train_styler = [np.array(image) for image in X_train_styler]

input_shape = np.array(X_train_styler)
print(input_shape[0].shape)

input_dim = (256, 256, 3)
encoding_dim = (32, 32, 128)
learning_rate = 0.0001

autoencoder = Autoencoder(input_dim, encoding_dim, learning_rate)

epochs = 10
batch_size = 128
validation_split = 0.2

custom_callback = CustomCallback()

optimizer = Adam(learning_rate=learning_rate)
autoencoder.autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')
autoencoder.autoencoder.summary()



autoencoder.autoencoder.fit(X_train_styler, X_train_styler,
    epochs=epochs, steps_per_epoch=128, validation_split=0.2,
    callbacks=custom_callback, verbose=2)

autoencoder.autoencoder.save('/mnt/d/Documents/Coolyeah/DL/models/autoencoder.h5')

print("Autoencoder training completed.")