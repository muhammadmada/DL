from preprocess import preprocess_images
from Loader import load_dataset_styler
from autoencoder import Autoencoder
from Callbacks import CustomCallback
import numpy as np
import time
from keras.optimizers import Adam

styler_path = r'/mnt/d/Documents/Coolyeah/DL/images/'
exts = {".jpg", ".jpeg", ".png"}

X_train_styler = load_dataset_styler(styler_path, exts)

image_paths = [path for _, class_images in X_train_styler for path in class_images]

X_train_styler = preprocess_images(image_paths)
X_train_styler = np.array(X_train_styler)

print(X_train_styler[0].shape)
print(X_train_styler.shape)

input_dim = (128, 128, 3)
conv_filters = (2, 16, 64, 128)
conv_kernels = (3, 5, 5, 3)
conv_strides = (1, 2, 2, 1)
latent_space_dim = 256
learning_rate = 1e-4

autoencoder = Autoencoder(input_dim, 
            conv_filters, conv_kernels, conv_strides, 
            latent_space_dim, learning_rate)

optimizer = Adam(learning_rate=learning_rate)
autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')
autoencoder.summary()

epochs = 1024
batch_size = 256
custom_callback = CustomCallback()


autoencoder.train(X_train_styler, X_train_styler, 
                  batch_size=batch_size,num_epochs=epochs,
                  callbacks= None, verbose=1)
#autoencoder.train_on_batch(x_train=X_train_styler, epochs = epochs,
#                           batch_size = batch_size)

time_now = time.localtime()
time_now = time.strftime("%d_%m_%y-%H_%M_%S",time_now)

autoencoder.save(f'/mnt/d/Documents/Coolyeah/DL/models/autoencoder-'+ time_now +'-'+ str(epochs) +'-' + str(latent_space_dim) +'.keras')

print("Autoencoder training completed.")