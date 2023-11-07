from preprocess import preprocess_images
from Loader import load_dataset_styler
from autoencoder import Autoencoder
from Callbacks import CustomCallback
import numpy as np
import time
from keras.optimizers import RMSprop

styler_path = r'/mnt/d/Documents/Coolyeah/DL/images/'
exts = {".jpg", ".jpeg", ".png"}

X_train_styler = load_dataset_styler(styler_path, exts)

image_paths = [path for _, class_images in X_train_styler for path in class_images]

X_train_styler = preprocess_images(image_paths)
X_train_styler = [np.array(image) for image in X_train_styler]

input_shape = np.array(X_train_styler)
print(input_shape[0].shape)

input_dim = (256, 256, 3)
conv_filters = (32, 64, 64, 64)
conv_kernels = (3, 5, 5, 3)
conv_strides = (1, 2, 2, 1)
latent_space_dim = 2
learning_rate = 1e-2

autoencoder = Autoencoder(input_dim, 
            conv_filters, conv_kernels, conv_strides, 
            latent_space_dim, learning_rate)

optimizer = RMSprop(learning_rate=learning_rate, momentum=0.9)
autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')
autoencoder.summary()

epochs = 2
batch_size = 4
custom_callback = CustomCallback()


autoencoder.train(X_train_styler,batch_size=batch_size,num_epochs=epochs,
                  callbacks=custom_callback, verbose=1)
#autoencoder.train_on_batch(x_train=X_train_styler, epochs = epochs,
#                           batch_size = batch_size)


autoencoder.autoencoder.save(f'/mnt/d/Documents/Coolyeah/DL/models/autoencoder.{int(time.time())}.h5')

print("Autoencoder training completed.")