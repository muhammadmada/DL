from keras.models import load_model
from preprocess import preprocess_images
from Loader import load_dataset_styler
from autoencoder import Autoencoder
from Callbacks import CustomCallback
import numpy as np
import time
from keras.optimizers import Adam

# Load the pretrained model
pretrained_model = load_model('/mnt/d/Documents/Coolyeah/DL/models/autoencoder-10_11_23-12_23_27-1024-256.keras')

# Compile the model with a new configuration
pretrained_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

styler_path = r'/mnt/d/Documents/Coolyeah/DL/images/'
exts = {".jpg", ".jpeg", ".png"}

X_train_styler = load_dataset_styler(styler_path, exts)

image_paths = [path for _, class_images in X_train_styler for path in class_images]

X_train_styler = preprocess_images(image_paths)
X_train_styler = np.array(X_train_styler)

print(X_train_styler[0].shape)

latent_space_dim = 256
learning_rate = 1e-4

optimizer = Adam(learning_rate=learning_rate)
pretrained_model.compile(optimizer=optimizer, loss='mean_squared_error')
pretrained_model.summary()

epochs = 1024
batch_size = 256
custom_callback = CustomCallback()


pretrained_model.fit(X_train_styler, X_train_styler, 
                  batch_size=batch_size,epochs=epochs,
                  callbacks= None, verbose=1)
#autoencoder.train_on_batch(x_train=X_train_styler, epochs = epochs,
#                           batch_size = batch_size)

time_now = time.localtime()
time_now = time.strftime("%d_%m_%y-%H_%M_%S",time_now)

pretrained_model.save(f'/mnt/d/Documents/Coolyeah/DL/models/retrained_autoencoder-'+ time_now +'-'+ str(epochs) +'-' + str(latent_space_dim) +'.keras')

print("Autoencoder training completed.")