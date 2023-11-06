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
X_train_styler = [np.array(image) for image in X_train_styler]

input_shape = np.array(X_train_styler)
print(input_shape[0].shape)

input_dim = (256, 256, 3)
encoding_dim = (32, 32, 128)
learning_rate = 0.001

autoencoder = Autoencoder(input_dim, encoding_dim, learning_rate)

epochs = 32
batch_size = 64
total_samples = len(X_train_styler)
steps_per_epoch = total_samples // batch_size

custom_callback = CustomCallback()

optimizer = Adam(learning_rate=learning_rate)
autoencoder.autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')
autoencoder.autoencoder.summary()

total_samples = len(X_train_styler)
steps_per_epoch = total_samples // batch_size

for epoch in range(epochs):
    start_time = time.time()
    
    for step in range(steps_per_epoch):
        batch_start = step * batch_size
        batch_end = (step + 1) * batch_size
        x_batch = X_train_styler[batch_start:batch_end]
        
        # Assuming your images are already loaded as numpy arrays
        x_batch = np.array(x_batch)
        
        # Ensure the shape matches the expected input shape (256, 256, 3)
        x_batch = x_batch.reshape((-1, 256, 256, 3))
        
        loss = autoencoder.autoencoder.train_on_batch(x_batch, x_batch)
        
        print(f"Epoch {epoch + 1}/{epochs} - Step {step + 1}/{steps_per_epoch} - Loss: {loss:.4f}", end='\r')

    end_time = time.time()
    time_per_epoch = end_time - start_time
    iter_per_second = steps_per_epoch / time_per_epoch

    print(f"Epoch {epoch + 1}/{epochs} - Step {steps_per_epoch}/{steps_per_epoch} - Loss: {loss:.4f} - {iter_per_second:.2f} iter/s")

# autoencoder.autoencoder.fit(X_train_styler, X_train_styler,
#    epochs=epochs, steps_per_epoch=128,max_queue_size=100,
#    callbacks=custom_callback, verbose=1)



autoencoder.autoencoder.save('/mnt/d/Documents/Coolyeah/DL/models/autoencoder.h5')

print("Autoencoder training completed.")