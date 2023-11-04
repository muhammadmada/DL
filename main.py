from preprocess import preprocess_images
from Loader import load_dataset_styler
from autoencoder import Autoencoder
import numpy as np

styler_path = r'/mnt/d/Documents/Coolyeah/DL/images/'
exts = {".jpg", ".jpeg", ".png"}

X_train_styler = load_dataset_styler(styler_path, exts)

image_paths = [path for _, class_images in X_train_styler for path in class_images]


X_train_styler = preprocess_images(image_paths)
X_train_styler = [np.array(image) for image in X_train_styler]

input_dim = (256, 256, 3)
encoding_dim = (32, 32, 128)

autoencoder = Autoencoder(input_dim, encoding_dim)

epochs = 32
batch_size = 128

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    total_loss = 0.0
    num_batches = len(X_train_styler) // batch_size

    for i in range(0, len(X_train_styler), batch_size):
        batch_X = X_train_styler[i:i + batch_size]
        batch_loss = 0

        for image in batch_X:
            image = image.reshape(1, 256, 256, 3)

            loss = autoencoder.autoencoder.train_on_batch(image, image)
            batch_loss += loss

        total_loss += batch_loss / len(batch_X)

    average_loss = total_loss / num_batches
    print(f"Epoch {epoch + 1} - Average Loss: {average_loss:.4f}")
autoencoder.autoencoder.save('/mnt/d/Documents/Coolyeah/DL/models/autoencoder.h5')
print("Autoencoder training completed.")