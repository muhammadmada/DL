from preprocess import preprocess_images
from Loader import load_dataset_styler
from autoencoder import Autoencoder
import numpy as np

styler_path = r'/mnt/d/Documents/Coolyeah/DL/images/'
exts = {".jpg", ".jpeg", ".png"}

# Load the data using load_dataset_styler
X_train_styler = load_dataset_styler(styler_path, exts)

# Extract image paths from the list of tuples
image_paths = [path for _, class_images in X_train_styler for path in class_images]

# Preprocess the data
X_train_styler = preprocess_images(image_paths)

input_dim = (1080, 1080, 3)
encoding_dim = 1080

autoencoder = Autoencoder(input_dim, encoding_dim)

autoencoder.autoencoder.fit(X_train_styler, X_train_styler, epochs=10, batch_size=128, shuffle=True)
