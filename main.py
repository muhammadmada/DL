from preprocess import preprocess_images
from Loader import load_dataset_styler
from autoencoder import Autoencoder

styler_path = r'/mnt/d/Documents/Coolyeah/DL/images/'
exts = {".jpg", ".jpeg", ".png"}

X_train_styler = load_dataset_styler(styler_path, exts)

image_paths = [path for _, class_images in X_train_styler for path in class_images]


X_train_styler = preprocess_images(image_paths)

input_dim = (256, 256, 3)
encoding_dim = 256

autoencoder = Autoencoder(input_dim, encoding_dim)

autoencoder.autoencoder.fit(X_train_styler, X_train_styler, epochs=10, batch_size=128, shuffle=True)
