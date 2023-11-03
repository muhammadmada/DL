from PIL import Image
import os

target_size = (1080, 1080)

def preprocess_image(input_path):
    image = Image.open(input_path)
    image = image.resize(target_size, Image.Resampling.NEAREST)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image

def preprocess_images(input_paths):
    preprocessed_images = []
    for input_path in input_paths:
        preprocessed_image = preprocess_image(input_path)
        preprocessed_images.append(preprocessed_image)
    return preprocessed_images
