import os
import shutil


def load_dataset_styler(root_path, valid_extensions):
    dataset = []
    total_images = 0  # Initialize a variable to keep track of the total number of images

    classes = os.listdir(root_path)

    for class_name in classes:
        class_path = os.path.join(root_path, class_name)
        if os.path.isdir(class_path):
            class_images = []
            for filename in os.listdir(class_path):
                if not filename.startswith('.'):  # Skip hidden files
                    if any(filename.endswith(ext) for ext in valid_extensions):
                        file_path = os.path.join(class_path, filename)
                        class_images.append(file_path)
            dataset.append((class_name, class_images))
            total_images += len(class_images)

    print(f"Total number of styler images loaded: {total_images}")
    return dataset

def load_dataset_target(root_path, valid_extensions):
    dataset = []
    data_shape = dataset.shape
    total_images = 0  # Initialize a variable to keep track of the total number of images

    classes = os.listdir(root_path)

    for class_name in classes:
        class_path = os.path.join(root_path, class_name)
        if os.path.isdir(class_path):
            class_images = []
            for filename in os.listdir(class_path):
                if not filename.startswith('.'):  # Skip hidden files
                    if any(filename.endswith(ext) for ext in valid_extensions):
                        file_path = os.path.join(class_path, filename)
                        class_images.append(file_path)
            dataset.append((class_name, class_images))
            total_images += len(class_images)

    print(f"Total number of training target images loaded: {total_images}")
    return dataset