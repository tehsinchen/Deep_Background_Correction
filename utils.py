from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
import numpy as np
import os
import tifffile
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale


def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="grayscale",
                   mask_color_mode="grayscale", image_save_prefix="img", mask_save_prefix="bg",
                   save_to_dir=None, target_size=(128, 128), seed=1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)

    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        yield (img, mask)


def testGenerator(test_path, test_folder, batch_size):
    test_datagen = ImageDataGenerator(rescale=1.0/65535)
    test_generator = test_datagen.flow_from_directory(
        test_path,
        classes=[test_folder],
        class_mode=None,
        color_mode="grayscale",
        target_size=(128, 128),
        batch_size=batch_size,
        shuffle=False)
    return test_generator


def step_decay_schedule(initial_lr, decay_factor, step_size):

    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch / step_size))

    return LearningRateScheduler(schedule)


def get_file_directory(foldername, filetype):
    file_directory = []
    for root, directories, files in os.walk(foldername):
        for filename in files:
            if filename.endswith(filetype):
                filepath = os.path.join(root, filename)
                file_directory.append(filepath)
    return file_directory
