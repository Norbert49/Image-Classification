# preprocess.py

import numpy as np
import pickle
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer

def load_data():
    """
    Loads the CIFAR-10 dataset.

    Returns:
        x_train (numpy.ndarray): Training images.
        y_train (numpy.ndarray): Training labels.
        x_test (numpy.ndarray): Test images.
        y_test (numpy.ndarray): Test labels.
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    return x_train, y_train, x_test, y_test

def normalize_data(x_train, x_test):
    """
    Normalizes pixel values to the range [0, 1].

    Args:
        x_train (numpy.ndarray): Training images.
        x_test (numpy.ndarray): Test images.

    Returns:
        x_train_normalized (numpy.ndarray): Normalized training images.
        x_test_normalized (numpy.ndarray): Normalized test images.
    """
    x_train_normalized = x_train.astype('float32') / 255.0
    x_test_normalized = x_test.astype('float32') / 255.0
    return x_train_normalized, x_test_normalized

def label_binarizer(y_train, y_test):
    """
    Performs label binarization and saves the LabelBinarizer object.

    Args:
        y_train (numpy.ndarray): Training labels.
        y_test (numpy.ndarray): Test labels.

    Returns:
        y_train_binarized (numpy.ndarray): Binarized training labels.
        y_test_binarized (numpy.ndarray): Binarized test labels.
        label_transformer (LabelBinarizer): LabelBinarizer object for future use.
    """
    label_transformer = LabelBinarizer()
    y_train_binarized = label_transformer.fit_transform(y_train)
    y_test_binarized = label_transformer.transform(y_test)
    
    # Save the LabelBinarizer object for future use
    with open('label_transform.pkl', 'wb') as f:
        pickle.dump(label_transformer, f)

    return y_train_binarized, y_test_binarized, label_transformer

def create_data_generator():
    """
    Creates an ImageDataGenerator for data augmentation.

    Returns:
        train_gen (ImageDataGenerator): Configured data generator for training.
    """
    # Define the data augmentation generator for training data
    train_gen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    return train_gen

def save_as_pickle(file_path, data):
    """
    Saves data as a pickle file.

    Args:
        file_path (str): File path to save the pickle file.
        data: Data to be saved.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    # Load the data
    x_train, y_train, x_test, y_test = load_data()
    
    # Normalize pixel values
    x_train_normalized, x_test_normalized = normalize_data(x_train, x_test)
    
    # Label binarizer
    y_train_binarized, y_test_binarized = label_binarizer(y_train, y_test)

    # Create and configure the data generator
    train_gen = create_data_generator()

    # Save images and labels as pickle files
    save_as_pickle('x_train.pickle', x_train_normalized)
    save_as_pickle('y_train.pickle', y_train_binarized)
    save_as_pickle('x_test.pickle', x_test_normalized)
    save_as_pickle('y_test.pickle', y_test_binarized)

    print('Data preprocessing completed successfully.')
