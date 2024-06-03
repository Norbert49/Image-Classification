# baseline_model.py

# Libraries for data wrangling
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cv2 # Computer vision
import os  # Directory

# Libraries for modelling
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

# Data
from tensorflow.keras.datasets import cifar10

# Load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalizing pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Print data types and shapes
print("Training images:", type(x_train), x_train.shape)
print("Training labels:", type(y_train), y_train.shape)
print("Test images:", type(x_test), x_test.shape)
print("Test labels:", type(y_test), y_test.shape)

# Importing preprocessing functions from preprocess.py
from preprocess import load_data, normalize_data, create_data_generator

# Load and preprocess data using functions from preprocess.py
x_train, y_train, x_test, y_test = load_data()
x_train_normalized, x_test_normalized = normalize_data(x_train, x_test)
print(f'Training set shape: {x_train_normalized.shape}, {y_train.shape}')
print(f'Test set shape: {x_test_normalized.shape}, {y_test.shape}')

# Create data generator and visualize augmented images
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_train_normalized[i])
    plt.title(f'Label: {y_train[i]}')
    plt.axis('off')
plt.show()

datagen = create_data_generator()
datagen.fit(x_train_normalized)

plt.figure(figsize=(10, 10))
for x_batch, y_batch in datagen.flow(x_train_normalized, y_train, batch_size=9):
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x_batch[i])
        plt.title(f'Label: {y_batch[i]}')
        plt.axis('off')
    plt.show()
    break

# Split the normalized training data into training and validation sets
x_train_normalized, x_val_normalized, y_train, y_val = train_test_split(x_train_normalized, y_train, test_size=0.2, random_state=42)

print(f'Training set shape: {x_train_normalized.shape}, {y_train.shape}')
print(f'Validation set shape: {x_val_normalized.shape}, {y_val.shape}')
print(f'Test set shape: {x_test_normalized.shape}, {y_test.shape}')

# One-hot encode the labels
y_train = tf.one_hot(y_train, depth=10)
y_train = y_train[:, 0]  # Remove extra dimension

print(f'y_val shape before one-hot: {y_val.shape}')
y_val = y_val.reshape(-1)
y_val = tf.one_hot(y_val, depth=10) 
print(f'y_val shape after one-hot: {y_val.shape}')

# Import TensorFlow
import tensorflow as tf

# Define the input shape for the model
input_shape = x_train_normalized[0].shape

# Create baseline model
def create_baseline_model(input_shape, num_classes=10):
    model = Sequential()
    chanDim = -1
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, x_train, y_train, x_val, y_val, epochs=10, batch_size=32):
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val))
    return history

# Train the baseline model
baseline_model = create_baseline_model(input_shape=input_shape)
baseline_history = train_model(baseline_model, x_train_normalized, y_train, x_val_normalized, y_val, epochs=10, batch_size=32)
