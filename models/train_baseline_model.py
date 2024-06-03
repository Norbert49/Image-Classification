import sys
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Add parent directory of Image-Classification to the Python path
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

# Import from the preprocess submodule
from data.preprocess import load_data, normalize_data

# Import from the baseline_model submodule
from models.baseline_model import create_baseline_model

def main():
    # Load and preprocess data
    x_train, y_train, x_test, y_test = load_data()
    x_train_normalized, x_test_normalized = normalize_data(x_train, x_test)

    # Split the normalized training data into training and validation sets
    x_train_normalized, x_val_normalized, y_train, y_val = train_test_split(x_train_normalized, y_train, test_size=0.2, random_state=42)

    print(f'Training set shape: {x_train_normalized.shape}, {y_train.shape}')
    print(f'Validation set shape: {x_val_normalized.shape}, {y_val.shape}')
    print(f'Test set shape: {x_test_normalized.shape}, {y_test.shape}')

    y_train = tf.one_hot(y_train, depth=10)
    y_train = y_train[:, 0]

    print(f'y_val shape before one-hot: {y_val.shape}')
    y_val = y_val.reshape(-1)
    y_val = tf.one_hot(y_val, depth=10)
    print(f'y_val shape after one-hot: {y_val.shape}')

    # Define input shape
    input_shape = x_train_normalized[0].shape

    # Train the model
    baseline_model = create_baseline_model(input_shape=input_shape)
    history = baseline_model.fit(x_train_normalized, y_train, epochs=10, batch_size=32, validation_data=(x_val_normalized, y_val))

if __name__ == "__main__":
    main()
