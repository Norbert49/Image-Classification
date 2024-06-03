# train_transfer_learning_model.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import sys
import os

# Adding parent directory of Image-Classification to the Python path
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

# Importing from the preprocess submodule
from data.preprocess import load_data, normalize_data

def build_transfer_learning_model(input_shape, n_classes, INIT_LR):
    base_inception = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_inception.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=base_inception.input, outputs=predictions)

    for layer in base_inception.layers:
        layer.trainable = False

    opt = Adam(learning_rate=INIT_LR)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    return model

def fine_tune_model(model, x_train, y_train, x_val, y_val, init_lr=1e-3, epochs=10, batch_size=32):
    for layer in model.layers[:-10]:
        layer.trainable = False

    opt = Adam(learning_rate=init_lr)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        verbose=1
    )
    return history

x_train, y_train, x_test, y_test = load_data()
x_train_normalized, x_test_normalized = normalize_data(x_train, x_test)

# Splitting the normalized training data into training and validation sets
x_train_normalized, x_val_normalized, y_train, y_val = train_test_split(x_train_normalized, y_train, test_size=0.2, random_state=42)

print(f'Training set shape: {x_train_normalized.shape}, {y_train.shape}')
print(f'Validation set shape: {x_val_normalized.shape}, {y_val.shape}')
print(f'Test set shape: {x_test_normalized.shape}, {y_test.shape}')

# One-hot encoding
y_train_encoded = to_categorical(y_train, num_classes=10)
y_val_encoded = to_categorical(y_val, num_classes=10)
y_test_encoded = to_categorical(y_test, num_classes=10)

print(f'Encoded shape: {y_train_encoded.shape}')
print(f'X train normalized shape: {x_train_normalized.shape}')
print(f'X test normalized shape: {x_test_normalized.shape}')

# Resizing the input data to match the model's expected shape
x_train_normalized_resized = tf.image.resize(x_train_normalized, (128, 128))
x_val_normalized_resized = tf.image.resize(x_val_normalized, (128, 128))
x_test_normalized_resized = tf.image.resize(x_test_normalized, (128, 128))

#variable definition
input_shape = (128, 128, 3)
n_classes = 10
INIT_LR = 1e-3
EPOCHS = 10
BS = 32

# Creating and compiling the InceptionV3 model
transfer_learning_model = build_transfer_learning_model(input_shape, n_classes, INIT_LR)
transfer_learning_model.summary()

# Fine-tuning the model
fine_tune_history = fine_tune_model(
    transfer_learning_model,
    x_train_normalized_resized,
    y_train_encoded,
    x_val_normalized_resized,
    y_val_encoded,
    init_lr=INIT_LR,
    epochs=EPOCHS,
    batch_size=BS
)

# Evaluating the fine-tuned model on the test set
test_loss, test_accuracy = transfer_learning_model.evaluate(x_test_normalized_resized, y_test_encoded, verbose=2)
print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_accuracy}')


model_filename = os.path.join('..', 'models', 'transfer_learning_model2.h5')
transfer_learning_model.save(model_filename)
print(f'Model saved to: {model_filename}')