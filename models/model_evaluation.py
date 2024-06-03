# model_evaluation.py

from train_baseline_model import main
from tensorflow.keras.utils import to_categorical
from data.preprocess import load_data, normalize_data

# Load and preprocess data
x_train, y_train, x_test, y_test = load_data()
x_train_normalized, x_test_normalized = normalize_data(x_train, x_test)

# Call the function to train the baseline model
main()

# One-hot encode the test labels
y_test_encoded = to_categorical(y_test, num_classes=10)

print(f'Encoded shape: {y_test_encoded.shape}')
print(f'X test normalized shape: {x_test_normalized.shape}')

# Evaluate the model on the test set
test_loss, test_accuracy = baseline_model.evaluate(x_test_normalized, y_test_encoded)

print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')
