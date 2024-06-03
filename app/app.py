from flask import Flask, render_template, request, redirect, url_for
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
import os
import io

# model path
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'transfer_learning_model.h5')

# Loading the pre-trained model
model = load_model(model_path)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            # Convert the FileStorage object to a file-like object
            image_stream = io.BytesIO(image_file.read())
            
            # Preprocess the image (resize, normalize) based on your model's requirements
            img = image.load_img(image_stream, target_size=(128, 128))  # Adjust based on your model
            img = image.img_to_array(img)
            img = preprocess_input(img)  # Assuming InceptionV3, adjust preprocessing if needed
            img = np.expand_dims(img, axis=0)  # Add a dimension for batch processing

            # Prediction
            prediction = model.predict(img)
            predicted_class = np.argmax(prediction)  # Get the index of the most likely class
            class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

            return render_template('index.html', prediction=class_labels[predicted_class])
        else:
            return redirect(url_for('index'))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
