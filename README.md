# Image-Classification
This project aims to classify images from the CIFAR-10 dataset into 10 categories: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, and Truck. The repository encompasses the complete workflow, from environment setup and exploratory data analysis to model deployment via a web application.

## Project Structure

The project has the following structure:

Image-Classification/
│
├── data/
│ └── test_file/ # Directory for test images
│
├── models/
│ └── transfer_learning_model.h5 # Pre-trained model
│
├── notebooks/
│ └── model_building.ipynb # Jupyter notebooks for model building and evaluation
│
├── app/
│ ├── app.py # Flask application code
│ └── templates/
│ └── index.html # HTML template for the web interface
│
├── reports/
│ └── project_report.pdf # Project documentation and reports
│
└── requirements.txt # List of dependencies and packages



## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:

```sh
git clone https://github.com/your-username/Image-Classification.git
cd Image-Classification

2. Create a virtual environment and activate it:

python -m venv venv
# On Windows
venv\Scripts\activate

3. Install the required packages:

pip install -r requirements.txt

4. Ensure the pre-trained model is in the models directory:

# models/
# └── transfer_learning_model.h5


#Usage

1. Run the Flask application:
--> cd app
--> python app.py

2. Open your web browser and go to http://127.0.0.1:5000/.
3. Upload an image (JPG or PNG format) and click "Classify" to see the predicted class label.



NB: Model Training
--> If you wish to train the model from scratch or further fine-tune it, refer to the Jupyter notebooks in the notebooks directory. The notebooks demonstrate the process of building and evaluating the model.

##License
None

Acknowledgements
--> This project uses the InceptionV3 model from TensorFlow Keras.



