### Imports 
from database_operations import insert_result, fetch_all, fetch_by_model
import keras
from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

model_path = './models/mnist_model1.keras'

def load_model(model_path: str) -> keras.Model:
    """Load a Keras model from the specified path."""
    return keras.models.load_model(model_path)

def convert_data(image_array):
    """Convert the input image array to the required format."""
    converted_data_list = [[[x[1] / image_array.max()] for x in lst] for lst in image_array]
    converted_data = np.array([converted_data_list])
    
    return converted_data

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
