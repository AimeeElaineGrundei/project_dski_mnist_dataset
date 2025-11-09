### Imports 
import base64
from io import BytesIO
import io
from database_operations import insert_result, fetch_all, fetch_by_model
import keras
from PIL import Image, ImageOps
from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import re

app = Flask(__name__)

model_type = "MLP"

if model_type == "MLP":
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'mnist_model_simple.keras')
elif model_type == "CNN":
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'mnist_model1.keras')

def load_model(model_path: str) -> keras.Model:
    """Load a Keras model from the specified path."""
    return keras.models.load_model(model_path)

model = load_model(model_path)

def convert_data(image_array):
    """Convert the input image array to the required format."""
    converted_data_list = [[[x[1] / image_array.max()] for x in lst] for lst in image_array]
    converted_data = np.array([converted_data_list])
    
    return converted_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/history')
def history():
    # results = fetch_all()
    return render_template('history.html')

@app.route('/statistics')
def statistics():
    # results = fetch_all()
    return render_template('statistics.html')

@app.route('/predict', methods=['POST'])
def predict():
    # get data from the canvas (base64 string)
    image_data = request.form['image']
    
    image_data = image_data.split(",")[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert("L")  # grayscale
    image = ImageOps.invert(image)  # invert black/white
    
    # shrink image to 28x28
    image = image.resize((28, 28))
    
    # cut picture and center the digit
    image = ImageOps.fit(image, (28, 28), centering=(0.5, 0.5))

    # to numpy array and convert (canvas background=0, digit=255)
    img_array = np.array(image).astype("float32") / 255.0 # normalize to [0, 1]

    if model_type == "CNN":
        # reshape to (1, 28, 28, 1)
        img_array = img_array.reshape(1, 28, 28, 1)
    elif model_type == "MLP":
        # reshape to (1, 28, 28, 1)
        img_array = img_array.reshape(1, 28 * 28)

    image.save("debug_input.png")
    
    predictions = model.predict(img_array)
    predicted_label = int(np.argmax(predictions))
    confidence = float(np.max(predictions))
    
    # true_label = data.get('true_label', None)
    # correct = (predicted_label == true_label) if true_label is not None else None
    
    # insert_result(
    #     model_name='mnist_model1',
    #     input_data=image_array.tolist(),
    #     predicted_label=predicted_label,
    #     true_label=true_label,
    #     correct=correct,
    #     confidence=confidence
    # )
    
    return jsonify({"prediction": predicted_label, "confidence": confidence})

if __name__ == "__main__":
    app.run(debug=True)
