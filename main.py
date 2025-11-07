### Imports 
import base64
from io import BytesIO
from database_operations import insert_result, fetch_all, fetch_by_model
import keras
from PIL import Image
from flask import Flask, render_template, request, jsonify
import numpy as np
import os

app = Flask(__name__)

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
    canvas_data_url = request.form['imageData']
    
    # Extrahiere den Base64-Teil aus dem Daten-URL
    _, encoded_data = canvas_data_url.split(',', 1)

    # Dekodiere den Base64-Teil in binäre Daten
    binary_data = base64.b64decode(encoded_data)
    
    temp_image_path = BytesIO(binary_data)

    img = Image.open(temp_image_path).convert('LA')  # 'LA' = Graustufen
    
    img = img.resize((28,28))
    image_array = np.array(img)
    
    input_data = convert_data(image_array)
    
    predictions = model.predict(input_data)
    predicted_label = np.argmax(predictions, axis=1)[0]
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
    
    return render_template(
        'index.html',
        predicted_label=int(predicted_label),
        confidence=round(confidence, 4)
    )

if __name__ == "__main__":
    app.run(debug=True)
