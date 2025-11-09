### Imports 
import base64
from io import BytesIO
import io
from typing import Counter
from database_operations import insert_result, fetch_all, fetch_by_model, create_table
import keras
from PIL import Image, ImageOps
from flask import Flask, render_template, request, jsonify, flash, url_for, redirect
import numpy as np
import os
import re

app = Flask(__name__)
app.secret_key = "supersecretkey"

create_table()

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

@app.route("/history")
def history():
    # get all results from the database
    results = fetch_all()

    # prepare results for the template
    # The DB returns: (id, timestamp, model_name, input_data, predicted_label, true_label, correct, confidence)
    history_list = []
    
    for row in results:
        entry = {
            "model_name": row[2],  # model_name
            "predicted": row[4],  # predicted_label
            "actual": row[5],     # true_label
            "correct": row[6],    # correct (1/0)
            "confidence": row[7]  # optional
        }
        history_list.append(entry)

    return render_template("history.html", history=history_list)


@app.route("/statistics")
def statistics():
    selected_model = request.args.get('model', 'all')
    results = fetch_all() if selected_model == 'all' else fetch_by_model(selected_model)

    total_predictions = len(results)
    correct_predictions = sum([row[6] for row in results])

    accuracy = round((correct_predictions / total_predictions * 100) if total_predictions else 0, 2)

    predictions_counter = Counter()
    accuracy_counter = {str(i): 0 for i in range(10)}
    count_per_digit = {str(i): 0 for i in range(10)}

    for row in results:
        pred = str(row[4])  # predicted_label
        true = str(row[5])  # true_label
        correct = row[6]

        predictions_counter[pred] += 1
        count_per_digit[true] += 1
        if correct:
            accuracy_counter[true] += 100

    # calculate accuracy per digit (if count=0 -> 0%)
    accuracy_per_digit = {}
    for digit in range(10):
        digit_str = str(digit)
        if count_per_digit[digit_str] > 0:
            accuracy_per_digit[digit_str] = round(accuracy_counter[digit_str] / count_per_digit[digit_str], 2)
        else:
            accuracy_per_digit[digit_str] = 0

    all_models = [row[2] for row in fetch_all()]
    models = sorted(list(set(all_models)))
    
    return render_template(
        "statistics.html",
        total_predictions=total_predictions,
        correct_predictions=correct_predictions,
        accuracy=accuracy,
        predictions_per_digit=predictions_counter,
        accuracy_per_digit=accuracy_per_digit,
        models=models,
        selected_model=selected_model
    )

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
    
    predictions = model.predict(img_array)
    predicted_label = int(np.argmax(predictions))
    confidence = float(np.max(predictions))
    
    return jsonify({
        "prediction": int(predicted_label),
        "confidence": float(confidence)
    })

@app.route("/feedback", methods=["POST"])
def feedback():
    image_data = request.form["image"]
    predicted_label = request.form["predicted_label"]
    true_label = request.form["true_label"]
    confidence = request.form.get("confidence", None)

    correct = int(predicted_label == true_label)

    insert_result(
        model_name=model_type,
        input_data=image_data,
        predicted_label=predicted_label,
        true_label=true_label,
        correct=correct,
        confidence=confidence
    )

    flash("Thank You for your feedback! The data has been saved.", "success")
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
