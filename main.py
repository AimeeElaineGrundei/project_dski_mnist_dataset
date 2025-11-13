### Imports 
import base64
import datetime
from io import BytesIO
import io
from typing import Counter
from database_operations import insert_result, fetch_all, fetch_by_model, create_table
import keras
from PIL import Image, ImageOps
from flask import Flask, render_template, request, jsonify, flash, url_for, redirect
import numpy as np
import os

app = Flask(__name__)
app.secret_key = "supersecretkey"

create_table()

def choose_model_path(model_type: str) -> str:
    """Choose the model path based on the model type."""
    if model_type == "MLP":
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'mnist_model_simple.keras')
    elif model_type == "CNN":
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'mnist_model1.keras')
    elif model_type == "CNN_optimized":
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'mnist_model_optimized.keras')
    
    return model_path

def load_model(model_path: str) -> keras.Model:
    """Load a Keras model from the specified path."""
    return keras.models.load_model(model_path)

model_type = "CNN_optimized"  # Options: "MLP", "CNN", "CNN_optimized"
model_path = choose_model_path(model_type)
model = load_model(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/history")
def history():
    """Display the history of predictions.
    
    Prepares data from the database and renders the history template.
    
    The database returns rows with the following structure:
        (id, timestamp, model_name, input_data, 
        predicted_label, true_label, correct, confidence)
    """
    
    results = fetch_all()

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

    all_results = fetch_all()
    all_models = sorted(list(set([row[2] for row in all_results])))
    
    if selected_model == 'all' and len(all_models) == 1:
        selected_model = all_models[0]

    digits = [str(i) for i in range(10)]
    dict_digits = {d: 0 for d in digits}
    
    if selected_model == 'all':
        predictions_per_model = {model: dict_digits.copy() for model in all_models}
        correct_counts_per_model = {model: dict_digits.copy() for model in all_models}
        total_counts_per_model = {model: dict_digits.copy() for model in all_models}

        for row in all_results:
            model_name = row[2]
            pred = str(row[4])
            true = str(row[5])
            correct = row[6]

            predictions_per_model[model_name][pred] += 1
            total_counts_per_model[model_name][true] += 1
            if correct:
                correct_counts_per_model[model_name][true] += 1

        accuracy_per_model = {}
        for model in all_models:
            accuracy_per_model[model] = {}
            for d in digits:
                if total_counts_per_model[model][d] > 0:
                    accuracy_per_model[model][d] = round(
                        correct_counts_per_model[model][d] / total_counts_per_model[model][d] * 100, 2)
                else:
                    accuracy_per_model[model][d] = 0

        total_predictions = len(all_results)
        correct_predictions = sum([row[6] for row in all_results])
        accuracy = round((correct_predictions / total_predictions * 100) if total_predictions else 0, 2)

        return render_template(
            "statistics.html",
            selected_model='all',
            models=all_models,
            total_predictions=total_predictions,
            correct_predictions=correct_predictions,
            accuracy=accuracy,
            predictions_per_model=predictions_per_model,
            accuracy_per_model=accuracy_per_model
        )

    else:
        results = fetch_by_model(selected_model)
        total_predictions = len(results)
        correct_predictions = sum([row[6] for row in results])
        accuracy = round((correct_predictions / total_predictions * 100) if total_predictions else 0, 2)

        predictions_counter = dict_digits.copy()
        correct_counts = dict_digits.copy()
        total_counts = dict_digits.copy()

        for row in results:
            pred = str(row[4])
            true = str(row[5])
            correct = row[6]

            predictions_counter[pred] += 1
            total_counts[true] += 1
            if correct:
                correct_counts[true] += 1

        accuracy_per_digit = {}
        for d in digits:
            if total_counts[d] > 0:
                accuracy_per_digit[d] = round(correct_counts[d] / total_counts[d] * 100, 2)
            else:
                accuracy_per_digit[d] = 0

        return render_template(
            "statistics.html",
            selected_model=selected_model,
            models=all_models,
            total_predictions=total_predictions,
            correct_predictions=correct_predictions,
            accuracy=accuracy,
            predictions_per_digit=predictions_counter,
            accuracy_per_digit=accuracy_per_digit
        )


@app.route('/predict', methods=['POST'])
def predict():
    # get data from the canvas (base64 string)
    image_data = request.form['image']
    
    image_data = image_data.split(",")[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert("L")  # grayscale
    mean_val = np.mean(np.array(image))
    if mean_val > 127:
        image = ImageOps.invert(image)
        
    img_array = np.array(image)
    img_array = np.where(img_array > 20, img_array, 0)  # delete noise

    coords = np.column_stack(np.where(img_array > 0))   # coordinates of non-zero pixels
    if coords.size != 0:  # only crop if there are non-zero pixels
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        image = image.crop((x_min, y_min, x_max, y_max))  # crop to bounding box

        # Add square padding
        w, h = image.size
        side = max(w, h)
        square = Image.new("L", (side, side), 0)
        square.paste(image, ((side - w) // 2, (side - h) // 2))
        image = square
        
    image = ImageOps.fit(image, (28, 28), Image.LANCZOS, centering=(0.5, 0.5))
    # to numpy array and convert (canvas background=0, digit=255)
    img_array = np.array(image).astype("float32") / 255.0 # normalize to [0, 1]

    if model_type == "CNN" or model_type == "CNN_optimized":
        # reshape to (1, 28, 28, 1)
        img_array = img_array.reshape(1, 28, 28, 1)
        
    elif model_type == "MLP":
        # reshape to (1, 784)
        img_array = img_array.reshape(1, 28 * 28)
    
    predictions = model.predict(img_array)
    predicted_label = int(np.argmax(predictions))
    confidence = float(np.max(predictions))
    
    return jsonify({
        "prediction": int(predicted_label),
        "confidence": float(confidence),
        "image_array": img_array.tolist()
    })

@app.route("/feedback", methods=["POST"])
def feedback():
    image_data = request.form["image_array"]
    predicted_label = request.form["predicted_label"]
    true_label = request.form["true_label"]
    confidence = request.form.get("confidence", None)

    correct = int(predicted_label == true_label)

    save_dir = os.path.join("pictures_drawn", str(true_label))
    os.makedirs(save_dir, exist_ok=True)
    
    img_lst = image_data.split(",")
    img_array = np.array(img_lst, dtype=float)
    img = Image.fromarray((img_array * 255).astype(np.uint8).reshape(28, 28), mode='L')

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{model_type}_pred{predicted_label}_conf{confidence}.png"
    filepath = os.path.join(save_dir, filename)

    img.save(filepath)
    
    insert_result(
        model_name=model_type,
        input_data=img_array,
        predicted_label=predicted_label,
        true_label=true_label,
        correct=correct,
        confidence=confidence
    )

    flash("Thank You for your feedback! The data has been saved.", "success")
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
