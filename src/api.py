import os
import sys

import pandas as pd
import torch
from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CLASSES, RESULT_DIR
from predict import DEVICE, load_trained_model, transform

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024
_model = None


def get_model():
    global _model
    if _model is None:
        _model = load_trained_model()
    return _model


def read_metrics():
    comparison_path = os.path.join(RESULT_DIR, "model_comparison.csv")
    class_report_path = os.path.join(RESULT_DIR, "best_model_class_report.csv")
    validation_path = os.path.join(RESULT_DIR, "validation.csv")
    required_paths = [comparison_path, class_report_path, validation_path]
    missing = [path for path in required_paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "Run `python src/evaluate_models.py` to generate the model evaluation data."
        )

    comparison = pd.read_csv(comparison_path).fillna(0)
    class_report = pd.read_csv(class_report_path).fillna(0)
    validation = pd.read_csv(validation_path)
    models = [
        {
            "model": str(row["model"]),
            "accuracy": float(row["accuracy"]),
            "macro_precision": float(row["macro_precision"]),
            "macro_recall": float(row["macro_recall"]),
            "macro_f1": float(row["macro_f1"]),
        }
        for _, row in comparison.iterrows()
    ]
    class_metrics = [
        {
            "class_name": str(row["class"]),
            "f1_score": float(row["f1-score"]),
            "support": int(row["support"]),
        }
        for _, row in class_report.iterrows()
    ]
    class_counts = validation["label"].value_counts().reindex(CLASSES, fill_value=0)
    return {
        "models": models,
        "classes": CLASSES,
        "class_metrics": class_metrics,
        "validation_images": int(len(validation)),
        "class_counts": [
            {"class_name": name, "count": int(class_counts[name])}
            for name in CLASSES
        ],
    }


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/metrics")
def metrics():
    try:
        return jsonify(read_metrics())
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503


@app.post("/api/predict")
def predict():
    uploaded_file = request.files.get("image")
    if uploaded_file is None:
        return jsonify({"error": "Choose an image to classify."}), 400

    try:
        image = Image.open(uploaded_file.stream).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "The uploaded file is not a readable image."}), 400

    try:
        model = get_model()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        probabilities = torch.softmax(model(input_tensor)[0], dim=0).cpu().tolist()

    scores = [
        {"class_name": class_name, "probability": float(probability)}
        for class_name, probability in zip(CLASSES, probabilities)
    ]
    best_score = max(scores, key=lambda score: score["probability"])
    return jsonify(
        {
            "predicted_class": best_score["class_name"],
            "confidence": best_score["probability"],
            "scores": scores,
            "filename": uploaded_file.filename or "image",
        }
    )


@app.errorhandler(RequestEntityTooLarge)
def request_too_large(_error):
    return jsonify({"error": "Image is too large. Maximum upload size is 12 MB."}), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
