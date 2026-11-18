
import os
import sys
import time
import joblib
import numpy as np
import pandas as pd

from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2hsv

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import CLASSES, MODEL_DIR, RESULT_DIR, SEED

CACHE_PATH = os.path.join(RESULT_DIR, "rf_features.npz")
IMAGE_SIZE = (128, 128)


def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    img = np.array(image)
    gray = np.array(image.convert("L"))

    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    hsv = rgb2hsv(img)

    hist_features = []
    for channel in range(3):
        hist, _ = np.histogram(
            hsv[:, :, channel],
            bins=8,
            range=(0, 1),
            density=True
        )
        hist_features.extend(hist)

    return np.concatenate([
        hog_features,
        np.array(hist_features)
    ]).astype(np.float32)


def load_features(csv_path):
    df = pd.read_csv(csv_path)

    X, y = [], []

    for i, row in df.iterrows():
        X.append(extract_features(row["filepath"]))
        y.append(row["label"])

        if (i + 1) % 200 == 0:
            print(f"Processed {i + 1}/{len(df)}")

    return np.array(X), np.array(y)


def get_data():
    if os.path.exists(CACHE_PATH):
        print("Loading cached features...")
        data = np.load(CACHE_PATH)
        return (
            data["X_train"], data["y_train"],
            data["X_val"], data["y_val"]
        )

    print("Extracting training features...")
    X_train, y_train = load_features(
        os.path.join(RESULT_DIR, "train.csv")
    )

    print("Extracting validation features...")
    X_val, y_val = load_features(
        os.path.join(RESULT_DIR, "validation.csv")
    )

    np.savez_compressed(
        CACHE_PATH,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val
    )

    return X_train, y_train, X_val, y_val


def main():
    X_train, y_train, X_val, y_val = get_data()

    configurations = [
        {
            "name": "RF_Baseline",
            "n_estimators": 200,
            "max_depth": 25,
            "max_features": "sqrt",
            "min_samples_leaf": 1
        },
        {
            "name": "RF_Deep",
            "n_estimators": 300,
            "max_depth": None,
            "max_features": "sqrt",
            "min_samples_leaf": 1
        },
        {
            "name": "RF_Regularized",
            "n_estimators": 300,
            "max_depth": 30,
            "max_features": 0.5,
            "min_samples_leaf": 2
        }
    ]

    all_results = []
    best_model = None
    best_f1 = -1
    best_name = ""

    for config in configurations:
        print(f"\nTraining {config['name']}...")

        model = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            max_features=config["max_features"],
            min_samples_leaf=config["min_samples_leaf"],
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1
        )

        start = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start

        predictions = model.predict(X_val)

        accuracy = accuracy_score(y_val, predictions)
        precision = precision_score(
            y_val, predictions,
            average="macro", zero_division=0
        )
        recall = recall_score(
            y_val, predictions,
            average="macro", zero_division=0
        )
        macro_f1 = f1_score(
            y_val, predictions,
            average="macro", zero_division=0
        )

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Training time: {training_time:.2f}s")

        all_results.append({
            "configuration": config["name"],
            "accuracy": accuracy,
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": macro_f1,
            "training_time_seconds": training_time
        })

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_model = model
            best_name = config["name"]

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(
        os.path.join(RESULT_DIR, "rf_tuning_results.csv"),
        index=False
    )

    model_path = os.path.join(
        MODEL_DIR, "random_forest_tuned.joblib"
    )
    joblib.dump(best_model, model_path)

    print("\nBest configuration:", best_name)
    print(f"Best validation Macro F1: {best_f1:.4f}")

    print("\nBest model classification report:")
    print(classification_report(
        y_val,
        best_model.predict(X_val),
        labels=CLASSES,
        zero_division=0
    ))

    print("\nSaved tuning results and best model.")


if __name__ == "__main__":
    main()
