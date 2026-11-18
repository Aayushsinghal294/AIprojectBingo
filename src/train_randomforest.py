
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

IMAGE_SIZE = (128, 128)


def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    img = np.array(image)

    # HOG features capture shape and texture
    gray = np.array(image.convert("L"))

    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    # HSV colour histogram
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

    X = []
    y = []

    for i, row in df.iterrows():
        try:
            features = extract_features(row["filepath"])
            X.append(features)
            y.append(row["label"])

            if (i + 1) % 200 == 0:
                print(f"Processed {i + 1}/{len(df)} images")

        except Exception as e:
            print(f"Skipping {row['filepath']}: {e}")

    return np.array(X), np.array(y)


def evaluate(model, X, y, split_name):
    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(
        y, predictions, average="macro", zero_division=0
    )
    recall = recall_score(
        y, predictions, average="macro", zero_division=0
    )
    f1 = f1_score(
        y, predictions, average="macro", zero_division=0
    )

    print(f"\n{split_name} Results")
    print("-" * 40)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Macro F1:  {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y, predictions, labels=CLASSES,
        zero_division=0
    ))

    return {
        "split": split_name,
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1
    }


def main():
    print("Loading training features...")
    X_train, y_train = load_features(
        os.path.join(RESULT_DIR, "train.csv")
    )

    print("\nLoading validation features...")
    X_val, y_val = load_features(
        os.path.join(RESULT_DIR, "validation.csv")
    )

    print("\nTraining Random Forest...")
    start = time.time()

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=25,
        min_samples_split=2,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    training_time = time.time() - start
    print(f"\nTraining time: {training_time:.2f} seconds")

    # Validation is used for model assessment.
    # Do not evaluate on the test set yet.
    results = evaluate(model, X_val, y_val, "Validation")

    results["training_time_seconds"] = training_time

    model_path = os.path.join(MODEL_DIR, "random_forest.joblib")
    joblib.dump(model, model_path)

    pd.DataFrame([results]).to_csv(
        os.path.join(RESULT_DIR, "random_forest_validation.csv"),
        index=False
    )

    print(f"\nModel saved to: {model_path}")
    print("Validation results saved.")


if __name__ == "__main__":
    main()
