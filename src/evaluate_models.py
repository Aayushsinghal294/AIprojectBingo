import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torchvision import models, transforms

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import CLASSES, MODEL_DIR, RESULT_DIR
from train_cnn import WasteCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_predictions(model, image_paths, transform):
    model.to(DEVICE).eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(image_paths), 32):
            batch = torch.stack(
                [
                    transform(Image.open(path).convert("RGB"))
                    for path in image_paths[start:start + 32]
                ]
            ).to(DEVICE)
            predictions.extend(model(batch).argmax(dim=1).cpu().numpy())
    return np.asarray(CLASSES)[np.asarray(predictions)]


def load_transfer_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, len(CLASSES)),
    )
    state = torch.load(
        os.path.join(MODEL_DIR, "best_transfer_cnn.pth"),
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(state)
    return model


def main():
    validation = pd.read_csv(os.path.join(RESULT_DIR, "validation.csv"))
    labels = validation["label"].to_numpy()
    image_paths = validation["filepath"].to_numpy()

    custom_transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )
    transfer_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )

    models_to_evaluate = [
        (
            "Custom CNN",
            WasteCNN(),
            os.path.join(MODEL_DIR, "best_custom_cnn.pth"),
            custom_transform,
        ),
        (
            "ResNet-18 transfer CNN",
            load_transfer_model(),
            None,
            transfer_transform,
        ),
    ]

    comparison = []
    predictions_by_model = {}
    for name, model, model_path, image_transform in models_to_evaluate:
        if model_path:
            state = torch.load(
                model_path,
                map_location=DEVICE,
                weights_only=True,
            )
            model.load_state_dict(state)
        predictions = load_predictions(model, image_paths, image_transform)
        predictions_by_model[name] = predictions
        comparison.append(
            {
                "model": name,
                "accuracy": accuracy_score(labels, predictions),
                "macro_precision": precision_score(
                    labels, predictions, labels=CLASSES,
                    average="macro", zero_division=0,
                ),
                "macro_recall": recall_score(
                    labels, predictions, labels=CLASSES,
                    average="macro", zero_division=0,
                ),
                "macro_f1": f1_score(
                    labels, predictions, labels=CLASSES,
                    average="macro", zero_division=0,
                ),
            }
        )

    feature_cache_path = os.path.join(RESULT_DIR, "rf_features.npz")
    with np.load(feature_cache_path) as feature_cache:
        forest_labels = feature_cache["y_val"].astype(str)
        forest_features = feature_cache["X_val"]
    if not np.array_equal(forest_labels, labels.astype(str)):
        raise ValueError(
            "Random forest feature cache labels do not match validation.csv."
        )

    for name, filename in [
        ("Random Forest (baseline)", "random_forest.joblib"),
        ("Random Forest (tuned)", "random_forest_tuned.joblib"),
    ]:
        forest = joblib.load(os.path.join(MODEL_DIR, filename))
        predictions = forest.predict(forest_features)
        predictions_by_model[name] = predictions
        comparison.append(
            {
                "model": name,
                "accuracy": accuracy_score(labels, predictions),
                "macro_precision": precision_score(
                    labels, predictions, labels=CLASSES,
                    average="macro", zero_division=0,
                ),
                "macro_recall": recall_score(
                    labels, predictions, labels=CLASSES,
                    average="macro", zero_division=0,
                ),
                "macro_f1": f1_score(
                    labels, predictions, labels=CLASSES,
                    average="macro", zero_division=0,
                ),
            }
        )

    comparison_df = pd.DataFrame(comparison).sort_values(
        "macro_f1", ascending=False
    )
    comparison_df.to_csv(
        os.path.join(RESULT_DIR, "model_comparison.csv"), index=False
    )

    best_model = comparison_df.iloc[0]["model"]
    report = pd.DataFrame(
        classification_report(
            labels,
            predictions_by_model[best_model],
            labels=CLASSES,
            output_dict=True,
            zero_division=0,
        )
    ).T.loc[CLASSES, ["precision", "recall", "f1-score", "support"]]
    report.index.name = "class"
    report.to_csv(os.path.join(RESULT_DIR, "best_model_class_report.csv"))

    print(f"Compared {len(comparison_df)} models on {len(validation)} images.")
    print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nBest model by validation macro-F1: {best_model}")


if __name__ == "__main__":
    main()
