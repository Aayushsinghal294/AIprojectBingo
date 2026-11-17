
import os
import hashlib
from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, UnidentifiedImageError

from config import DATASET_DIR, CLASSES, RESULT_DIR

# Supported image formats
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def get_hash(image_path):
    """Generate SHA-256 hash for exact duplicate detection."""
    sha = hashlib.sha256()

    with open(image_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha.update(chunk)

    return sha.hexdigest()


def analyze_dataset():

    root = Path(DATASET_DIR)

    if not root.exists():
        print("Dataset folder not found:", root)
        return

    records = []
    corrupted = []
    duplicate_groups = defaultdict(list)

    print("\n========== DATASET AUDIT ==========\n")

    for class_name in CLASSES:

        class_dir = root / class_name

        if not class_dir.exists():
            print("Missing class:", class_name)
            continue

        files = [
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        ]

        valid_count = 0

        for image_path in files:

            try:
                with Image.open(image_path) as img:
                    img.verify()

                # Verify that image can actually be decoded
                with Image.open(image_path) as img:
                    width, height = img.size

                image_hash = get_hash(image_path)

                duplicate_groups[image_hash].append(
                    str(image_path)
                )

                records.append({
                    "filename": image_path.name,
                    "path": str(image_path),
                    "class": class_name,
                    "width": width,
                    "height": height
                })

                valid_count += 1

            except (OSError, ValueError, UnidentifiedImageError):
                corrupted.append(str(image_path))

        print(f"{class_name:12} : {valid_count} valid images")

    df = pd.DataFrame(records)

    if df.empty:
        print("\nNo valid images found.")
        return

    duplicates = {
        key: paths
        for key, paths in duplicate_groups.items()
        if len(paths) > 1
    }

    print("\n========== SUMMARY ==========")
    print("Total valid images:", len(df))
    print("Corrupted images:", len(corrupted))
    print("Exact duplicate groups:", len(duplicates))

    print("\nClass distribution:")
    print(df["class"].value_counts().to_string())

    # Save dataset report
    df.to_csv(
        os.path.join(RESULT_DIR, "dataset_report.csv"),
        index=False
    )

    pd.Series(corrupted).to_csv(
        os.path.join(RESULT_DIR, "corrupted_images.csv"),
        index=False
    )

    # Class distribution chart
    counts = df["class"].value_counts().reindex(
        CLASSES, fill_value=0
    )

    plt.figure(figsize=(10, 5))
    counts.plot(kind="bar")

    plt.title("TrashNet Dataset Class Distribution")
    plt.xlabel("Waste Category")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(
        os.path.join(RESULT_DIR, "class_distribution.png"),
        dpi=300
    )

    plt.show()

    # Display sample images
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for i, class_name in enumerate(CLASSES):

        sample = df[df["class"] == class_name]

        if len(sample) == 0:
            continue

        image_path = sample.iloc[0]["path"]

        with Image.open(image_path) as img:
            axes[i // 3, i % 3].imshow(img.convert("RGB"))

        axes[i // 3, i % 3].set_title(class_name)
        axes[i // 3, i % 3].axis("off")

    plt.tight_layout()

    plt.savefig(
        os.path.join(RESULT_DIR, "sample_images.png"),
        dpi=300
    )

    plt.show()

    print("\nDataset analysis completed!")


if __name__ == "__main__":
    analyze_dataset()
