
import os
import pandas as pd

from sklearn.model_selection import train_test_split

from config import DATASET_DIR, RESULT_DIR, CLASSES, SEED

# Collect image paths and labels
records = []

for class_name in CLASSES:

    class_dir = os.path.join(DATASET_DIR, class_name)

    for filename in os.listdir(class_dir):

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        ):

            records.append({
                "filepath": os.path.join(class_dir, filename),
                "label": class_name
            })

df = pd.DataFrame(records)

# First split: 70% training, 30% remaining
train_df, remaining_df = train_test_split(
    df,
    test_size=0.30,
    random_state=SEED,
    stratify=df["label"]
)

# Second split: 15% validation, 15% testing
val_df, test_df = train_test_split(
    remaining_df,
    test_size=0.50,
    random_state=SEED,
    stratify=remaining_df["label"]
)

# Save splits
train_df.to_csv(
    os.path.join(RESULT_DIR, "train.csv"), index=False
)

val_df.to_csv(
    os.path.join(RESULT_DIR, "validation.csv"), index=False
)

test_df.to_csv(
    os.path.join(RESULT_DIR, "test.csv"), index=False
)

print("Dataset split completed!")

print("\nTraining:", len(train_df))
print("Validation:", len(val_df))
print("Testing:", len(test_df))

print("\nTraining distribution:")
print(train_df["label"].value_counts())

print("\nValidation distribution:")
print(val_df["label"].value_counts())

print("\nTesting distribution:")
print(test_df["label"].value_counts())

print("\nAll CSV files saved successfully!")
