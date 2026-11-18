
import os
import pandas as pd

from sklearn.model_selection import train_test_split

from config import RESULT_DIR, CLASSES, SEED

# Load audited dataset
report_path = os.path.join(RESULT_DIR, "dataset_report.csv")
df = pd.read_csv(report_path)

# Keep only required columns
df = df[["path", "class"]].copy()
df.rename(columns={
    "path": "filepath",
    "class": "label"
}, inplace=True)

# Verify class labels
df = df[df["label"].isin(CLASSES)]

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

# Save all splits
train_df.to_csv(
    os.path.join(RESULT_DIR, "train.csv"), index=False
)

val_df.to_csv(
    os.path.join(RESULT_DIR, "validation.csv"), index=False
)

test_df.to_csv(
    os.path.join(RESULT_DIR, "test.csv"), index=False
)

print("========== DATASET SPLIT ==========")

print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Testing images:", len(test_df))
print("Total images:", len(df))

print("\nTraining distribution:")
print(train_df["label"].value_counts().reindex(CLASSES))

print("\nValidation distribution:")
print(val_df["label"].value_counts().reindex(CLASSES))

print("\nTesting distribution:")
print(test_df["label"].value_counts().reindex(CLASSES))

print("\nDataset splitting completed successfully!")
