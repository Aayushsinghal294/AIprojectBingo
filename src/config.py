
import os
import random
import numpy as np
import tensorflow as tf

# Reproducibility
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(
    BASE_DIR, "dataset", "raw", "dataset-resized"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR, "dataset", "processed"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")

# Six classification categories
CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

NUM_CLASSES = len(CLASSES)

# Model configurations
RF_IMAGE_SIZE = (128, 128)
CNN_IMAGE_SIZE = (96, 96)
MOBILENET_IMAGE_SIZE = (160, 160)

CNN_BATCH_SIZE = 16
MOBILENET_BATCH_SIZE = 8

CNN_EPOCHS = 20
MOBILENET_EPOCHS = 12

# Dataset split
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# CPU configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)

# Create required directories
for folder in [PROCESSED_DIR, MODEL_DIR, RESULT_DIR]:
    os.makedirs(folder, exist_ok=True)
