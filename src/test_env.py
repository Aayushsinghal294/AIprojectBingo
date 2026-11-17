
import sys
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import cv2

print("Python:", sys.version)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Scikit-learn:", sklearn.__version__)
print("TensorFlow:", tf.__version__)
print("OpenCV:", cv2.__version__)

print("\nAvailable GPU devices:")
print(tf.config.list_physical_devices('GPU'))

print("\nBinGo AI environment setup successful!")
