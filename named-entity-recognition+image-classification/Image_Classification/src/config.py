import os
import torch
from pathlib import Path

# Define the base directory (the root of your project)
BASE_DIR = Path(__file__).resolve().parent.parent

# Select device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dataset paths using BASE_DIR
RAW_DATASET_PATH = BASE_DIR / "data" / "raw-img"
PROCESSED_DATASET_PATH = BASE_DIR / "data" / "processed-img"
AUGMENTED_DATASET_PATH = BASE_DIR / "data" / "augmented-img"

# Ensure the necessary directories exist
os.makedirs(PROCESSED_DATASET_PATH, exist_ok=True)
os.makedirs(AUGMENTED_DATASET_PATH, exist_ok=True)

# Path to the dataset_config.pkl file
DATASET_CONFIG_PATH = BASE_DIR / "src" / "data" / "dataset_config.pkl"

# Absolute path to the image classifier model save file
IMAGE_CLASSIFIER_MODEL_PATH = BASE_DIR / "model" / "image_classifier_model.pth"

# Path to the class labels JSON file
CLASS_LABELS_PATH = BASE_DIR / "src" / "data" / "class_labels.json"

# Directory containing demo images for inference
DEMO_IMG_DIR = BASE_DIR / "data" / "demo_img"

# Image and DataLoader parameters
TARGET_SIZE = (300, 300)  # Target image size
BATCH_SIZE = 32

# Additional configuration
NUM_CLASSES = 10  # Adjust based on your dataset
DEVICE = device
