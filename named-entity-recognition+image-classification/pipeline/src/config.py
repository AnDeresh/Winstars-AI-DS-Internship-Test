import torch
from pathlib import Path

# Base directory is assumed to be the parent of the pipeline folder
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Evaluation dataset configuration
EVALUATION_DATASET_DIR = BASE_DIR / "Pipeline" / "evaluation_dataset"
EVALUATION_IMAGES_DIR = EVALUATION_DATASET_DIR / "evaluation_images"
EVALUATION_JSON_PATH = EVALUATION_DATASET_DIR / "evaluation_data.json"

# NER model configuration
NER_MODEL_DIR = BASE_DIR / "NER" / "model"

# Image classifier configuration
IMAGE_CLASSIFIER_MODEL_PATH = BASE_DIR / "Image_Classification" / "model" / "image_classifier_model.pth"
CLASS_LABELS_PATH = BASE_DIR / "Image_Classification" / "src" / "data" / "class_labels.json"

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Number of classes for image classification
NUM_CLASSES = 10

# DataLoader parameters
BATCH_SIZE = 32
