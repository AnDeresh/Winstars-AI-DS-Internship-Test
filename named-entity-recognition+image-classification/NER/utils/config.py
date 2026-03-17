import os
from pathlib import Path

# Get the absolute path to the folder where config.py is located
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Data paths
NER_DATASET_JSON = DATA_DIR / "ner_dataset.json"
BIO_DATASET_JSON = DATA_DIR / "ner_bio_dataset.json"
BIO_DATASET_CSV  = DATA_DIR / "ner_bio_dataset_fixed.csv"

# Model paths
MODEL_DIR = BASE_DIR / "model"

# Training parameters
TRAINING_ARGS = {
    "learning_rate": 1e-5,
    "batch_size": 8,
    "num_epochs": 10,
    "weight_decay": 0.05,
    "warmup_steps": 500
}