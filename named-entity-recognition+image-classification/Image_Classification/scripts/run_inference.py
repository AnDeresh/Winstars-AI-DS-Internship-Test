import os, sys
import torch
from pathlib import Path

# Add the project root directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from Image_Classification.src.config import (
    IMAGE_CLASSIFIER_MODEL_PATH,
    CLASS_LABELS_PATH,
    DEMO_IMG_DIR,
    NUM_CLASSES,
    DEVICE,
)
from Image_Classification.src.model.model_loader import load_model, load_class_labels
from Image_Classification.src.model.inference_utils import get_inference_transform, predict_image
from Image_Classification.src.data.visualization import display_predictions

def main():
    # Load the model and class labels from configuration
    model = load_model(IMAGE_CLASSIFIER_MODEL_PATH, NUM_CLASSES, DEVICE)
    class_labels = load_class_labels(CLASS_LABELS_PATH)

    # Get the transformation for inference
    transform = get_inference_transform()

    # Retrieve test images (filter common image extensions)
    test_image_dir = DEMO_IMG_DIR
    if not os.path.exists(test_image_dir):
        os.makedirs(test_image_dir)
        print(f"Please add test images to the folder '{test_image_dir}' and rerun the script.")
        exit()

    test_images = [
        os.path.join(test_image_dir, img)
        for img in os.listdir(test_image_dir)
        if img.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    if len(test_images) == 0:
        raise FileNotFoundError(f"No test images found in '{test_image_dir}'. Please add at least one test image.")

    # Define a prediction function
    predict_fn = lambda img_path: predict_image(model, transform, class_labels, img_path, DEVICE)

    # Display predictions
    display_predictions(test_images, predict_fn, num_images=10)

if __name__ == "__main__":
    main()