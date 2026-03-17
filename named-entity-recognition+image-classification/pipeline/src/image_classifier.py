import sys
from pathlib import Path
import torch
import json
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from pipeline.src.config import (
    IMAGE_CLASSIFIER_MODEL_PATH,
    CLASS_LABELS_PATH,
    NUM_CLASSES,
    DEVICE
)

def load_image_model():
    """
    Load the image classification model and adjust the final layer.
    
    Returns:
        model: The loaded image classification model.
    """
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, NUM_CLASSES)
    model.load_state_dict(torch.load(str(IMAGE_CLASSIFIER_MODEL_PATH), map_location="cpu"))
    model.to(DEVICE)
    model.eval()
    return model

def get_transform_pipeline():
    """
    Define the image transformation pipeline for inference.
    
    Returns:
        transform: Composed transformation.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

def load_class_labels() -> dict:
    """
    Load class labels from a JSON file.
    
    Returns:
        dict: Dictionary mapping class indices (as strings) to labels.
    """
    with open(str(CLASS_LABELS_PATH), "r", encoding="utf-8") as f:
        return json.load(f)

def predict_image(model, transform, image_path: str, class_labels: dict) -> tuple:
    """
    Predict the class of the given image.
    
    Args:
        model: The image classification model.
        transform: Transformation pipeline.
        image_path (str): Path to the image.
        class_labels (dict): Mapping of class indices to labels.
    
    Returns:
        tuple: (Predicted class label, PIL Image object)
    """
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(image_tensor.to(DEVICE))
        predicted_class = torch.argmax(output, dim=1).item()
    
    return class_labels[str(predicted_class)], image
