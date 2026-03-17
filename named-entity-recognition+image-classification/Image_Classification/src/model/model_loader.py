import torch
from torchvision import models
import json

def load_model(model_path, num_classes, device):
    """
    Load a ResNet18 model, adjust the final layer for the given number of classes,
    load the saved state dictionary, and set the model to evaluation mode.
    
    Args:
        model_path (str): Path to the model checkpoint.
        num_classes (int): Number of output classes.
        device (torch.device): Device to load the model onto.
    
    Returns:
        model: The loaded model.
    """
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_class_labels(labels_path):
    """
    Load class labels from a JSON file.
    
    Args:
        labels_path (str): Path to the JSON file containing class labels.
    
    Returns:
        dict: Dictionary mapping class indices to labels.
    """
    with open(labels_path, "r") as f:
        class_labels = json.load(f)
    return class_labels
