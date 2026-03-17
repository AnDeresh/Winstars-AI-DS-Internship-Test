import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

def get_model(num_classes, device):
    """
    Load a pre-trained ResNet18 model and modify the final layer for the given number of classes.

    Args:
        num_classes (int): Number of output classes.
        device (torch.device): Device to move the model to.

    Returns:
        model: The modified model.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model.to(device)

def get_loss_optimizer(model, class_weights, lr=0.001):
    """
    Define the loss function and optimizer.

    Args:
        model: The neural network.
        class_weights: Weights for each class to handle imbalance.
        lr (float): Learning rate.

    Returns:
        criterion: Loss function.
        optimizer: Optimizer.
    """
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return criterion, optimizer
