import sys
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add the project root directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

# Local modules
from Image_Classification.src.data.custom_dataset import CustomDataset
from Image_Classification.src.data.transforms_setup import get_train_transform, get_val_transform
from Image_Classification.src.model.model_setup import get_model, get_loss_optimizer
from Image_Classification.src.config import DATASET_CONFIG_PATH, IMAGE_CLASSIFIER_MODEL_PATH

def load_dataset_config(config_path="dataset_config.pkl"):
    """
    Load dataset configuration from a pickle file.

    Returns:
        data (dict): Dictionary with keys 'train_samples', 'val_samples', 'device', 'full_dataset_classes'
    """
    with open(config_path, "rb") as f:
        data = pickle.load(f)
    return data

def create_dataloaders(train_samples, val_samples, train_transform, val_transform, batch_size=32):
    """
    Create training and validation dataloaders.

    Args:
        train_samples (list): List of training (image_path, label) tuples.
        val_samples (list): List of validation (image_path, label) tuples.
        train_transform: Transformations for training data.
        val_transform: Transformations for validation data.
        batch_size (int): Batch size.

    Returns:
        train_loader, val_loader: DataLoader objects for training and validation sets.
    """
    train_dataset = CustomDataset(train_samples, transform=train_transform)
    val_dataset = CustomDataset(val_samples, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def compute_class_weights(train_samples, device):
    """
    Compute class weights using a balanced strategy.

    Args:
        train_samples (list): List of (image_path, label) tuples for training data.
        device (torch.device): Device to move the weights to.

    Returns:
        class_weights (torch.Tensor): Computed class weights.
    """
    labels = [label for _, label in train_samples]
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    return torch.tensor(class_weights, dtype=torch.float).to(device)

def train_model(model, criterion, optimizer, train_loader, val_loader, device, epochs=20):
    """
    Train the model and evaluate on the validation set after each epoch.

    Args:
        model: The neural network.
        criterion: Loss function.
        optimizer: Optimizer.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        device (torch.device): Device (CPU/GPU) for training.
        epochs (int): Number of training epochs.
    """
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / len(train_loader))
        
        # Evaluate on the validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

def main():
    # Load dataset configuration from a pickle file
    data = load_dataset_config(DATASET_CONFIG_PATH)
    train_samples = data["train_samples"]
    val_samples = data["val_samples"]
    device = data["device"]
    num_classes = len(data["full_dataset_classes"])
    
    print(f"Data loaded successfully! Number of classes: {num_classes}")
    
    # Define transforms for training and validation data
    train_transform = get_train_transform()
    val_transform = get_val_transform()
    
    # Create DataLoaders for training and validation sets
    train_loader, val_loader = create_dataloaders(train_samples, val_samples, train_transform, val_transform, batch_size=32)
    
    # Compute class weights to handle imbalance
    class_weights = compute_class_weights(train_samples, device)
    
    # Initialize the model, loss function, and optimizer
    model = get_model(num_classes, device)
    criterion, optimizer = get_loss_optimizer(model, class_weights, lr=0.001)
    
    # Train the model
    train_model(model, criterion, optimizer, train_loader, val_loader, device, epochs=20)
    
    # Save the trained model state
    IMAGE_CLASSIFIER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), IMAGE_CLASSIFIER_MODEL_PATH)
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()
