import pickle
from collections import Counter
from torchvision import datasets
from torch.utils.data import DataLoader, random_split

def load_dataset(dataset_path):
    """
    Load the dataset from the specified path using ImageFolder.
    """
    return datasets.ImageFolder(root=dataset_path)

def get_class_distribution(dataset):
    """
    Return a dictionary with the count of images per class.
    """
    return Counter(label for _, label in dataset.samples)

def split_dataset(dataset, train_ratio=0.8):
    """
    Split the dataset into training and validation sets based on the given ratio.
    Returns a tuple (train_dataset, val_dataset).
    """
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    return random_split(dataset, [train_size, val_size])

def create_dataloaders(train_dataset, val_dataset, batch_size=32):
    """
    Create DataLoaders for both the training and validation sets.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def save_dataset_config(file_path, data_to_save):
    """
    Save the dataset configuration to a pickle file.
    """
    with open(file_path, "wb") as f:
        pickle.dump(data_to_save, f)
