import os
from torchvision import datasets, transforms, models
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import torch
import argparse

# Define the path to the image folder
image_folder = os.path.join(os.path.dirname(__file__), "data", "raw-img")

# Define transformations for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),      # change size to 224x224
    transforms.ToTensor(),              # convert to tensor (0-255 → 0.0-1.0)
    transforms.Normalize(               # normalize
        mean=[0.485, 0.456, 0.406],     # mean for RGB channels of ImageNet
        std=[0.229, 0.224, 0.225]       # standard deviation for ImageNet
    )
])

# Load the dataset using ImageFolder
dataset = datasets.ImageFolder(root=image_folder, transform=transform)
# Translate class names from Italian to English
italian_to_english = {
    "cane": "dog",
    "cavallo": "horse", 
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "ragno": "spider",
    "scoiattolo": "squirrel"
}
dataset.classes = [italian_to_english[class_name] for class_name in dataset.classes]

# Print some information about the dataset
print("Classes found:", dataset.classes)
print("Number of images:", len(dataset))

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Print the sizes of the training and validation sets
print("Training set size:", len(train_dataset))
print("Validation set size:", len(val_dataset))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train image classification model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output_dir", type=str, default="./models/image_classification_model")
    args = parser.parse_args()

    # Create data loaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Load a pre-trained MobileNetV2 model and modify the classifier for 10 classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, 10)
    model = model.to(device)

    # training setup
    num_epochs = args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # train loop
    for epoch in range(num_epochs):
        model.train() # set model to training mode
        total_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images) 
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # validation loop
        model.eval() # set model to evaluation mode
        total_val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                total_val_loss += criterion(outputs, labels).item()

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {total_val_loss/len(val_loader):.4f}")
    
    # save model
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{args.output_dir}/model.pth")
