from torchvision import models, transforms
import torch
import torch.nn as nn
from PIL import Image
import argparse
import os

# Define class names corresponding to the dataset
class_names = ["dog", "horse", "elephant", "butterfly", "chicken", "cat", "cow", "sheep", "spider", "squirrel"]

# transorm
transform = transforms.Compose([
    transforms.Resize((224, 224)),      # change size to 224x224
    transforms.ToTensor(),              # convert to tensor (0-255 → 0.0-1.0)
    transforms.Normalize(               # normalize
        mean=[0.485, 0.456, 0.406],     # mean for RGB channels of ImageNet
        std=[0.229, 0.224, 0.225]       # standard deviation for ImageNet
    )
])

# load model and modify classifier for 10 classes
model = models.mobilenet_v2(weights=None) # load model without pre-trained weights
model.classifier[1] = nn.Linear(model.last_channel, 10)

# load weights
model_dir = os.path.join(os.path.dirname(__file__), "..", "models", "image_classification_model")
model.load_state_dict(torch.load(
    f"{model_dir}/model.pth",
    map_location=torch.device('cpu')
))
model.eval()

def predict_animal(image_path):
    # preprocess image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_class = torch.argmax(outputs, dim=1).item()
    
    return class_names[predicted_class]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    predicted_label = predict_animal(args.image)
    print(f"Predicted class: {predicted_label}")