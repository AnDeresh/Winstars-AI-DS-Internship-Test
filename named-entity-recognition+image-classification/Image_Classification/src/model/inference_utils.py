from PIL import Image
import torch

def get_inference_transform():
    """
    Return the transformation pipeline for inference.
    
    Returns:
        transform: A composed transformation for image preprocessing.
    """
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

def predict_image(model, transform, class_labels, image_path, device):
    """
    Predict the class of a given image.
    
    Args:
        model: The trained model.
        transform: Transformation pipeline for preprocessing the image.
        class_labels (dict): Mapping of class indices to labels.
        image_path (str): Path to the input image.
        device (torch.device): Device on which inference is performed.
    
    Returns:
        tuple: (predicted_label, image) where image is a PIL Image.
    """
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    return class_labels[str(predicted_class)], image
