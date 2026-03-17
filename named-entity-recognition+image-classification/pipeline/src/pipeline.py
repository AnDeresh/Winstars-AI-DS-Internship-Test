import  sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from pipeline.src.ner import extract_animal_entities
from pipeline.src.config import DEVICE
from pipeline.src.image_classifier import (
    load_image_model,
    load_class_labels,
    get_transform_pipeline,
    predict_image
)

# Load model and labels once
_image_model = load_image_model()
_class_labels = load_class_labels()
_transform = get_transform_pipeline()

def run_pipeline(text: str, image_path: str) -> bool:
    """
    Run the complete pipeline:
      1) Extract animal entities from text using NER.
      2) Predict the animal class from the image.
      3) Compare the predicted label with the extracted entities.
    
    Args:
        text (str): Input text message.
        image_path (str): Path to the input image.
        
    Returns:
        bool: True if the predicted animal is mentioned in the text; otherwise, False.
    """
    extracted_animals = extract_animal_entities(text)
    print(f"Extracted animal entities: {extracted_animals}")
    
    predicted_label, _ = predict_image(_image_model, _transform, image_path, _class_labels)
    print(f"Predicted animal from image: {predicted_label}")
    
    # Case-insensitive comparison
    predicted_label = predicted_label.lower()
    extracted_animals = [animal.lower() for animal in extracted_animals]
    
    return any(predicted_label in animal for animal in extracted_animals)