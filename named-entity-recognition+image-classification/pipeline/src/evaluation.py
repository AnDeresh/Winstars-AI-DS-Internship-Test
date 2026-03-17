import os, sys
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from pipeline.src.pipeline import run_pipeline, extract_animal_entities
from pipeline.src.config import EVALUATION_JSON_PATH, EVALUATION_IMAGES_DIR

def test_evaluation_data():
    """
    Automatically test the pipeline using evaluation_data.json and test images.
    """
    if not EVALUATION_JSON_PATH.exists():
        print(f"Evaluation JSON not found at {EVALUATION_JSON_PATH}")
        return
    
    with EVALUATION_JSON_PATH.open("r", encoding="utf-8") as f:
        evaluation_data = json.load(f)
    
    valid_extensions = [".jpg", ".jpeg", ".png"]
    known_animals = ["dog", "cat", "butterfly", "elephant", "horse", "spider", "bird"]
    
    for item in evaluation_data:
        text_input = item["text"]
        image_name = item["image"]
        expected = item["expected"]
        
        image_path = None
        for ext in valid_extensions:
            candidate_path = EVALUATION_IMAGES_DIR / f"{image_name}{ext}"
            if candidate_path.exists():
                image_path = candidate_path
                break
        
        if image_path is None:
            print(f"Image file '{image_name}' not found (expected one of {', '.join(valid_extensions)}).")
            continue
        
        result = run_pipeline(text_input, str(image_path))
        extracted_entities = extract_animal_entities(text_input)
        
        text_lower = text_input.lower()
        animal_in_text = any(animal in text_lower for animal in known_animals)
        
        if animal_in_text and not extracted_entities:
            status = "FAILED (No animal extracted from text)"
        else:
            status = "PASSED" if (result == expected) else "FAILED"
        
        print(f"Text: {text_input}")
        print(f"Extracted animal entities: {extracted_entities}")
        print(f"Image: {image_path}")
        print(f"Predicted: {result}, Expected: {expected}")
        print(f"Test {status}\n")

def interactive_mode():
    """
    Run the pipeline in interactive mode for user input.
    """
    print("Interactive mode for animal recognition pipeline. Type 'exit' to quit.\n")
    while True:
        text_input = input("Enter a text message: ")
        if text_input.lower() == "exit":
            break
        
        image_path = input("Enter the full path to an image: ")
        if image_path.lower() == "exit":
            break
        
        image_path = image_path.strip().strip('"').strip("'")
        if not os.path.exists(image_path):
            print(f"File '{image_path}' not found. Please try again.\n")
            continue
        
        result = run_pipeline(text_input, image_path)
        print(f"\nPipeline result: {result}\n")
        print("-" * 50 + "\n")
