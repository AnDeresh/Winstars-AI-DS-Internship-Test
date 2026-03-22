import argparse

from ner.inference import extract_animal
from classification.inference import predict_animal

def run_pipeline(text, image_path):
    # 1. extract animal from text
    animal_from_text = extract_animal(text)
    
    # 2. predict animal from image
    animal_from_image = predict_animal(image_path)
    
    # 3. compare
    return animal_from_text == animal_from_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()
    
    result = run_pipeline(args.text, args.image)
    print(result)