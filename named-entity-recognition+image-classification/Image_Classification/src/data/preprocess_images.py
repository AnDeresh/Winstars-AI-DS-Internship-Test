import os
from PIL import Image
from ..config import RAW_DATASET_PATH, PROCESSED_DATASET_PATH, TARGET_SIZE

def process_images():
    """
    Processes images by resizing and converting them to JPEG format.
    """
    for category in sorted(os.listdir(RAW_DATASET_PATH)):
        input_folder = os.path.join(RAW_DATASET_PATH, category)
        output_folder = os.path.join(PROCESSED_DATASET_PATH, category)
        os.makedirs(output_folder, exist_ok=True)
        
        for img_name in os.listdir(input_folder):
            img_path = os.path.join(input_folder, img_name)
            new_img_name = os.path.splitext(img_name)[0] + ".jpg"  # Convert all to .jpg
            new_img_path = os.path.join(output_folder, new_img_name)
            
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")  # Ensure RGB format
                    img = img.resize(TARGET_SIZE)  # Resize to target size
                    img.save(new_img_path, "JPEG")  # Save as .jpg
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    print("Image processing completed!")

def preprocess_user_image(image_path):
    """
    Loads an image, converts it to RGB, resizes, and returns the processed image.
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")  # Convert to RGB
            img = img.resize(TARGET_SIZE)  # Resize to match model input
            return img
    except Exception as e:
        print(f"Error processing image: {e}")
        return None