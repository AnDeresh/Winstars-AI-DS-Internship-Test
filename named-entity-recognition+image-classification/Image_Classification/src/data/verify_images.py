import os
from PIL import Image
from ..config import PROCESSED_DATASET_PATH

def check_processed_images():
    """
    Checks the processed images by listing available categories and verifying image sizes.
    """
    categories = sorted(os.listdir(PROCESSED_DATASET_PATH))
    print(f"Processed categories: {categories}")

    image_sizes = set()
    for category in categories:
        category_path = os.path.join(PROCESSED_DATASET_PATH, category)
        images = os.listdir(category_path)
        
        print(f"{category}: {len(images)} images")

        for img_name in images[:5]:  # Check first 5 images per category
            img_path = os.path.join(category_path, img_name)
            with Image.open(img_path) as img:
                image_sizes.add(img.size)

    print("Unique image sizes:", image_sizes)