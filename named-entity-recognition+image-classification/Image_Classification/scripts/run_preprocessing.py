import sys
from pathlib import Path

# Add the project root directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from Image_Classification.src.data.preprocess_images import process_images
from Image_Classification.src.data.verify_images import check_processed_images

def main():
    process_images()
    check_processed_images()

if __name__ == "__main__":
    main()