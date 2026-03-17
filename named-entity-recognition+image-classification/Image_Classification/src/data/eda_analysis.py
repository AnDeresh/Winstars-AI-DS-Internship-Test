import os
import random
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from ..config import RAW_DATASET_PATH

# Italian to English translation dictionary
translate = {
    "cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly",
    "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep",
    "ragno": "spider", "scoiattolo": "squirrel"
}

def detect_and_rename_categories(dataset_path=RAW_DATASET_PATH, translate_dict=translate):
    """
    Checks whether the category names are Italian or English and renames them from Italian to English.
    """
    categories = sorted(os.listdir(dataset_path))
    all_italian = all(cat in translate_dict for cat in categories)
    all_english = all(cat in translate_dict.values() for cat in categories)

    if all_italian:
        print("Detected Italian category names. Renaming to English...")
        for it_name, en_name in translate_dict.items():
            it_path = os.path.join(dataset_path, it_name)
            en_path = os.path.join(dataset_path, en_name)
            if os.path.exists(it_path):
                os.rename(it_path, en_path)
                print(f"Renamed: {it_name} -> {en_name}")
            else:
                print(f"Skipped: {it_name} (folder not found)")
        print("Renaming completed!")
    elif all_english:
        print("Detected English category names. No renaming needed.")
    else:
        print("Mixed naming detected. Please check manually.")

def get_categories(dataset_path=RAW_DATASET_PATH):
    """Returns a sorted list of categories (folders)"""
    return sorted(os.listdir(dataset_path))

def count_images_in_categories(dataset_path=RAW_DATASET_PATH, categories=None):
    """Counts the number of images in each category."""
    if categories is None:
        categories = get_categories(dataset_path)
    class_counts = {}
    for cat in categories:
        cat_path = os.path.join(dataset_path, cat)
        class_counts[cat] = len(os.listdir(cat_path))
    return class_counts

def plot_class_distribution(class_counts):
    """Displays a barplot with the number of images in each category."""
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
    plt.xticks(rotation=45)
    plt.xlabel("Animal Category")
    plt.ylabel("Number of Images")
    plt.title("Class Distribution in the Dataset")
    plt.show()

def display_random_images(dataset_path=RAW_DATASET_PATH, categories=None, n=10, rows=2, cols=5):
    """Displays random images from the specified categories."""
    if categories is None:
        categories = get_categories(dataset_path)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 6))
    fig.suptitle("Example Images from Different Classes", fontsize=16)
    sample_cats = random.sample(categories, min(n, len(categories)))
    
    for ax, cat in zip(axes.flatten(), sample_cats):
        cat_path = os.path.join(dataset_path, cat)
        img_name = random.choice(os.listdir(cat_path))
        img_path = os.path.join(cat_path, img_name)
        with Image.open(img_path) as img:
            ax.imshow(img)
        ax.set_title(cat)
        ax.axis("off")
    plt.show()

def analyze_image_sizes(dataset_path=RAW_DATASET_PATH, categories=None):
    """Collects information about image sizes."""
    if categories is None:
        categories = get_categories(dataset_path)
    image_sizes = Counter()
    for cat in categories:
        cat_path = os.path.join(dataset_path, cat)
        for img_name in os.listdir(cat_path):
            img_path = os.path.join(cat_path, img_name)
            with Image.open(img_path) as img:
                image_sizes[img.size] += 1
    return image_sizes

def analyze_file_extensions(dataset_path=RAW_DATASET_PATH, categories=None):
    """Collects information about image formats."""
    if categories is None:
        categories = get_categories(dataset_path)
    file_extensions = Counter()
    for cat in categories:
        cat_path = os.path.join(dataset_path, cat)
        for img_name in os.listdir(cat_path):
            ext = os.path.splitext(img_name)[-1].lower()
            file_extensions[ext] += 1
    return file_extensions