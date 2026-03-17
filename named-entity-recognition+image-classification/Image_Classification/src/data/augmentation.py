import os
import random
from PIL import Image
from torchvision import transforms

def get_augmentation_transform():
    """
    Create and return a Compose transform with all the augmentation operations.
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(300, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    ])

def augment_images(full_dataset, augmentation_transform, max_samples, augmented_dataset_path):
    """
    For each class:
      - Copy the original images to a new folder.
      - Augment images until the total number reaches max_samples.
    """
    from collections import Counter

    for class_idx, class_name in enumerate(full_dataset.classes):
        class_dir = os.path.join(augmented_dataset_path, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # Retrieve image paths for the current class
        original_images = [
            img_path for img_path, label in full_dataset.samples 
            if label == class_idx
        ]

        # Copy original images to the new directory
        for img_path in original_images:
            img = Image.open(img_path).convert("RGB")
            new_img_path = os.path.join(class_dir, os.path.basename(img_path))
            img.save(new_img_path)

        # Augment images until reaching max_samples for the class
        images = list(original_images)  # List of all images in the class
        while len(images) < max_samples:
            img_path = random.choice(original_images)
            img = Image.open(img_path).convert("RGB")
            img = augmentation_transform(img)

            new_img_path = os.path.join(class_dir, f"aug_{len(images)}.jpg")
            img.save(new_img_path)
            images.append(new_img_path)
