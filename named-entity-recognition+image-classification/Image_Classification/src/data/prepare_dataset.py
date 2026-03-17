import torch
from Image_Classification.src.config import (
    device,
    PROCESSED_DATASET_PATH,
    AUGMENTED_DATASET_PATH,
    BATCH_SIZE
)
from Image_Classification.src.data.dataset_utils import (
    load_dataset,
    get_class_distribution,
    split_dataset,
    create_dataloaders,
    save_dataset_config
)
from Image_Classification.src.data.augmentation import get_augmentation_transform, augment_images

def main():
    print("Using device:", device)

    # 1. Load the original dataset
    full_dataset = load_dataset(PROCESSED_DATASET_PATH)
    class_counts = get_class_distribution(full_dataset)
    max_samples = max(class_counts.values())
    print("Class distribution before augmentation:", class_counts)
    print(f"Target number of images per class: {max_samples}")

    # 2. Augment the dataset
    augmentation_transform = get_augmentation_transform()
    augment_images(full_dataset, augmentation_transform, max_samples, AUGMENTED_DATASET_PATH)
    print("Final augmented dataset saved successfully!")

    # 3. Load the augmented dataset
    augmented_dataset = load_dataset(AUGMENTED_DATASET_PATH)
    class_counts = get_class_distribution(augmented_dataset)
    print("Final class distribution:", class_counts)

    # 4. Split the dataset into training and validation sets
    train_dataset, val_dataset = split_dataset(augmented_dataset, train_ratio=0.8)

    # 5. Create DataLoaders for both sets
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, BATCH_SIZE)
    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")

    # 6. Save the dataset configuration to a pickle file
    data_to_save = {
        "train_samples": [(img[0], img[1]) for img in train_dataset.dataset.samples],
        "val_samples": [(img[0], img[1]) for img in val_dataset.dataset.samples],
        "device": device,
        "full_dataset_classes": full_dataset.classes
    }
    save_dataset_config("dataset_config.pkl", data_to_save)
    print("Dataset configuration saved successfully!")

if __name__ == "__main__":
    main()
