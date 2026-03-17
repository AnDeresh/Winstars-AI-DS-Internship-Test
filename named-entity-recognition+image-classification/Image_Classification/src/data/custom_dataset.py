from torch.utils.data import Dataset
from torchvision import datasets
from Image_Classification.src.config import AUGMENTED_DATASET_PATH 

class CustomDataset(Dataset):
    def __init__(self, samples, transform=None):
        """
        Custom dataset that loads images dynamically.

        Args:
            samples (list): List of (image_path, label) tuples.
            transform (callable, optional): Transformations to apply to images.
        """
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):
        rel_img_path, label = self.samples[index]
        # If rel_img_path contains an unnecessary prefix, for example, "data/augmented-img", remove it:
        prefix = "data/augmented-img"
        if rel_img_path.startswith(prefix):
            # Remove the prefix and possible slashes
            rel_img_path = rel_img_path[len(prefix):].lstrip("/\\")
        # Shaping the absolute path
        img_path = AUGMENTED_DATASET_PATH / rel_img_path
        img = datasets.folder.default_loader(str(img_path))
        if self.transform:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.samples)