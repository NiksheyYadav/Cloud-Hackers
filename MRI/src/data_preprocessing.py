import os
import torch # pylint: disable=unused-import
from PIL import Image
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms

from src.config import (BATCH_SIZE, IMG_SIZE, NORMALIZE_MEAN, NORMALIZE_STD, TEST_DIR, TRAIN_DIR, VAL_DIR) # pylint: disable=unused-import

class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = [f for f in os.listdir(root) if f.endswith('.png') or f.endswith('.jpg')]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.images[idx])
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, 0  # Placeholder for label if not available

def load_data(data_dir, img_size=IMG_SIZE, mean=None, std=None):
    """
    Loads and preprocesses the image data from the specified directory.
    
    Args:
        data_dir (str): Directory containing the images.
        img_size (int): Target size to resize images.
        mean (list, optional): Mean values for normalization.
        std (list, optional): Standard deviation values for normalization.
        
    Returns:
        Dataset: Preprocessed dataset object.
    """
    mean = mean or NORMALIZE_MEAN
    std = std or NORMALIZE_STD

    data_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return datasets.ImageFolder(data_dir, transform=data_transforms)


def preprocess_images(dataset):
    """
    Applies any additional preprocessing (if needed). In this case, transformations are applied during loading.
    """
    # Since transformations are applied in `load_data`, nothing extra here unless needed.
    return dataset

    
def split_data(dataset, val_split=0.2, test_split=0.1):
    """
    Splits the dataset into training, validation, and test sets.
    
    Args:
        dataset (Dataset): The full dataset to split.
        val_split (float): Fraction of data to use for validation.
        test_split (float): Fraction of data to use for testing.
        
    Returns:
        tuple: Splitted training, validation, and test datasets.
    """
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    test_size = int(total_size * test_split)
    train_size = total_size - val_size - test_size

    # Split the dataset into train, validation, and test
    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])
    
    print(f"Data split into {train_size} training, {val_size} validation, and {test_size} test images")
    
    return train_data, val_data, test_data


def create_dataloaders(train_data, val_data, test_data):
    """
    Creates DataLoaders for training, validation, and testing.
    
    Args:
        train_data (Dataset): Training dataset.
        val_data (Dataset): Validation dataset.
        test_data (Dataset): Test dataset.
        
    Returns:
        tuple: Dataloaders for train, validation, and test datasets.
    """
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader