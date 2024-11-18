import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import Optional, Tuple

class CustomImageDataset(Dataset):
    """
    Custom dataset for loading images and their labels from a given directory 
    and dataframe.

    Args:
        img_dir (str): Path to the image directory.
        df (pd.DataFrame): Dataframe containing image filenames and corresponding labels.
        transform (Optional[transforms.Compose], optional): Transformations to apply to the images. Defaults to None.

    Attributes:
        img_dir (str): Path to the image directory.
        img_name (np.ndarray): List of image filenames.
        img_labels (np.ndarray): Corresponding labels for each image.
        transform (Optional[transforms.Compose]): Transformation to apply to the images.
    """
    
    def __init__(self, img_dir: str, df: pd.DataFrame, transform: Optional[transforms.Compose] = None) -> None:
        self.img_dir = img_dir
        self.img_name = df['file_name'].values
        self.img_labels = df['label'].values.astype(float)
        self.transform = transform

    def __len__(self) -> int:
        """
        Returns the total number of images in the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return len(self.img_name)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        """
        Fetches an image and its corresponding label by index.

        Args:
            idx (int): Index of the image in the dataset.

        Returns:
            Tuple[torch.Tensor, float]: A tuple containing the transformed image tensor and its label.
        """
        img_path = os.path.join(self.img_dir, self.img_name[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def get_augmentations(phase: str, level_int: int) -> transforms.Compose:
    """
    Returns image augmentations based on the training phase and intensity level.

    Args:
        phase (str): The phase of the process, either 'train' or 'test'.
        level_int (int): The intensity level of augmentations (1, 2, or 3 for 'train' phase).

    Returns:
        transforms.Compose: A composition of transformations to be applied to the images.
    """
    if phase == 'train':
        if level_int == 1:
            print("Applying basic augmentations (level 1) for training")
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation([90, 270]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif level_int == 2:
            print("Applying moderate augmentations (level 2) for training")
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(hue=0.1, contrast=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=None, shear=None),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif level_int == 3:
            print("Applying advanced augmentations (level 3) for training")
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(hue=0.1, contrast=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=None, shear=None),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    else:
        # Default test-time augmentations
        print("Applying test-time augmentations")
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
