import os
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
from data_preparation import CustomImageDataset, get_augmentations

def get_train_test_dataset(DATASET_PTH: str, val_numb: int, aug_level: int) -> Tuple[CustomImageDataset, CustomImageDataset]:
    """
    Loads and prepares the training and testing datasets from the specified directory, 
    applying the required augmentations for each.

    Args:
        DATASET_PTH (str): Path to the dataset directory.
        val_numb (int): The value representing the 'donor' used for validation split.
        aug_level (int): The level of augmentations to apply (passed to `get_augmentations`).

    Returns:
        Tuple[CustomImageDataset, CustomImageDataset]: A tuple containing the training and testing datasets.
    """
    # Define the path to the dataset and set up augmentations for training and testing
    img_dir = DATASET_PTH
    train_transform = get_augmentations('train', aug_level)
    test_transform = get_augmentations('test', aug_level)

    # Read the annotations CSV file
    train_val_df = pd.read_csv(os.path.join(img_dir, 'annotations.csv'))
    print(f"Loaded annotations file: {os.path.join(img_dir, 'annotations.csv')}")

    # Split the dataset into training and testing based on 'donor'
    train_df = train_val_df[train_val_df['donnor'] != val_numb]
    test_df = train_val_df[train_val_df['donnor'] == val_numb]
    print(f"Training data size: {len(train_df)} samples, Test data size: {len(test_df)} samples")

    # Create the custom datasets for training and testing
    train_dataset = CustomImageDataset(img_dir, train_df, train_transform)
    test_dataset = CustomImageDataset(img_dir, test_df, test_transform)

    return train_dataset, test_dataset
