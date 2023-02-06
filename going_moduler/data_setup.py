
"""
Contains functionality for creating Pytorch DataLoader's for image classification data.
"""

import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import os

NUM_WORKERS = os.cpu_count()


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform,
    batch_size: int,
    num_workers: int = NUM_WORKERS
    ):
    
    """
    Creates training and testing  DataLoaders.
    
    Takes in a training and testing directory path and turns them into
    Pytorch Datasets and then into Pytorch DataLoaders.
    
    Args:
        train_dir: Path to training dir.
        test_dir: Path to testing dir.
        transform: torchvision transforms to perform on training and testing data.
        batch_size: no of Batch Size.
        num_workers: workers per DataLoader. 
    
    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
        Where class_names is a list of the target classes.
    """
    
    
    
    # Use ImageFolder to create datasets
    train_data = datasets.ImageFolder(train_dir, transform = transform)
    test_data = datasets.ImageFolder(test_dir, transform = transform)
    
    # Get class names
    class_names  = train_data.classes
    
    # Turn images into DataLoaders
    train_dataloader = DataLoader(
        train_data,
        batch_size = batch_size,
        shuffle = True,
        num_workers = NUM_WORKERS,
        pin_memory = True
        )
    
    test_dataloader = DataLoader(
        test_data,
        batch_size = batch_size,
        shuffle = False,
        num_workers = NUM_WORKERS,
        pin_memory = True
        )
    
    return train_dataloader, test_dataloader, class_names
