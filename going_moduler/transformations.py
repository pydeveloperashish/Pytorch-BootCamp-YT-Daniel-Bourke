
# Lets create transformations
import torch
from torchvision import datasets, transforms


def data_transform_function(img_size: int):
    data_transform = transforms.Compose([
        transforms.Resize(size = (img_size, img_size)),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomVerticalFlip(p = 0.5),
        transforms.ToTensor()
        ])
    return data_transform
