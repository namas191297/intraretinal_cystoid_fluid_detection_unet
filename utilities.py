import torch
import PIL
import numpy as np
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import hyperparams
from dataset import RetinalDataset
from torch.utils.data import DataLoader

def get_transforms():

    # Compute the training transforms
    train_transform = A.Compose(
        [
            A.Resize(height=hyperparams['image_height'], width=hyperparams['image_width']),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(   # Needs to be changed for 3 channels
                mean=[0.0], #[0.0, 0.0, 0.0]
                std=[1.0], #[1.0, 1.0, 1.0]
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    # Compute the Validation transforms - will not require Rotations and Flips
    val_transform = A.Compose(
        [
            A.Resize(height=hyperparams['image_height'], width=hyperparams['image_width']),
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    return train_transform, val_transform

def get_dataloaders():

    # Get the transforms
    train_transform, val_transform = get_transforms()

    # Initialize the datasets for training and validation
    train_dataset = RetinalDataset(transform=train_transform, train=True)
    val_dataset = RetinalDataset(transform=val_transform, train=False)

    # Create the dataloaders
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=hyperparams['batch_size'], pin_memory=hyperparams['pin_memory'], num_workers=hyperparams['num_workers'])
    val_loader = DataLoader(dataset=val_dataset, shuffle=True, batch_size=hyperparams['batch_size'],
                              pin_memory=hyperparams['pin_memory'], num_workers=hyperparams['num_workers'])

    return train_loader, val_loader