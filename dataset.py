import PIL
import numpy as np
import torch
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

class RetinalDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None, dataset_path='intraretinal_dataset', train=True):
        super().__init__()
        self.transform = transform
        if train == True:
            self.images_folder = dataset_path + '/train_images'
            self.masks_folder = dataset_path + '/train_masks'
        else:
            self.images_folder = dataset_path + '/val_images'
            self.masks_folder = dataset_path + '/val_masks'

        self.images = os.listdir(self.images_folder)
        self.masks = os.listdir(self.masks_folder)

    def __getitem__(self, idx):
        image_path = self.images_folder + '/' + f'{self.images[idx]}'
        mask_path = self.masks_folder + '/' + f'{self.masks[idx]}'
        image = np.array(PIL.Image.open(image_path).convert('L'), dtype=np.float32)
        mask = np.array(PIL.Image.open(mask_path).convert('L'), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

    def __len__(self):
        return len(os.listdir(self.images_folder))

def test_dataset():
    train_dataset = RetinalDataset(train=True)
    img, mask = train_dataset[0]
    print(f'Image Size:{img.shape}')
    print(f'Mask Size:{mask.shape}')

if __name__ == '__main__':
    test_dataset()