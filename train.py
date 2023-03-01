import torch
import torch.nn as nn
import PIL
import numpy as np
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import hyperparams
from dataset import RetinalDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from utilities import get_dataloaders
from model import UNet
from eval import evaluate
import skimage
from skimage import io
from skimage import color

def main():

    # Get the loaders
    train_loader, val_loader = get_dataloaders()

    # Initialize the network
    model = UNet(hyperparams['img_channels'], hyperparams['out_channels']).to(hyperparams['device'])

    # Initialize optimizer, loss and scaler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    scaler = torch.cuda.amp.GradScaler()

    # TQDM for progress bar
    training_loop = tqdm(train_loader)

    # Forward loop
    for epoch in range(hyperparams['num_epochs']):
        losses = []
        for batch_id, (imgs, masks) in enumerate(training_loop):

            # Move the images and masks to the device
            imgs = imgs.to(hyperparams['device'])
            masks = masks.float().unsqueeze(1).to(hyperparams['device'])

            # Predict the masks using the network
            with torch.cuda.amp.autocast():
                pred_masks = model(imgs)
                loss = criterion(pred_masks, masks)

            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update progress bar
            training_loop.set_postfix(loss=loss.item())
            losses.append(loss.item())

        print(f'Epoch:{epoch + 1} Loss:{np.asarray(losses).mean()}')

    # Evaluation (can be skipped)
    torch.save(model.state_dict(), 'saved_models/first_model')
    evaluate(model, val_loader)

if __name__ == '__main__':
    main()