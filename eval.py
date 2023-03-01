import torch
from config import hyperparams
import torchvision
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io
from skimage import color
from utilities import get_dataloaders
from model import UNet

def evaluate(model, val_loader, save_dir='eval_outputs'):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for idx, (x, y) in enumerate(val_loader):
            x = x.to(hyperparams['device'])
            y = y.to(hyperparams['device']).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
            torchvision.utils.save_image(x, f"{save_dir}/images/image{idx}.png")
            torchvision.utils.save_image(preds, f"{save_dir}/predicted_masks/pred_mask{idx}.png")
            torchvision.utils.save_image(y, f'{save_dir}/real_masks/real_mask{idx}.png')

    print(f'Dice Score:{dice_score/len(val_loader)}')

    # Predicted CME and Ground Truth CME
    plot_eval_results()

def plot_eval_results():
    eval_output_dir = 'eval_outputs'
    images_dir = eval_output_dir + '/' + 'images'
    real_masks_dir = eval_output_dir + '/' + 'real_masks'
    predicted_masks_dir = eval_output_dir + '/' + 'predicted_masks'
    output_dir = eval_output_dir + '/' + 'results'

    # Open the images as PIL.Image
    for i in range(len(os.listdir(images_dir))):
        image = cv2.imread(images_dir + f'/image{i}.png')
        real_mask = np.array(Image.open(real_masks_dir + f'/real_mask{i}.png').convert('L'))
        pred_mask = np.array(Image.open(predicted_masks_dir + f'/pred_mask{i}.png').convert('L'))

        # Apply the mask to the image
        masked_real = color.label2rgb(real_mask, image)
        masked_pred = color.label2rgb(pred_mask, image)

        # Create a grid and save the images
        fig = plt.figure(figsize=(10,10))
        s1 = fig.add_subplot(3,1,1)
        plt.title('Images')
        plt.imshow(image)
        s2 = fig.add_subplot(3,1,2)
        plt.title('Images with Real Masks')
        plt.imshow(masked_real)
        s3 = fig.add_subplot(3,1,3)
        plt.title('Images with Predicted Masks')
        plt.imshow(masked_pred)
        plt.savefig(f'{output_dir}/result{i}.png')

if __name__ == '__main__':
    # Will require you to run train.py first
    train_loader, val_loader = get_dataloaders()
    model = UNet(hyperparams['img_channels'], hyperparams['out_channels']).to(hyperparams['device'])
    model.load_state_dict(torch.load(hyperparams['best_model_path']))
    evaluate(model, val_loader)