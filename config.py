import torch

hyperparams = {
    'learning_rate':1e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 3,
    'num_epochs': 3,
    'image_height': 512,
    'image_width': 512,
    'pin_memory':True,
    'num_workers':3,
    'load_model':False,
    'img_channels':1, # Change this for RGB image, 1 channel for Binary Images, 3 for RGB
    'out_channels':1, # Change this for multi-class segmentation
    'best_model_path':'saved_models/first_model'
}