import matplotlib.pyplot as plt
from itertools import product
from torchvision import transforms
from PIL import Image
import numpy as np
import random
import glob
import torch
import os

def visualize_predictions(images, masks, outputs, save_path, epoch, batch_idx):
    '''
    Visualizes and saves sample predictions for a given batch of images, masks, and model outputs.

    Args:
    - images (torch.Tensor): Input images (batch of tensors).
    - masks (torch.Tensor): Ground truth segmentation masks (batch of tensors).
    - outputs (torch.Tensor): Model outputs (batch of tensors).
    - save_path (str): Directory path where the visualization will be saved.
    - epoch (int): Current epoch number (for labeling the file).
    - batch_idx (int): Index of the current batch (for labeling the file).
    
    Functionality:
    - Displays and saves the first few samples from the batch, showing the input images, ground truth masks, and predicted masks.
    - Applies a sigmoid function to the outputs and uses a threshold of 0.5 to convert them to binary masks.
    '''

    outputs = torch.sigmoid(outputs) > 0.5

    fig, axes = plt.subplots(len(images), 3, figsize=(12, len(images) * 4))
    for i in range(len(images)):
        axes[i, 0].imshow(images[i].cpu().squeeze(), cmap='gray')
        axes[i, 0].set_title("Input Image")
        axes[i, 1].imshow(masks[i].cpu().squeeze(), cmap='gray')
        axes[i, 1].set_title("Ground Truth")
        axes[i, 2].imshow(outputs[i].cpu().squeeze(), cmap='gray')
        axes[i, 2].set_title("Prediction")

    plt.tight_layout()
    plt.savefig(f"{save_path}/epoch_{epoch}_batch_{batch_idx}_predictions.png")
    plt.close()

def plot_train_val_history(train_loss_history, val_loss_history, plot_dir, args):
    '''
    Plots and saves the training and validation loss curves.

    Args:
    - train_loss_history (list): List of training loss values over epochs.
    - val_loss_history (list): List of validation loss values over epochs.
    - plot_dir (str): Directory path where the plot will be saved.
    - args (argparse.Namespace): Parsed arguments containing experiment details.
    
    Functionality:
    - Plots the train and validation loss curves.
    - Saves the plot as a JPG file in the specified directory.
    '''

    plt.figure()
    plt.plot(train_loss_history, label="Training Loss")
    plt.plot(val_loss_history, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig(f"{plot_dir}/{args.exp_id}_loss_curve.jpg")
    plt.close()

def plot_metric(x, label, plot_dir, args, metric):
    '''
    Plots and saves a metric curve over epochs.

    Args:
    - x (list): List of metric values over epochs.
    - label (str): Label for the y-axis (name of the metric).
    - plot_dir (str): Directory path where the plot will be saved.
    - args (argparse.Namespace): Parsed arguments containing experiment details.
    - metric (str): Name of the metric (used for naming the saved file).
    
    Functionality:
    - Plots the given metric curve.
    - Saves the plot as a JPEG file in the specified directory.
    '''
    plt.figure()
    plt.plot(x, label=label)
    plt.xlabel("Epochs")
    plt.ylabel(label)
    plt.title(f"{metric} over Epochs")
    plt.savefig(f"{plot_dir}/{args.exp_id}_{metric}_curve.jpg")
    plt.close()