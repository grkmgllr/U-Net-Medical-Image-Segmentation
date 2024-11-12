import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from model.unet import UNet
from utils.model_utils import train_arg_parser, set_seed
from utils.data_utils import MadisonStomach
from utils.viz_utils import visualize_predictions, plot_train_val_history, plot_metric
from utils.metric_utils import compute_dice_score


def train_model(model, train_loader, val_loader, optimizer, criterion, args, save_path):
    '''
    Trains the given model over multiple epochs, tracks training and validation losses, 
    and saves model checkpoints periodically.

    Args:
    - model (torch.nn.Module): The neural network model to be trained.
    - train_loader (DataLoader): DataLoader for the training dataset.
    - val_loader (DataLoader): DataLoader for the validation dataset.
    - optimizer (torch.optim.Optimizer): The optimizer used for updating model weights.
    - criterion (torch.nn.Module): The loss function used for training.
    - args (argparse.Namespace): Parsed arguments containing training configuration (e.g., epochs, batch size, device).
    - save_path (str): Directory path to save model checkpoints and training history.

    Functionality:
    - Creates directories to save results and checkpoints.
    - Calls `train_one_epoch` to train and validate the model for each epoch.
    - Saves model checkpoints every 5 epochs.
    - Plots the training and validation loss curves and the Dice coefficient curve.
    '''
    os.makedirs(os.path.join(save_path, args.exp_id), exist_ok=True)
    os.makedirs(os.path.join(save_path, args.exp_id, 'model'), exist_ok=True)

    train_loss_history = []
    val_loss_history = []
    dice_coef_history = []

    for epoch in range(args.epoch):
        train_one_epoch(model, 
                        train_loader, 
                        val_loader, 
                        train_loss_history, 
                        val_loss_history, 
                        dice_coef_history, 
                        optimizer, 
                        criterion, 

                        args, 
                        epoch, 
                        save_path)
        
        if (epoch + 1) % 5 == 0:
            torch.save(model, os.path.join(save_path, args.exp_id, 'model', f'unet_{epoch}.pt'))

    plot_train_val_history(train_loss_history, val_loss_history, save_path, args)
    plot_metric(dice_coef_history, label="dice coeff", plot_dir=save_path, args=args, metric='dice_coeff')

def train_one_epoch(model, train_loader, val_loader, train_loss_history, val_loss_history, 
                    dice_coef_history, optimizer, criterion, args, epoch, save_path):
    '''
    Performs one full epoch of training and validation, computes metrics, and visualizes predictions.

    Args:
    - model (torch.nn.Module): The neural network model to train.
    - train_loader (DataLoader): DataLoader for the training dataset.
    - val_loader (DataLoader): DataLoader for the validation dataset.
    - train_loss_history (list): List to store the average training loss per epoch.
    - val_loss_history (list): List to store the average validation loss per epoch.
    - dice_coef_history (list): List to store the Dice coefficient per epoch.
    - optimizer (torch.optim.Optimizer): The optimizer used for updating model weights.
    - criterion (torch.nn.Module): The loss function used for training.
    - args (argparse.Namespace): Parsed arguments containing training configuration.
    - epoch (int): The current epoch number.
    - save_path (str): Directory path to save visualizations and model checkpoints.

    Functionality:
    - Sets the model to training mode and performs a forward and backward pass for each batch in the training data.
    - Computes the training loss and updates the weights.
    - Sets the model to evaluation mode and computes validation loss and Dice coefficients.
    - Visualizes predictions periodically and saves them to the specified directory.
    - Appends the average training and validation losses, and the Dice coefficient to their respective lists.
    - Prints the Dice coefficient and loss values for the current epoch.
    '''

    model.train()
    train_loss = 0.0
    train_dice = 0.0

    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epoch}"):
        images, masks = images.to(args.device), masks.to(args.device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        train_dice += compute_dice_score(preds, masks)

    train_loss /= len(train_loader)
    train_dice /= len(train_loader)
    train_loss_history.append(train_loss)

    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(args.device), masks.to(args.device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            val_dice += compute_dice_score(preds, masks)

    val_loss /= len(val_loader)
    val_dice /= len(val_loader)
    val_loss_history.append(val_loss)
    dice_coef_history.append(val_dice)

    print(
        f"Epoch {epoch + 1}/{args.epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

    if epoch % 5 == 0:
        visualize_predictions(images, masks, outputs, save_path, epoch, batch_idx=0)


if __name__ == '__main__':

    args = train_arg_parser()
    save_path = "/path/to/your/results/folder"
    set_seed(42)

    dataset = MadisonStomach(data_path="/path/to/your/madison-stomach/folder",
                            mode=args.mode)


    train_indices, val_indices = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.bs, shuffle=False)

    model = UNet(in_channels=1, out_channels=1).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_model(model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                args=args,
                save_path=save_path)
