import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from model.unet import UNet
from utils.model_utils import test_arg_parser, set_seed
from utils.data_utils import MadisonStomach
from utils.viz_utils import visualize_predictions, plot_train_val_history, plot_metric
from utils.metric_utils import compute_dice_score

def test_model(model, args, save_path):
    '''
    Tests the model on the test dataset and computes the average Dice score.
    
    Args:
    - model (torch.nn.Module): The PyTorch model to test.
    - args (argparse.Namespace): Parsed arguments for device, batch size, etc.
    - save_path (str): Directory where results (e.g., metrics plot) will be saved.
    
    Functionality:
    - Sets the model to evaluation mode and iterates over the test dataset.
    - Computes the Dice score for each batch and calculates the average.
    - Saves a plot of the Dice coefficient history.
    '''

    model.eval()
    dice_scores = []

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(test_dataloader, desc="Testing")):
            images, masks = images.to(args.device), masks.to(args.device)

            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            dice_score = compute_dice_score(preds, masks)
            dice_scores.append(dice_score)

            if batch_idx % 5 == 0:
                visualize_predictions(images, masks, outputs, save_path, epoch=0, batch_idx=batch_idx)

    avg_dice_score = sum(dice_scores) / len(dice_scores)
    print(f"Average Dice Score on Test Set: {avg_dice_score:.4f}")

    plot_metric(dice_scores, label="Dice Score", plot_dir=save_path, args=args, metric='dice_score')
    return avg_dice_score

if __name__ == '__main__':

    args = test_arg_parser()
    save_path = "/path/to/your/results/folder"
    set_seed(42)

    dataset = MadisonStomach(data_path="/path/to/your/madison-stomach/folder",
                            mode="test")

    test_dataloader = DataLoader(dataset, batch_size=args.bs)

    model = UNet(in_channels=1, out_channels=1).to(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))

    test_model(model=model,
                args=args,
                save_path=save_path)
