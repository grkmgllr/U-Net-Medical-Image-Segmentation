# U-Net Medical Image Segmentation with PyTorch

This project implements a U-Net model for binary image segmentation, specifically for medical images related to stomach segmentation. The model is trained and evaluated on a public dataset, with the goal of accurately segmenting relevant regions in each image.

Table of Contents:

-> Project Overview

-> Project Structure

-> Requirements

-> Setup and Installation

-> Dataset Preparation

-> Training the Model

-> Testing the Model

-> Results

## Project Overview

This project leverages a U-Net model architecture to perform binary segmentation on medical images. The U-Net architecture, with its encoder-decoder structure and skip connections, is well-suited for medical image analysis. The project includes:

- A training script to train the U-Net model.
- A testing script to evaluate the model on unseen data.
- Utility modules for data handling, metric computation, and visualization.


## Project Structure

-> model/

- unet.py # U-Net model definition
  
->utils/

- data_utils.py # Dataset class and data loading utilities
- metric_utils.py # Metric computation functions (e.g., Dice score)
- model_utils.py # Model saving, argument parsing, and seed setting
- viz_utils.py # Visualization and plotting functions

-> train.py # Script to train the U-Net model

->test.py # Script to evaluate the model on test data

->README.md # Project documentation

## Requirements

This project requires the following libraries:

- Python 3.7 or higher
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- tqdm
- scikit-learn
- torchvision

## Installation

To install the required libraries, run:

- pip install torch torchvision opencv-python numpy matplotlib tqdm scikit-learn

For M1/M2 Macs, ensure you have the correct version of PyTorch with Metal (MPS) acceleration enabled.

## Setup and Installation

1. Clone the Repository:

- git clone [https://github.com/grkmgllr/U-Net-Medical-Image-Segmentation.git]
- cd your repository directory

2. Configure Paths:
- Update data_path in train.py and test.py to point to your dataset.
- Update save_path to specify where to save results and model checkpoints.

3. (Optional) Create a Virtual Environment:
- python3 -m venv venv
- source venv/bin/activate

4. Install Dependencies (if not already installed):
- pip install -r requirements.txt

## Dataset Preparation

This project uses a public dataset for stomach segmentation, adapted from the UW-Madison GI Tract Image Segmentation (https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation). Place the dataset in the specified data_path, with the following structure:

dataset/

- train/
  - image1.png
  - mask1.png
  - ...
- test/
  - image1.png
  - mask1.png # Optional for test mode
  - ...

## Training the Model

To train the U-Net model, run:

- python train.py --epoch 10 --bs 8

This command will train the model for 10 epochs with a batch size of 8. The model checkpoints and training history plots will be saved to the specified save_path.

## Testing the Model

After training, evaluate the model on the test set with:

- python test.py --model_path /path/to/best/model.pt --bs 8

This command loads the specified model checkpoint and runs inference on the test set, outputting Dice scores and predictions.

## Results

The model’s performance on the test set will be summarized in terms of Dice score. Results and visualizations are saved in the save_path directory, with plots of training and validation losses, as well as segmentation results for a sample of test images.

## Acknowledgments

- U-Net: Convolutional Networks for Biomedical Image Segmentation - Link to paper: https://arxiv.org/abs/1505.04597
- UW-Madison GI Tract Image Segmentation: https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation
