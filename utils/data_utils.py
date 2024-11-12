import torch
import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from typing import Any
from torch.utils.data import Dataset
from torchvision import transforms

class MadisonStomach(Dataset):
    '''
    Custom PyTorch Dataset class to load and preprocess images and their corresponding segmentation masks.
    
    Args:
    - data_path (str): The root directory of the dataset.
    - mode (str): The mode in which the dataset is used, either 'train' or 'test'.
    
    Attributes:
    - image_paths (list): Sorted list of file paths for images.
    - mask_paths (list): Sorted list of file paths for masks.
    - transform (Compose): Transformations to apply to the images (convert to tensor and resize).
    - mask_transform (Compose): Transformations to apply to the masks (convert to tensor and resize).
    - augment (bool): Whether to apply data augmentation (only for training mode).
    - augmentation_transforms (Compose): Augmentation transformations (horizontal flip, vertical flip, color jitter).
    '''

    def __init__(self, data_path, mode='train') -> None:
        # Load and sort image and mask file paths
        self.image_paths = sorted(glob.glob(os.path.join(data_path, mode, '*image*.png')))
        self.mask_paths  = sorted(glob.glob(os.path.join(data_path, mode, '*mask*.png')))

        self.mode = mode
        assert self.mask_paths is None or len(self.image_paths) == len(self.mask_paths)

        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])
        self.mask_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])

    def __len__(self):
        # Return the total number of samples
        return len(self.image_paths)

    def __getitem__(self, index):
        '''
        Load and preprocess an image and its corresponding mask at the given index.
        
        Args:
        - index (int): Index of the sample to fetch.
        
        Returns:
        - img (Tensor): Transformed image tensor.
        - mask (Tensor): Transformed mask tensor.
        '''

        img = cv2.imread(self.image_paths[index], cv2.IMREAD_GRAYSCALE)
        img = self.transform(transforms.ToPILImage()(img))

        if self.mode == 'test':
            mask = torch.zeros_like(img)
        else:
            mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_GRAYSCALE)
            mask = self.mask_transform(transforms.ToPILImage()(mask))

        return img, mask
