# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 15:00:06 2024

@author: mammadli
"""
import torch
from torchvision import transforms
# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image

class RGBTransform:
    def __call__(self, img):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

class ZScoreNormalization:
    """
    Apply Z-score normalization to the input and then scale to [0, 1].
    """
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        super(ZScoreNormalization, self).__init__()

    def __call__(self, img):
        """
        Apply the transform to `img`.
        
        :param img: Input data (expects a PyTorch tensor)
        :return: Normalized and scaled data
        """
        # Ensure the input tensor is of floating point type
        img = img.float()

        # Check for NaNs or Infinities in the input
        if torch.isnan(img).any() or torch.isinf(img).any():
            raise ValueError("Input tensor contains NaNs or Infinities")

        mean = torch.mean(img)
        std = torch.std(img)

        # Check for zero standard deviation
        if std == 0:
            std += self.epsilon

        img = (img - mean) / std

        # Scale the normalized image to [0, 1] range
        img_min = img.min()
        img_max = img.max()

        # Check if img_min is equal to img_max to avoid division by zero
        if img_min == img_max:
            # If all values in the tensor are the same, set to zero to avoid NaN
            img = torch.zeros_like(img)
        else:
            img = (img - img_min) / (img_max - img_min)
        
        return img

# Original transform pipeline
transform_zscore = transforms.Compose([
    RGBTransform(),                # Ensure the image is in RGB format
    transforms.Resize((224, 224)), # Resize image to 224x224
    transforms.ToTensor(),         # Convert image to tensor
    ZScoreNormalization()          # Apply [0, 1] normalization
])


