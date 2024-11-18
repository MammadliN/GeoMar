# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 00:03:47 2024

@author: mammadli
"""

import torch
from torchvision import transforms

class RGBTransform:
    def __call__(self, img):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    
class To01:
    """
    Apply Min-Max scaling to the input to scale pixel values to [0, 1].
    """
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        super(To01, self).__init__()

    def __call__(self, img):
        """
        Apply Min-Max scaling to `img`.

        :param img: Input data (expects a PyTorch tensor)
        :return: Scaled data in the range [0, 1]
        """
        # Ensure the input tensor is of floating point type
        img = img.float()

        # Check for NaNs or Infinities in the input
        if torch.isnan(img).any() or torch.isinf(img).any():
            raise ValueError("Input tensor contains NaNs or Infinities")

        # Calculate min and max of the image
        img_min = img.min()
        img_max = img.max()

        # Check if img_min is equal to img_max to avoid division by zero
        if img_min == img_max:
            # If all values in the tensor are the same, set to zero to avoid NaN
            img = torch.zeros_like(img)
        else:
            # Perform Min-Max scaling to scale values between 0 and 1
            img = (img - img_min) / (img_max - img_min + self.epsilon)

        return img

transform_minmax = transforms.Compose([
    RGBTransform(),                # Ensure the image is in RGB format
    transforms.Resize((224, 224)), # Resize image to 224x224
    transforms.ToTensor(),         # Convert image to tensor
    To01()                         # Apply [0, 1] normalization
])


