# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 00:23:35 2024

@author: mammadli
"""

from pathlib import Path
import numpy as np
import os, shutil
import matplotlib.pyplot as plt
import pandas as pd

from PIL import Image

from tqdm.auto import tqdm

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, f1_score

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import torch.optim as optim
import torch.nn as nn

from torchvision.models import vgg16, VGG16_Weights

class vgg_feature(torch.nn.Module):
    def __init__(self):
        """This class extracts the feature maps from a pretrained VGG16 model."""
        super(vgg_feature, self).__init__()
        self.model = vgg16(weights=VGG16_Weights.DEFAULT)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Hook to extract feature maps
        def hook(module, input, output) -> None:
            """This hook saves the extracted feature map on self.features."""
            self.features.append(output)

        #self.model.features[16].register_forward_hook(hook)  # Example layer 1
        self.model.features[23].register_forward_hook(hook)  # Example layer 2
        self.model.features[30].register_forward_hook(hook)  # Example layer 3

    def forward(self, input):

        self.features = []
        with torch.no_grad():
            _ = self.model(input)

        self.avg = torch.nn.AvgPool2d(3, stride=1)
        fmap_size = self.features[0].shape[-2]         # Feature map sizes h, w
        self.resize = torch.nn.AdaptiveAvgPool2d(fmap_size)

        resized_maps = [self.resize(self.avg(fmap)) for fmap in self.features]
        patch = torch.cat(resized_maps, 1)            # Merge the resized feature maps

        return patch
    
    
class VGG_CAE(nn.Module):
    """Autoencoder."""

    def __init__(self, in_channels=1024, latent_dim=100, is_bn=True):
        super(VGG_CAE, self).__init__()

        layers = []
        layers += [nn.Conv2d(in_channels, (in_channels + 2 * latent_dim) // 2, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=(in_channels + 2 * latent_dim) // 2)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d((in_channels + 2 * latent_dim) // 2, 2 * latent_dim, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=2 * latent_dim)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d(2 * latent_dim, latent_dim, kernel_size=1, stride=1, padding=0)]

        self.encoder = nn.Sequential(*layers)

        layers = []
        layers += [nn.Conv2d(latent_dim, 2 * latent_dim, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=2 * latent_dim)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d(2 * latent_dim, (in_channels + 2 * latent_dim) // 2, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=(in_channels + 2 * latent_dim) // 2)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d((in_channels + 2 * latent_dim) // 2, in_channels, kernel_size=1, stride=1, padding=0)]

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
def train_vgg(model, backbone, criterion, optimizer, batch_size, train_dataset, val_dataset, num_epochs, output_csv, output_model):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    train_loss_list = []
    val_loss_list = []

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        
        # Training loop
        for data in train_loader:
            inputs = data.cuda()
            optimizer.zero_grad()
            
            with torch.no_grad():
                features = backbone(inputs)
            
            outputs = model(features)
            loss = criterion(outputs, features)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_loss_list.append(avg_train_loss)

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        running_val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                inputs = data.cuda()
                features = backbone(inputs)
                outputs = model(features)
                val_loss = criterion(outputs, features)
                running_val_loss += val_loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_loss_list.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save the model checkpoint after each epoch if validation loss improves
        torch.save(model.state_dict(), output_model)

    # Save losses to a CSV file
    df = pd.DataFrame({'epoch': list(range(1, num_epochs+1)), 'train_loss': train_loss_list, 'val_loss': val_loss_list})
    df.to_csv(output_csv, index=False)

    return model