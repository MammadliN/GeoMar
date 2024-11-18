# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 00:01:46 2024

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
import seaborn as sns

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import torch.optim as optim
import torch.nn as nn

import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch

from Norm.MinMax import transform_minmax
from Norm.Zscore import transform_zscore
from VGG.VGG import VGG_CAE, vgg_feature, train_vgg


def euclidean_loss1(output, target):
    return torch.norm(output - target, p=1, dim=1).mean()

def euclidean_loss2(output, target):
    return torch.norm(output - target, p=2, dim=1).mean()

# Define the custom dataset to handle CSV with image paths
class ImageDatasetFromCSV(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)  # Load CSV file
        self.transform = transform

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Get the file path of the image
        img_path = self.data_frame.iloc[idx]["files"]  # Assuming "files" column has image paths
        
        # Open the image using PIL
        image = Image.open(img_path).convert("RGB")
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image
    
def process_images_from_csv(backbone, model, transform, test_csv, output_folder, num_epochs, loss="mse", norm="min", mask=False):
    """
    Function to process images from a CSV file, and save the result.

    Args:
        backbone (torch.nn.Module): Feature extractor (backbone) used in the model.
        model (torch.nn.Module): The model to perform the reconstruction.
        transform (torchvision.transforms.Compose): Image transformations to be applied.
        test_csv (str): Path to the CSV file containing image paths.
        output_folder (str): Folder to save processed images.
        mask (bool): If True, include the mask in the display; otherwise, skip it.
        
    Returns:
        None. The function processes images and displays results, optionally with masks.
    """
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Read the CSV file
    test_df = pd.read_csv(test_csv)
    
    # Loop through all images in the CSV file
    for idx, row in test_df.iterrows():
        image_path = row['files']  # Image paths stored in the 'files' column
        image = Image.open(image_path)  # Open the image

        # Apply transformations and add batch dimension
        image = transform(image).unsqueeze(0)

        # Handle the mask if mask argument is True
        if mask:
            # Use the get_mask_path function to get the mask path
            mask_path = get_mask_path(image_path)
            mask_img = Image.open(mask_path)

        # No need to calculate gradients, inference mode
        with torch.no_grad():
            features = backbone(image.cuda())
            recon = model(features)

        # Compute the reconstruction error
        recon_error = ((features - recon) ** 2).mean(axis=1).unsqueeze(0)

        # Upscale the error map to the original image resolution
        segm_map = torch.nn.functional.interpolate(
            recon_error,
            size=(256, 256),
            mode='bilinear'
        )

        # Convert segmentation map to numpy for visualization
        segm_map_np = segm_map.squeeze().cpu().numpy()

        # Normalize segm_map_np between 0 and 1
        segm_map_np = (segm_map_np - segm_map_np.min()) / (segm_map_np.max() - segm_map_np.min())

        # Convert the image tensor back to a displayable format
        image_np = image.squeeze().cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format
        image_np = (image_np * 255).astype(np.uint8)  # Convert to uint8 type

        # Plot the images
        plt.figure(figsize=(15, 8))
        
        # Show the original image
        plt.subplot(1, 2 + int(mask), 1)
        plt.imshow(image_np)
        plt.title('Original Image')

        # If mask is true, show the mask image
        if mask:
            plt.subplot(1, 3, 2)
            plt.imshow(mask_img, cmap='gray')
            plt.title('Ground Truth')

        # Show the heat map
        plt.subplot(1, 2 + int(mask), 2 + int(mask))
        plt.imshow(segm_map_np, cmap='inferno')
        plt.title('Heat Map')
        plt.tight_layout()
        plt.show()

        # Save the heatmap as an image in the output folder
        heatmap_output_path = os.path.join(output_folder, f"{loss}/{norm}/heatmap_{num_epochs}_{idx}.png")
        plt.imsave(heatmap_output_path, segm_map_np, cmap='inferno')
        print(f"Saved heatmap: {heatmap_output_path}")

def get_mask_path(image_path, mask_folder_name='cancer_masks', mask_prefix='cancer_'):
    """
    Generate the corresponding mask path for a given image path.

    Parameters:
    - image_path (str): The path to the original image.
    - mask_folder_name (str): The folder name where masks are stored.
    - mask_prefix (str): The prefix to be added to the image name to get the mask name.

    Returns:
    - mask_path (str): The path to the corresponding mask.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(image_path)))  # Up two levels
    mask_dir = os.path.join(base_dir, mask_folder_name)

    # Extract the image name and create the mask name
    image_name = os.path.basename(image_path)
    mask_name = mask_prefix + image_name

    # Full path to the mask
    mask_path = os.path.join(mask_dir, mask_name)
    return mask_path

def decision_function(segm_map):  
    """
    Compute the decision function by taking the mean of sorted flattened segmentation map.

    Args:
        segm_map (tensor): Segmentation map from which to compute anomaly score.

    Returns:
        tensor: Computed anomaly scores.
    """
    mean_values = []

    for map in segm_map:
        # Flatten the tensor
        flattened_tensor = map.reshape(-1)

        # Sort the flattened tensor along the feature dimension (descending order)
        sorted_tensor, _ = torch.sort(flattened_tensor, descending=True)

        mean_value = sorted_tensor.mean()

        mean_values.append(mean_value)

    return torch.stack(mean_values)


def evaluate_model(backbone, model, train_loader, healthy_test_csv, cancer_test_csv, transform, output_csv, num_epochs, loss="mse", norm="min", output_folder="."):
    """
    Function to evaluate the model for anomaly detection and save ROC metrics and plots as PDFs.
    
    Args:
        backbone (torch.nn.Module): Feature extractor (backbone) used in the model.
        model (torch.nn.Module): The model to perform the reconstruction.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset (healthy data).
        healthy_test_csv (str): Path to the CSV file containing healthy test image paths.
        cancer_test_csv (str): Path to the CSV file containing cancer test image paths.
        transform (torchvision.transforms.Compose): Image transformations to be applied.
        output_csv (str): Path to save the output ROC metrics.
        num_epochs (int): Number of epochs for model evaluation.
        loss (str): Loss function type, either "mse" or "euc".
        output_folder (str): Folder where the plotted PDF images will be saved.
        
    Returns:
        None
    """

    # Ensure output directories exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # ---------------- Training phase: Compute reconstruction errors --------------------
    model.eval()
    backbone.eval()

    RECON_ERROR = []
    
    # Loop through training data - Compute reconstruction error on healthy data
    for data in train_loader:
        with torch.no_grad():
            features = backbone(data.cuda()).squeeze()
            recon = model(features)
        if loss == "mse":
            segm_map = ((features - recon) ** 2).mean(axis=(1))[:, 3:-3, 3:-3]
        elif loss == "euc1":
            segm_map = torch.norm(features - recon, p=1, dim=1)[:, 3:-3, 3:-3]
        elif loss == "euc2":
            segm_map = torch.norm(features - recon, p=2, dim=1)[:, 3:-3, 3:-3]
        anomaly_score = decision_function(segm_map)
        RECON_ERROR.append(anomaly_score)
    
    RECON_ERROR = torch.cat(RECON_ERROR).cpu().numpy()
    
    # Determine the best threshold based on training data (3-sigma)
    #best_threshold_3sigma = np.mean(RECON_ERROR) + 3 * np.std(RECON_ERROR)

    # Plot the distribution of reconstruction errors using KDE (PDF plot)
    plt.figure(figsize=(6, 4))
    sns.kdeplot(RECON_ERROR, fill=True, color="green", label=r'$D_{in}$')
    #plt.axvline(x=best_threshold_3sigma, color='r', linestyle='--', label=r'Threshold ($\mu + 3 \sigma$)')
    plt.legend(loc='upper right')
    plt.xlabel('Anomaly Scores')
    plt.ylabel('PDF')
    plt.title('PDF of Anomaly Scores (Training Data)')
    
    # Save the first plot (PDF1)
    pdf1_output_path = os.path.join(output_folder, f"{loss}/{norm}/PDF/train/pdf_{num_epochs}_train.svg")
    os.makedirs(os.path.dirname(pdf1_output_path), exist_ok=True)  # Ensure the directory exists
    plt.savefig(pdf1_output_path)
    plt.show()

    # ---------------- Test phase: Process healthy and cancerous test data --------------------

    y_true_test = []
    y_score_test = []

    # Process healthy test images (label = 0)
    healthy_test_df = pd.read_csv(healthy_test_csv)
    for idx, row in healthy_test_df.iterrows():
        image_path = row['files']  # Path to the test image
        test_image = transform(Image.open(image_path)).cuda().unsqueeze(0)
        
        with torch.no_grad():
            features = backbone(test_image)
            recon = model(features)
        
        if loss == "mse":
            segm_map = ((features - recon) ** 2).mean(axis=(1))[:, 3:-3, 3:-3]
        elif loss == "euc1":
            segm_map = torch.norm(features - recon, p=1, dim=1)[:, 3:-3, 3:-3]
        elif loss == "euc2":
            segm_map = torch.norm(features - recon, p=2, dim=1)[:, 3:-3, 3:-3]

        y_score_image = decision_function(segm_map=segm_map)

        y_true_test.append(0)  # Healthy label is 0
        y_score_test.append(y_score_image.cpu().numpy().mean())  # Ensure scalar value

    # Process cancerous test images (label = 1)
    cancer_test_df = pd.read_csv(cancer_test_csv)
    for idx, row in cancer_test_df.iterrows():
        image_path = row['files']  # Path to the test image
        test_image = transform(Image.open(image_path)).cuda().unsqueeze(0)
        
        with torch.no_grad():
            features = backbone(test_image)
            recon = model(features)
        
        if loss == "mse":
            segm_map = ((features - recon) ** 2).mean(axis=(1))[:, 3:-3, 3:-3]
        elif loss == "euc1":
            segm_map = torch.norm(features - recon, p=1, dim=1)[:, 3:-3, 3:-3]
        elif loss == "euc2":
            segm_map = torch.norm(features - recon, p=2, dim=1)[:, 3:-3, 3:-3]

        y_score_image = decision_function(segm_map=segm_map)

        y_true_test.append(1)  # Cancerous label is 1
        y_score_test.append(y_score_image.cpu().numpy().mean())  # Ensure scalar value

    y_true_test = np.array(y_true_test)
    y_score_test = np.array(y_score_test)

    # ---------------- Calculate thresholds based on test data --------------------

    fpr, tpr, roc_thresholds = roc_curve(y_true_test, y_score_test)

    # 1. Youden's J statistic
    J = tpr - fpr
    best_threshold_j = roc_thresholds[np.argmax(J)]

    # 2. F1 score maximization
    f1_scores = [f1_score(y_true_test, y_score_test >= threshold) for threshold in roc_thresholds]
    best_threshold_f1 = roc_thresholds[np.argmax(f1_scores)]
    
    # 3. FPR at 95% TPR (FPR95)
    target_tpr = 0.95
    idx = np.argmin(np.abs(tpr - target_tpr))
    best_threshold_fpr95 = roc_thresholds[idx]

    print(f"Best threshold based on Youden's J: {best_threshold_j}")
    #print(f"Best threshold based on F1 score: {best_threshold_f1}")
    #print(f"Threshold for FPR at 95% TPR: {best_threshold_fpr95}")

    # Generate predictions for each threshold
    y_pred_j = (y_score_test >= best_threshold_j).astype(int)
    y_pred_f1 = (y_score_test >= best_threshold_f1).astype(int)
    y_pred_fpr95 = (y_score_test >= best_threshold_fpr95).astype(int)
    #y_pred_3sigma = (y_score_test >= best_threshold_3sigma).astype(int)

    # Calculate AUC-ROC scores for each threshold
    auc_roc_j = roc_auc_score(y_true_test, y_pred_j)
    auc_roc_f1 = roc_auc_score(y_true_test, y_pred_f1)
    auc_roc_fpr95 = roc_auc_score(y_true_test, y_pred_fpr95)
    #auc_roc_3sigma = roc_auc_score(y_true_test, y_pred_3sigma)

    print(f"AUC-ROC Score (Youden's J): {auc_roc_j}")
    print(f"AUC-ROC Score (F1): {auc_roc_f1}")
    print(f"AUC-ROC Score (FPR95): {auc_roc_fpr95}")
    #print(f"AUC-ROC Score (3-sigma): {auc_roc_3sigma}")

    # ---------------- Save ROC metrics to CSV --------------------

    roc_metrics = {
        'y_true': y_true_test.tolist(),
        'y_score': y_score_test.tolist(),
        'y_pred_j': y_pred_j.tolist(),
        'y_pred_f1': y_pred_f1.tolist(),
        'y_pred_fpr95': y_pred_fpr95.tolist(),
        #'y_pred_3sigma': y_pred_3sigma.tolist(),
        'roc_auc_j': auc_roc_j,
        'roc_auc_f1': auc_roc_f1,
        'roc_auc_fpr95': auc_roc_fpr95,
        # 'roc_auc_3sigma': auc_roc_3sigma,
        'best_threshold_j': best_threshold_j,  # Save best threshold for Youden's J
        'best_threshold_f1': best_threshold_f1,  # Save best threshold for F1 score
        'best_threshold_fpr95': best_threshold_fpr95,  # Save best threshold for FPR95
        #'best_threshold_3sigma': best_threshold_3sigma  # Save 3-sigma threshold
    }

    roc_metrics_df = pd.DataFrame(roc_metrics)
    roc_metrics_df.to_csv(output_csv, index=False)
    print(f"ROC metrics saved to {output_csv}")

    # ---------------- Plot distribution of scores with the thresholds --------------------

    plt.figure(figsize=(6, 4))
    sns.kdeplot(y_score_test[y_true_test == 0].squeeze(), fill=True, color="green", label=r'$D_{in}$')  # Healthy - light green
    sns.kdeplot(y_score_test[y_true_test == 1].squeeze(), fill=True, color="red", label=r'$D_{out}$')  # Cancerous - red
    plt.axvline(x=best_threshold_j, color='blue', linestyle='-', label=r"Best Threshold (Youden's J)")
    #plt.axvline(x=best_threshold_f1, color='orange', linestyle='-', label=r'Best Threshold (F1)')
    #plt.axvline(x=best_threshold_fpr95, color='purple', linestyle='-', label=r'FPR95 Threshold')
    #plt.axvline(x=best_threshold_3sigma, color='red', linestyle='--', label=r'Threshold ($\mu + 3 \sigma$)')
    plt.legend(loc='upper right')
    plt.xlabel('Anomaly Scores')
    plt.ylabel('PDF')
    plt.title('PDF of Scores (Test Data)')
    
    # Save the second plot (PDF2)
    pdf2_output_path = os.path.join(output_folder, f"{loss}/{norm}/PDF/test/pdf_{num_epochs}_test.svg")
    os.makedirs(os.path.dirname(pdf2_output_path), exist_ok=True)  # Ensure the directory exists
    plt.savefig(pdf2_output_path)
    plt.show()
    

def run_vgg(BS, num_epochs, latent_dim, patch, transform_minmax, transform_zscore, train_csv_path, val_csv_path, healthy_test_csv, cancer_test_csv):
    """
    Run the experiment by training and evaluating models based on VGG features.
    
    Args:
        BS (int): Batch size.
        num_epochs (int): Number of epochs for training.
        latent_dim (int): Latent space dimension.
        patch (str): Patch type (e.g., 'full', 'partial').
        transform_minmax: Transformation to apply for min-max normalization.
        transform_zscore: Transformation to apply for z-score normalization.
        train_csv_path (Path): Path to the training CSV file.
        val_csv_path (Path): Path to the validation CSV file.
        healthy_test_csv (Path): Path to the healthy test CSV file.
        cancer_test_csv (Path): Path to the cancer test CSV file.
    
    Returns:
        None
    """
    
    # Create datasets from the CSV files
    train_dataset_min = ImageDatasetFromCSV(csv_file=train_csv_path, transform=transform_minmax)
    val_dataset_min = ImageDatasetFromCSV(csv_file=val_csv_path, transform=transform_minmax)

    # train_dataset_zs = ImageDatasetFromCSV(csv_file=train_csv_path, transform=transform_zscore)
    # val_dataset_zs = ImageDatasetFromCSV(csv_file=val_csv_path, transform=transform_zscore)

    # Set masking based on the patch type
    if patch == 'full':
        masking = False
    else:
        masking = True
    
    # Set image output folder based on the patch type
    image_output_folder = f"Result_images/VGG/{patch}"

    # Create data loaders for training and validation datasets
    train_loader_min = DataLoader(train_dataset_min, batch_size=BS, shuffle=True)
    val_loader_min = DataLoader(val_dataset_min, batch_size=BS, shuffle=True)

    # train_loader_zs = DataLoader(train_dataset_zs, batch_size=BS, shuffle=True)
    # val_loader_zs = DataLoader(val_dataset_zs, batch_size=BS, shuffle=True)

    # Initialize the backbone (VGG feature extractor)
    backbone = vgg_feature()

    # Initialize the model
    model = VGG_CAE(in_channels=1024, latent_dim=latent_dim).cuda()
    backbone.cuda()

    # Define loss functions
    criterion_mse = torch.nn.MSELoss()
    criterion_euc1 = euclidean_loss1
    criterion_euc2 = euclidean_loss2

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training and processing for different configurations
    vgg_mse_min = train_vgg(
        model, backbone, criterion_mse, optimizer, BS, train_dataset_min, val_dataset_min,
        num_epochs=num_epochs, output_csv=f'CSVs/Loss/VGG/{patch}/mse/MIN_{num_epochs}.csv',
        output_model=f"Models/VGG/{patch}/mse/MIN_{num_epochs}.pth"
    )
    process_images_from_csv(backbone, vgg_mse_min, transform_minmax, cancer_test_csv, image_output_folder, num_epochs, loss="mse", norm="min", mask=masking)

    evaluate_model(
        backbone, vgg_mse_min, train_loader_min, healthy_test_csv, cancer_test_csv,
        transform_minmax, output_csv=f"CSVs/ROC/VGG/{patch}/mse/MIN_{num_epochs}.csv",
        num_epochs=num_epochs, loss="mse", norm="min", output_folder=image_output_folder
    )

    vgg_euc1_min = train_vgg(
        model, backbone, criterion_euc1, optimizer, BS, train_dataset_min, val_dataset_min,
        num_epochs=num_epochs, output_csv=f'CSVs/Loss/VGG/{patch}/euc1/MIN_{num_epochs}.csv',
        output_model=f"Models/VGG/{patch}/euc1/MIN_{num_epochs}.pth"
    )
    process_images_from_csv(backbone, vgg_euc1_min, transform_minmax, cancer_test_csv, image_output_folder, num_epochs, loss="euc1", norm="min", mask=masking)

    evaluate_model(
        backbone, vgg_euc1_min, train_loader_min, healthy_test_csv, cancer_test_csv,
        transform_minmax, output_csv=f"CSVs/ROC/VGG/{patch}/euc1/MIN_{num_epochs}.csv",
        num_epochs=num_epochs, loss="euc1", norm="min", output_folder=image_output_folder
    )
    
    vgg_euc2_min = train_vgg(
        model, backbone, criterion_euc2, optimizer, BS, train_dataset_min, val_dataset_min,
        num_epochs=num_epochs, output_csv=f'CSVs/Loss/VGG/{patch}/euc2/MIN_{num_epochs}.csv',
        output_model=f"Models/VGG/{patch}/euc2/MIN_{num_epochs}.pth"
    )
    process_images_from_csv(backbone, vgg_euc2_min, transform_minmax, cancer_test_csv, image_output_folder, num_epochs, loss="euc2", norm="min", mask=masking)

    evaluate_model(
        backbone, vgg_euc2_min, train_loader_min, healthy_test_csv, cancer_test_csv,
        transform_minmax, output_csv=f"CSVs/ROC/VGG/{patch}/euc2/MIN_{num_epochs}.csv",
        num_epochs=num_epochs, loss="euc2", norm="min", output_folder=image_output_folder
    )

    # vgg_mse_zs = train_vgg(
    #     model, backbone, criterion_mse, optimizer, BS, train_dataset_zs, val_dataset_zs,
    #     num_epochs=num_epochs, output_csv=f'CSVs/Loss/VGG/{patch}/mse/ZS_{num_epochs}.csv',
    #     output_model=f"Models/VGG/{patch}/mse/ZS_{num_epochs}.pth"
    # )
    # process_images_from_csv(backbone, vgg_mse_zs, transform_zscore, cancer_test_csv, image_output_folder, num_epochs, loss="mse", norm="zs", mask=masking)

    # evaluate_model(
    #     backbone, vgg_mse_zs, train_loader_zs, healthy_test_csv, cancer_test_csv,
    #     transform_zscore, output_csv=f"CSVs/ROC/VGG/{patch}/mse/ZS_{num_epochs}.csv",
    #     num_epochs=num_epochs, loss="mse", norm="zs", output_folder=image_output_folder
    # )

    # vgg_euc1_zs = train_vgg(
    #     model, backbone, criterion_euc1, optimizer, BS, train_dataset_zs, val_dataset_zs,
    #     num_epochs=num_epochs, output_csv=f'CSVs/Loss/VGG/{patch}/euc1/ZS_{num_epochs}.csv',
    #     output_model=f"Models/VGG/{patch}/euc1/ZS_{num_epochs}.pth"
    # )
    # process_images_from_csv(backbone, vgg_euc1_zs, transform_zscore, cancer_test_csv, image_output_folder, num_epochs, loss="euc1", norm="zs", mask=masking)

    # evaluate_model(
    #     backbone, vgg_euc1_zs, train_loader_zs, healthy_test_csv, cancer_test_csv,
    #     transform_zscore, output_csv=f"CSVs/ROC/VGG/{patch}/euc1/ZS_{num_epochs}.csv",
    #     num_epochs=num_epochs, loss="euc1", norm="zs", output_folder=image_output_folder
    # )
    
    # vgg_euc2_zs = train_vgg(
    #     model, backbone, criterion_euc2, optimizer, BS, train_dataset_zs, val_dataset_zs,
    #     num_epochs=num_epochs, output_csv=f'CSVs/Loss/VGG/{patch}/euc2/ZS_{num_epochs}.csv',
    #     output_model=f"Models/VGG/{patch}/euc2/ZS_{num_epochs}.pth"
    # )
    # process_images_from_csv(backbone, vgg_euc2_zs, transform_zscore, cancer_test_csv, image_output_folder, num_epochs, loss="euc2", norm="zs", mask=masking)

    # evaluate_model(
    #     backbone, vgg_euc2_zs, train_loader_zs, healthy_test_csv, cancer_test_csv,
    #     transform_zscore, output_csv=f"CSVs/ROC/VGG/{patch}/euc2/ZS_{num_epochs}.csv",
    #     num_epochs=num_epochs, loss="euc2", norm="zs", output_folder=image_output_folder
    # )