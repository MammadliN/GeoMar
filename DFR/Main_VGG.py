# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 16:24:31 2024

@author: mammadli
"""

from VGG.VGG_Run import run_vgg
from Norm.MinMax import transform_minmax
from Norm.Zscore import transform_zscore

from pathlib import Path


train_full_path = Path(r"/home/ubuntu/Thesis/MorphAEus-main/data/CXR/full_breast_id/cancer_false_train.csv")
val_full_path = Path(r"/home/ubuntu/Thesis/MorphAEus-main/data/CXR/full_breast_id/cancer_false_val.csv")
healthy_test_full = Path(r"/home/ubuntu/Thesis/MorphAEus-main/data/CXR/full_breast_id/cancer_false_test.csv")
cancer_test_full = Path(r"/home/ubuntu/Thesis/MorphAEus-main/data/CXR/full_breast_id/cancer_true_test.csv")

run_vgg(
    BS=128,
    num_epochs=100,
    latent_dim=128,
    patch='full',
    transform_minmax=transform_minmax,
    transform_zscore=transform_zscore,
    train_csv_path=train_full_path,
    val_csv_path=val_full_path,
    healthy_test_csv=healthy_test_full,
    cancer_test_csv=cancer_test_full
)


train_side = Path(r"/home/ubuntu/Thesis/MorphAEus-main/data/CXR/left_right_id/cancer_false_train.csv")
val_side = Path(r"/home/ubuntu/Thesis/MorphAEus-main/data/CXR/left_right_id/cancer_false_val.csv")
healthy_test_side = Path(r"/home/ubuntu/Thesis/MorphAEus-main/data/CXR/left_right_id/cancer_false_test.csv")
cancer_test_side = Path(r"/home/ubuntu/Thesis/MorphAEus-main/data/CXR/left_right_id/cancer_true_test.csv")

run_vgg(
    BS=128,
    num_epochs=100,
    latent_dim=128,
    patch='side',
    transform_minmax=transform_minmax,
    transform_zscore=transform_zscore,
    train_csv_path=train_side,
    val_csv_path=val_side,
    healthy_test_csv=healthy_test_side,
    cancer_test_csv=cancer_test_side
)


train_masked = Path(r"/home/ubuntu/Thesis/MorphAEus-main/data/CXR/masked_id/cancer_false_train.csv")
val_masked = Path(r"/home/ubuntu/Thesis/MorphAEus-main/data/CXR/masked_id/cancer_false_val.csv")
healthy_test_masked = Path(r"/home/ubuntu/Thesis/MorphAEus-main/data/CXR/masked_id/cancer_false_test.csv")
cancer_test_masked = Path(r"/home/ubuntu/Thesis/MorphAEus-main/data/CXR/masked_id/cancer_true_test.csv")

run_vgg(
    BS=128,
    num_epochs=100,
    latent_dim=128,
    patch='masked',
    transform_minmax=transform_minmax,
    transform_zscore=transform_zscore,
    train_csv_path=train_masked,
    val_csv_path=val_masked,
    healthy_test_csv=healthy_test_masked,
    cancer_test_csv=cancer_test_masked
)