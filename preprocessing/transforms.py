"""
Runtime augmentations (Albumentations).

These run every time an image is loaded during training, so the model sees a
slightly different version each epoch. Validation gets only normalization — no
random changes, so results are reproducible.

Normalization uses the ImageNet mean/std because our backbones were pre-trained
on ImageNet and expect inputs in that distribution.
"""
from __future__ import annotations

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transforms(size: int = 512) -> A.Compose:
    return A.Compose([
        A.Resize(size, size),                       # safety net if not pre-sized
        A.Rotate(limit=25, border_mode=cv2.BORDER_CONSTANT, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1,
                                   contrast_limit=0.1, p=0.4),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2(),                               # HWC uint8 -> CHW float tensor
    ])


def val_transforms(size: int = 512) -> A.Compose:
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2(),
    ])
