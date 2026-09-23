"""
PyTorch Dataset.

It reads a MANIFEST csv (one unified table built from all datasets) with columns:

    image_path,                          # absolute or relative path to the image
    DR, Glaucoma, Cataract, AMD, Hypertensive_Retinopathy,   # 0 / 1 / -1(unknown)
    severity                             # 0..4, or -1 if unknown

Building that manifest is done once by data/prepare_manifest.py.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from preprocessing.fundus_preprocess import preprocess_fundus
from data.label_map import TARGET_DISEASES, UNKNOWN_SEVERITY


class FundusDataset(Dataset):
    def __init__(self, manifest_csv, transform, image_size=512,
                 use_clahe=True, use_denoise=False, preprocess=True):
        self.df = pd.read_csv(manifest_csv)
        self.transform = transform
        self.image_size = image_size
        self.use_clahe = use_clahe
        self.use_denoise = use_denoise
        self.preprocess = preprocess  # set False if manifest already points at cached images

    def __len__(self):
        return len(self.df)

    def _load_rgb(self, path):
        # cv2 loads BGR; convert to RGB so every downstream step is consistent.
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self._load_rgb(row["image_path"])

        if self.preprocess:
            img = preprocess_fundus(img, self.image_size,
                                    self.use_clahe, self.use_denoise)

        img = self.transform(image=img)["image"]   # -> CHW float tensor

        # --- labels ---
        # value of -1 in the manifest means "unknown for this image".
        raw = np.array([row[d] for d in TARGET_DISEASES], dtype=np.float32)
        mask = (raw != -1).astype(np.float32)       # 1 where known
        labels = np.where(raw == -1, 0.0, raw).astype(np.float32)  # zero the unknowns

        severity = int(row.get("severity", UNKNOWN_SEVERITY))

        return {
            "image": img,
            "disease_labels": torch.from_numpy(labels),
            "disease_mask": torch.from_numpy(mask),
            "severity": torch.tensor(severity, dtype=torch.long),
        }
