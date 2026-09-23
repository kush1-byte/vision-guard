"""
Deterministic fundus preprocessing (OpenCV).

These steps are the SAME every time for a given image, so they should be run
once and cached to disk (see notes in README) rather than repeated every epoch:

  1. auto-crop the black border around the circular fundus,
  2. (optional) denoise,
  3. CLAHE on the green channel to accentuate blood vessels,
  4. resize to a square.

The random augmentations (rotation, flip, ...) are a SEPARATE step and live in
transforms.py, because those must change every epoch.

IMPORTANT: every function here expects an RGB uint8 image (H, W, 3).
If you load with cv2.imread you get BGR — convert first with
cv2.cvtColor(img, cv2.COLOR_BGR2RGB). If you load with PIL, it's already RGB.
"""
from __future__ import annotations

import cv2
import numpy as np


def crop_dark_borders(image: np.ndarray, tol: int = 7) -> np.ndarray:
    """Crop away the near-black rectangle around the round fundus.

    We build a mask of "bright enough" pixels and crop to their bounding box.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    mask = gray > tol
    if mask.sum() == 0:            # totally black image -> nothing to crop
        return image

    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1    # +1 because slicing is exclusive at the end
    return image[y0:y1, x0:x1]


def clahe_green_channel(image: np.ndarray, clip_limit: float = 2.0,
                        tile_grid: int = 8) -> np.ndarray:
    """Apply CLAHE to the GREEN channel only, then recombine.

    In fundus photography the green channel has the strongest contrast between
    vessels and background, so enhancing it makes microaneurysms and haemorrhages
    stand out while leaving overall colour roughly intact.
    """
    r, g, b = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                            tileGridSize=(tile_grid, tile_grid))
    g_eq = clahe.apply(g)
    return cv2.merge([r, g_eq, b])


def denoise(image: np.ndarray, strength: int = 3) -> np.ndarray:
    """Remove sensor noise.

    fastNlMeans gives the cleanest result but is slow — only worth it as a
    one-time cached step. For a faster alternative use a bilateral filter
    (kept below, commented) which preserves edges.
    """
    return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
    # return cv2.bilateralFilter(image, d=5, sigmaColor=50, sigmaSpace=50)


def preprocess_fundus(image: np.ndarray,
                      size: int = 512,
                      use_clahe: bool = True,
                      use_denoise: bool = False,
                      clahe_clip_limit: float = 2.0,
                      clahe_tile_grid: int = 8) -> np.ndarray:
    """Full deterministic pipeline. Returns an RGB uint8 square image."""
    img = crop_dark_borders(image)
    if use_denoise:
        img = denoise(img)
    if use_clahe:
        img = clahe_green_channel(img, clahe_clip_limit, clahe_tile_grid)
    # INTER_AREA is the right choice when shrinking images.
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img
