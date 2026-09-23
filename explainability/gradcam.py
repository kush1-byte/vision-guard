"""
Grad-CAM explainability.

Given an image and a disease, this produces a heatmap showing WHERE in the retina
the model "looked" to make that prediction — e.g. it should light up over
microaneurysms/haemorrhages for DR, or the optic disc for glaucoma. This is what
lets a clinician sanity-check the model instead of trusting a black box.

pytorch-grad-cam expects a model that returns a plain tensor of class scores, but
ours returns a dict. So we wrap it to expose just the disease logits.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class _DiseaseLogitWrapper(nn.Module):
    """Exposes model(x) -> disease_logits so Grad-CAM can use it."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["disease_logits"]


def generate_gradcam(model, input_tensor, target_disease_idx,
                     rgb_image_float, target_layers):
    """
    input_tensor      : (1, 3, H, W) normalized tensor (the model's input)
    target_disease_idx: which of the 5 diseases to explain (0..4)
    rgb_image_float   : the ORIGINAL rgb image as float in [0, 1], shape (H, W, 3),
                        used only as the background to draw the heatmap on
    target_layers     : from models.get_target_layers(...)

    Returns (overlay_uint8, raw_heatmap_float).
    """
    wrapper = _DiseaseLogitWrapper(model).eval()
    cam = GradCAM(model=wrapper, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_disease_idx)]

    grayscale = cam(input_tensor=input_tensor, targets=targets)[0]   # (H, W) in [0,1]
    overlay = show_cam_on_image(rgb_image_float, grayscale, use_rgb=True)
    return overlay, grayscale
