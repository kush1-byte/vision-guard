"""
End-to-end predictor used by the API.

Loads a trained checkpoint once, then for each image runs:
  preprocess -> model -> sigmoid/softmax -> safety layer (+ optional Grad-CAM).

Keeps a single model in memory so the API doesn't reload weights per request.
"""
from __future__ import annotations

import numpy as np
import torch

from config import CFG
from data.label_map import TARGET_DISEASES, SEVERITY_CLASSES
from explainability.gradcam import generate_gradcam
from inference.safety import apply_safety_logic, assess_quality
from models.vision_guard_net import VisionGuardNet, get_target_layers
from preprocessing.fundus_preprocess import preprocess_fundus
from preprocessing.transforms import val_transforms


class VisionGuardPredictor:
    def __init__(self, checkpoint_path, device=None):
        self.device = device or (CFG.device if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.backbone = ckpt.get("backbone", CFG.backbone)

        self.model = VisionGuardNet(self.backbone, CFG.num_diseases,
                                    CFG.num_severity, pretrained=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()

        self.transform = val_transforms(CFG.image_size)
        self.target_layers = get_target_layers(self.model, self.backbone)
        self.trained_diseases = self._resolve_trained_diseases(ckpt)
        self.reported_diseases = self._resolve_reported_diseases()

    def _resolve_reported_diseases(self):
        """The subset of trained heads we actually score and display.

        A head can be trained and still not be ready to report: the four
        non-DR heads see only ODIR's partial labels, which is enough to move
        the weights but not enough to fix a threshold we would stand behind.
        CFG.reported_diseases narrows the trained set to the ones that are.
        Returns None only when nothing is being withheld.
        """
        scope = CFG.reported_diseases
        if not scope:
            return self.trained_diseases
        trained = self.trained_diseases
        if trained is None:
            return {d for d in TARGET_DISEASES if d in scope}
        return {d for d in trained if d in scope}

    @staticmethod
    def _resolve_trained_diseases(ckpt):
        """Which disease heads actually received gradient during training.

        Newer checkpoints record this directly. Older ones predate the field,
        so fall back to the training manifest: a disease whose column is -1 on
        every row was never supervised, and its head is still at its random
        init. Returns None only if we genuinely cannot tell, which keeps the
        original "trust every head" behaviour.
        """
        recorded = ckpt.get("trained_diseases")
        if recorded:
            return set(recorded)

        try:
            import pandas as pd
            df = pd.read_csv(CFG.manifest_train, usecols=lambda c: c in TARGET_DISEASES)
        except Exception:
            return None
        known = {d for d in TARGET_DISEASES if d in df.columns and (df[d] != -1).any()}
        return known or None

    def _prepare(self, rgb_image):
        """rgb_image: uint8 (H,W,3). Returns (input_tensor, processed_uint8)."""
        processed = preprocess_fundus(rgb_image, CFG.image_size,
                                      CFG.use_clahe, CFG.use_denoise)
        tensor = self.transform(image=processed)["image"].unsqueeze(0)  # (1,3,H,W)
        return tensor.to(self.device), processed

    @torch.no_grad()
    def predict(self, rgb_image):
        quality = assess_quality(rgb_image)
        input_tensor, _ = self._prepare(rgb_image)

        out = self.model(input_tensor)
        disease_probs = torch.sigmoid(out["disease_logits"])[0].cpu().numpy()
        severity_probs = torch.softmax(out["severity_logits"], dim=1)[0].cpu().numpy()

        probs = {d: float(disease_probs[i]) for i, d in enumerate(TARGET_DISEASES)}
        severity_idx = int(np.argmax(severity_probs))

        safety = apply_safety_logic(probs, quality, self.reported_diseases)

        held = sorted(d for d in TARGET_DISEASES
                      if d not in (self.reported_diseases or TARGET_DISEASES)
                      and (self.trained_diseases is None
                           or d in self.trained_diseases))

        return {
            "disease_probabilities": probs,
            "trained_diseases": (sorted(self.trained_diseases)
                                 if self.trained_diseases else None),
            "reported_diseases": (sorted(self.reported_diseases)
                                  if self.reported_diseases else None),
            "held_diseases": held,
            "dr_severity": {
                "stage": SEVERITY_CLASSES[severity_idx],
                "confidence": float(severity_probs[severity_idx]),
            },
            "quality": quality,
            "safety": safety,
        }

    def explain(self, rgb_image, disease_name):
        """Grad-CAM overlay (uint8 RGB) for one disease. Enables grad on purpose."""
        idx = TARGET_DISEASES.index(disease_name)
        input_tensor, processed = self._prepare(rgb_image)
        rgb_float = processed.astype(np.float32) / 255.0
        overlay, _ = generate_gradcam(self.model, input_tensor, idx,
                                      rgb_float, self.target_layers)
        return overlay
