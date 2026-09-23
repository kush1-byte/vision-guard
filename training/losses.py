"""
Loss function that handles partial labels.

Two things are combined:

  1. Multi-label disease loss: BCEWithLogitsLoss, but MASKED. For each image,
     only the diseases that are actually known (mask == 1) contribute; unknown
     ones are ignored. This is what lets us mix ODIR (knows all 5) with APTOS
     (knows only DR) in one training run.

  2. Severity loss: CrossEntropyLoss with ignore_index=-1, so images without a
     severity label simply don't contribute to it.

`pos_weight` up-weights the rare positive class for each disease, which matters
a lot here because most screening images are normal.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from data.label_map import UNKNOWN_SEVERITY


class VisionGuardLoss(nn.Module):
    def __init__(self, pos_weight=None, severity_weight=0.5):
        super().__init__()
        # reduction="none" so we can apply the per-element mask ourselves.
        self.bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
        self.ce = nn.CrossEntropyLoss(ignore_index=UNKNOWN_SEVERITY)
        self.severity_weight = severity_weight

    def forward(self, outputs, targets):
        logits = outputs["disease_logits"]
        labels = targets["disease_labels"]
        mask = targets["disease_mask"]

        # per-element BCE, then keep only known entries
        bce_all = self.bce(logits, labels)             # (B, 5)
        denom = mask.sum().clamp(min=1.0)              # avoid divide-by-zero
        disease_loss = (bce_all * mask).sum() / denom

        # CrossEntropy already ignores severity == -1. But if a whole batch has
        # NO known severity, it returns nan (0/0); guard against that.
        severity_loss = self.ce(outputs["severity_logits"], targets["severity"])
        if torch.isnan(severity_loss):
            severity_loss = torch.zeros((), device=logits.device)

        total = disease_loss + self.severity_weight * severity_loss
        # detach before float(): these are for logging only, and converting a
        # grad-tracking tensor to a scalar warns about exactly that confusion.
        parts = {"disease": disease_loss.detach().item(),
                 "severity": severity_loss.detach().item(),
                 "total": total.detach().item()}
        return total, parts
