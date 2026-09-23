"""
Metrics for a multi-label screening model.

AUROC (area under the ROC curve) is the standard metric here because it doesn't
depend on a chosen threshold and handles class imbalance sensibly. We compute it
per disease and then average (macro-AUROC) as the single "how good is the model"
number to pick the best checkpoint.

Only KNOWN labels (mask == 1) are used for each disease.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from data.label_map import TARGET_DISEASES


def multilabel_auroc(probs: np.ndarray, labels: np.ndarray, mask: np.ndarray):
    """probs/labels/mask: arrays of shape (N, 5).

    Returns a dict {disease: auroc} plus 'macro'. A disease with only one class
    present in the (masked) set can't have an AUROC, so it's reported as None.
    """
    results = {}
    valid_scores = []
    for i, disease in enumerate(TARGET_DISEASES):
        m = mask[:, i] == 1
        y_true = labels[m, i]
        y_score = probs[m, i]
        if len(np.unique(y_true)) < 2:      # need both a 0 and a 1 to score
            results[disease] = None
            continue
        auc = roc_auc_score(y_true, y_score)
        results[disease] = float(auc)
        valid_scores.append(auc)

    results["macro"] = float(np.mean(valid_scores)) if valid_scores else 0.0
    return results
