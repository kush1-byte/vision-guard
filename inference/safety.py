"""
Safety layer.

Two jobs, both about knowing when NOT to trust the model:

  1. Image-quality gate: a screening model must not "confidently" read a photo
     that's too dark, blurry, or barely contains a retina. We compute a few cheap
     heuristics and flag bad scans. The cutoffs live in config.py and were
     measured with evaluation/calibrate_quality.py rather than guessed -- a gate
     that flags most ordinary images is the same as having no gate at all.

  2. Confidence thresholding: even on a good image, predictions in a grey zone
     (not clearly negative, not clearly positive) get flagged for human review.
     And because this is SCREENING, any positive prediction is routed to an
     ophthalmologist regardless — a false alarm is far cheaper than a missed
     disease.

Re-run the calibration if you change the image size, CLAHE settings, or camera.
"""
from __future__ import annotations

import cv2
import numpy as np

from config import CFG
from data.label_map import TARGET_DISEASES


def assess_quality(rgb_image: np.ndarray) -> dict:
    """rgb_image: uint8 (H, W, 3). Returns a quality report."""
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    brightness = float(gray.mean())
    # Variance of the Laplacian is a classic sharpness/blur measure: low = blurry.
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # Fraction of the frame that is actually retina (not black border).
    fundus_fraction = float((gray > 10).mean())

    issues = []
    if brightness < CFG.min_brightness or brightness > CFG.max_brightness:
        issues.append("poor_exposure")
    if sharpness < CFG.min_sharpness:
        issues.append("blurry_or_out_of_focus")
    if fundus_fraction < CFG.min_fundus_fraction:
        issues.append("insufficient_fundus_area")

    return {
        "brightness": round(brightness, 1),
        "sharpness": round(sharpness, 1),
        "fundus_fraction": round(fundus_fraction, 3),
        "issues": issues,
        "usable": len(issues) == 0,
    }


def apply_safety_logic(disease_probs: dict, quality: dict,
                       scored_diseases=None) -> dict:
    """
    disease_probs  : {disease_name: probability in [0,1]}
    quality        : output of assess_quality()
    scored_diseases: names allowed to influence the verdict. A name is left out
                     either because its head was never supervised (its output is
                     noise around 0.5) or because it is trained but not yet
                     validated to a reportable standard. Either way its number
                     must not move the decision. None means "trust them all".

    Returns a structured recommendation.
    """
    flags, reasons = [], []

    if not quality["usable"]:
        flags.append("POOR_QUALITY")
        reasons.append("image_quality_issues: " + ", ".join(quality["issues"]))

    # Only heads in scope get a vote. An untrained head emits ~0.5 forever,
    # which lands inside the uncertain band and above the referral threshold
    # half the time -- so without this filter every image is referred for
    # reasons that carry no information. A trained-but-unvalidated head is
    # excluded for a different reason: its number may well be meaningful, but
    # we have not fixed a threshold for it, so it must not drive a referral.
    scored = {d: p for d, p in disease_probs.items()
              if scored_diseases is None or d in scored_diseases}

    positives = [d for d, p in scored.items() if p >= CFG.refer_threshold]
    uncertain = [d for d, p in scored.items()
                 if CFG.uncertain_low < p < CFG.uncertain_high]

    if positives:
        flags.append("POSITIVE_SCREEN")
        reasons.append("above referral threshold: " + ", ".join(positives))
    if uncertain:
        flags.append("LOW_CONFIDENCE")
        reasons.append("in uncertain band: " + ", ".join(uncertain))

    # Decision: anything flagged -> a human must look. Otherwise auto-clear.
    if flags:
        decision = "MANDATORY_OPHTHALMOLOGIST_REVIEW"
    else:
        decision = "AUTO_CLEARED_NORMAL"

    ignored = [d for d in disease_probs if d not in scored]
    if ignored:
        reasons.append("not scored (outside the validated scope): "
                       + ", ".join(ignored))

    return {
        "decision": decision,
        "flags": flags,
        "reasons": reasons,
        "positive_diseases": positives,
        "uncertain_diseases": uncertain,
        "unscored_diseases": ignored,
    }
