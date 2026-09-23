"""
Unified label schema.

The hardest part of a multi-disease, multi-dataset model is that every dataset
labels things differently. ODIR-5K, MuReD and APTOS do NOT agree on columns.
So we define ONE target schema, and give each dataset a small mapping into it.

Key idea: PARTIAL LABELS.
  - ODIR-5K and MuReD tell us which diseases are present (multi-label), but say
    nothing about DR *severity*.
  - APTOS/MESSIDOR tell us DR severity (0-4), but nothing about the other 4
    diseases.

So for any given image, some labels are KNOWN and some are UNKNOWN. We represent
unknown disease labels with a mask (so the loss ignores them) and unknown
severity with -1 (CrossEntropyLoss ignores that with ignore_index=-1).
"""
from __future__ import annotations

# The 5 diseases the model predicts (order is fixed — the model's 5 outputs
# line up with this list).
TARGET_DISEASES = [
    "DR",                       # Diabetic Retinopathy (presence)
    "Glaucoma",
    "Cataract",
    "AMD",                      # Age-related Macular Degeneration
    "Hypertensive_Retinopathy",
]
DISEASE_TO_IDX = {name: i for i, name in enumerate(TARGET_DISEASES)}

# DR severity grades (APTOS 2019 / MESSIDOR grading scale).
SEVERITY_CLASSES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative"]

UNKNOWN_SEVERITY = -1   # ignore_index used by the severity loss

# --- How each source dataset's columns map into TARGET_DISEASES ---
# You only use these while BUILDING the manifest (see data/prepare_manifest.py).
# The value is the column name in that dataset's own label file.

ODIR_COLUMN_MAP = {
    # ODIR label columns -> our disease name
    "D": "DR",            # 'D' = diabetes/diabetic retinopathy
    "G": "Glaucoma",
    "C": "Cataract",
    "A": "AMD",
    "H": "Hypertensive_Retinopathy",
    # ODIR also has N (normal), M (myopia), O (other) which we don't target.
}

MURED_COLUMN_MAP = {
    # MuReD uses full disease names; these are the ones we care about.
    "DR": "DR",
    "ARMD": "AMD",           # MuReD calls AMD "ARMD"
    "MH": None,              # media haze -> not a target
    "Glaucoma": "Glaucoma",
    "Cataract": "Cataract",
    # MuReD does not have a dedicated hypertensive-retinopathy column in all
    # releases; leave Hypertensive_Retinopathy UNKNOWN for MuReD rows.
}

# APTOS gives a single 'diagnosis' column: 0..4 severity.
# From severity we can ALSO derive DR presence (severity > 0 -> DR present).
APTOS_SEVERITY_COLUMN = "diagnosis"


def blank_disease_vector():
    """Return (labels, mask) with everything unknown."""
    labels = [0.0] * len(TARGET_DISEASES)
    mask = [0.0] * len(TARGET_DISEASES)   # 0 = unknown, 1 = known
    return labels, mask
