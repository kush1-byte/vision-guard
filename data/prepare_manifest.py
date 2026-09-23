"""
Build the unified manifest from the three source datasets.

Run this ONCE after you download the datasets. It reads each dataset's own label
file, maps it into our 5-disease + severity schema, and writes a single csv that
FundusDataset can read.

You must adapt the small `load_*` helpers to wherever your files actually live —
the column names below match the public releases, but folder paths differ per
machine. Nothing here downloads data (the datasets need Kaggle / IEEE accounts);
download them yourself, then point these functions at the folders.

    ODIR-5K   : https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k
    MuReD     : published as "Multi-Label Retinal Diseases (MuReD) Dataset"
    APTOS 2019: https://www.kaggle.com/competitions/aptos2019-blindness-detection
"""
from __future__ import annotations

import os
import random

import pandas as pd

from data.label_map import (TARGET_DISEASES, ODIR_COLUMN_MAP, MURED_COLUMN_MAP,
                            APTOS_SEVERITY_COLUMN)

UNKNOWN = -1


def _empty_row(image_path):
    """A row with every label unknown; fill in what each dataset knows."""
    row = {"image_path": image_path, "severity": UNKNOWN}
    for d in TARGET_DISEASES:
        row[d] = UNKNOWN
    return row


# ODIR's per-image `labels` column uses single-letter codes.
_ODIR_LABEL_TO_DISEASE = {
    "D": "DR",
    "G": "Glaucoma",
    "C": "Cataract",
    "A": "AMD",
    "H": "Hypertensive_Retinopathy",
    # "M" (pathological myopia) and "O" (other) are real findings but not ones
    # we target, and a single-label annotation cannot tell us our five are
    # absent -- so those images are skipped rather than guessed at.
}


def load_odir(labels_csv, images_dir):
    """ODIR-5K -> our schema, using the PER-EYE labels.

    Two label sources exist in `full_df.csv` and choosing wrongly poisons the
    data:

      * the N/D/G/C/A/H/M/O columns are PATIENT-level. Both eyes of a patient
        carry identical flags (verified: 0 of 3358 patients differ), so a
        one-eyed disease is also stamped on the healthy fellow eye.
      * the `labels` column is PER-EYE, derived from that eye's own diagnostic
        keywords. That is the one we use.

    `labels` carries a single finding per image, so a positive tells us nothing
    about the other four conditions. Rather than invent negatives we mark them
    UNKNOWN (-1) and let the masked loss ignore them. Negatives come only from
    images annotated normal, where absence really was asserted.
    """
    df = pd.read_csv(labels_csv)
    rows, skipped = [], 0

    for _, r in df.iterrows():
        # the column is a stringified list, e.g. "['G']"
        codes = [c.strip(" '\"") for c in str(r["labels"]).strip("[]").split(",")]
        codes = [c for c in codes if c]

        path = os.path.join(images_dir, r["filename"])
        row = _empty_row(path)                      # everything UNKNOWN (-1)

        if codes == ["N"]:
            for d in TARGET_DISEASES:               # a normal eye is negative
                row[d] = 0                          # for all five
        else:
            hit = [_ODIR_LABEL_TO_DISEASE[c] for c in codes
                   if c in _ODIR_LABEL_TO_DISEASE]
            if not hit:                             # only M / O -> unusable
                skipped += 1
                continue
            for d in hit:
                row[d] = 1                          # the rest stay UNKNOWN

        rows.append(row)

    print(f"  ODIR: {len(rows)} usable rows, skipped {skipped} "
          f"(myopia/other only)")
    return rows


def load_mured(labels_csv, images_dir):
    df = pd.read_csv(labels_csv)
    rows = []
    for _, r in df.iterrows():
        path = os.path.join(images_dir, r["ID"] + ".png")   # adjust extension
        row = _empty_row(path)
        for col, disease in MURED_COLUMN_MAP.items():
            if disease and col in r:
                row[disease] = int(r[col])
        rows.append(row)
    return rows


def load_aptos(labels_csv, images_dir):
    """APTOS gives DR severity 0..4. We record severity AND derive DR presence."""
    df = pd.read_csv(labels_csv)
    rows = []
    for _, r in df.iterrows():
        path = os.path.join(images_dir, r["id_code"] + ".png")  # adjust extension
        row = _empty_row(path)
        sev = int(r[APTOS_SEVERITY_COLUMN])
        row["severity"] = sev
        row["DR"] = 1 if sev > 0 else 0                        # derived, now KNOWN
        rows.append(row)
    return rows


def load_severity_imagefolder(split_dir, exclude_base_ids=None):
    """Load a DR-severity dataset laid out as class folders: <split>/0 .. <split>/4.

    This is the layout of the EyePACS/APTOS-style download in D:/Eye_detection —
    no csv, the folder name IS the severity grade. As with APTOS we record the
    severity AND derive DR presence from it; the other four diseases stay
    UNKNOWN (-1) because this dataset says nothing about them.

    Many files are augmented copies of one eye ("<id>-600-FA.jpg", "-FS", "-HB",
    ...), so the base id is everything before the first '-'. Pass
    `exclude_base_ids` to keep eyes that appear in another split out of this one.
    """
    split_dir = os.fspath(split_dir)
    exclude = exclude_base_ids or set()
    rows, skipped = [], 0
    for sev in range(5):
        class_dir = os.path.join(split_dir, str(sev))
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            if base_id(fname) in exclude:
                skipped += 1
                continue
            row = _empty_row(os.path.join(class_dir, fname))
            row["severity"] = sev
            row["DR"] = 1 if sev > 0 else 0        # derived, now KNOWN
            rows.append(row)
    if skipped:
        print(f"  {split_dir}: skipped {skipped} files whose eye appears in another split")
    return rows


def base_id(filename):
    """'001639a390f0-600-FA.jpg' -> '001639a390f0' (one real eye, many crops)."""
    return os.path.splitext(filename)[0].split("-")[0]


def collect_base_ids(split_dir):
    """Every distinct eye id under a <split>/0..4 tree."""
    ids = set()
    for sev in range(5):
        class_dir = os.path.join(split_dir, str(sev))
        if os.path.isdir(class_dir):
            ids.update(base_id(f) for f in os.listdir(class_dir))
    return ids


def build(output_csv, sources):
    """sources: list of already-loaded row-lists to concatenate."""
    all_rows = []
    for src in sources:
        all_rows.extend(src)
    df = pd.DataFrame(all_rows)
    # column order: image_path, the 5 diseases, severity
    df = df[["image_path"] + TARGET_DISEASES + ["severity"]]
    df.to_csv(output_csv, index=False)
    print(f"Wrote {len(df)} rows -> {output_csv}")
    return df


if __name__ == "__main__":
    # Run with:  python -m data.prepare_manifest
    #
    # Current source: the DR-severity ImageFolder download at DR_ROOT below.
    # It grades DR severity only, so every row has severity + DR known and the
    # other four diseases UNKNOWN (-1). Add load_odir(...) / load_mured(...)
    # rows here once you have those datasets to train the other heads.
    DR_ROOT = "D:/Eye_detection"
    ODIR_CSV = "datasets/ODIR/full_df.csv"
    ODIR_IMAGES = "datasets/ODIR/ODIR-5K/ODIR-5K/Training Images"

    train_dir = os.path.join(DR_ROOT, "train")
    val_dir = os.path.join(DR_ROOT, "val")

    # --- source 1: the DR-severity set (severity + DR only) ---
    # The download reuses the same eye across splits (augmented copies), which
    # would leak training images into validation. Train wins; val gives them up.
    train_ids = collect_base_ids(train_dir)
    print(f"{len(train_ids)} distinct eyes in train")

    train_rows = load_severity_imagefolder(train_dir)
    val_rows = load_severity_imagefolder(val_dir, exclude_base_ids=train_ids)

    # --- source 2: ODIR-5K (the other four diseases) ---
    if os.path.exists(ODIR_CSV):
        odir = load_odir(ODIR_CSV, ODIR_IMAGES)

        # Split by PATIENT, not by image. Both eyes of one person are
        # correlated, so letting them straddle the split would inflate val.
        odir_df = pd.read_csv(ODIR_CSV)
        patient_of = dict(zip(odir_df["filename"], odir_df["ID"]))
        patients = sorted(set(patient_of.values()))

        rng = random.Random(0)                      # reproducible split
        rng.shuffle(patients)
        val_patients = set(patients[:max(1, int(0.15 * len(patients)))])

        o_train = [r for r in odir
                   if patient_of.get(os.path.basename(r["image_path"]))
                   not in val_patients]
        o_val = [r for r in odir
                 if patient_of.get(os.path.basename(r["image_path"]))
                 in val_patients]
        print(f"  ODIR split by patient: {len(o_train)} train / {len(o_val)} val "
              f"({len(val_patients)} of {len(patients)} patients held out)")

        train_rows += o_train
        val_rows += o_val
    else:
        print(f"ODIR not found at {ODIR_CSV} -- building DR-only manifests.")

    build("datasets/train_manifest.csv", [train_rows])
    build("datasets/val_manifest.csv", [val_rows])

    # --- a TEST manifest, touched by neither training nor checkpoint selection ---
    # Validation numbers are optimistic: the training loop picks the best epoch
    # by them. This split exists to give an honest final figure.
    test_dir = os.path.join(DR_ROOT, "test")
    if os.path.isdir(test_dir):
        seen = train_ids | collect_base_ids(val_dir)
        test_rows = load_severity_imagefolder(test_dir, exclude_base_ids=seen)
        build("datasets/test_manifest.csv", [test_rows])

    # --- report what each head will actually be supervised on ---
    for name, path in [("train", "datasets/train_manifest.csv"),
                       ("val", "datasets/val_manifest.csv")]:
        df = pd.read_csv(path)
        print(f"\n{name}: {len(df)} rows")
        for d in TARGET_DISEASES:
            known = df[df[d] != -1][d]
            print(f"   {d:28s} {int((known == 1).sum()):>6} pos / "
                  f"{int((known == 0).sum()):>6} neg / {int((df[d] == -1).sum()):>6} unknown")
