"""
Calibrate the image-quality gate against real data.

`inference/safety.py` rejects an image when

    brightness < 25 or brightness > 230
    sharpness  < 100          <- variance of the Laplacian
    fundus_fraction < 0.15

Those are round numbers, not measurements, and the sharpness one is visibly
wrong for this pipeline: it flags clearly focused fundus photographs. Laplacian
variance is not a scale-free quantity -- it depends on image size, on CLAHE, and
on how much of the frame is retina. A cutoff that suits raw camera output does
not survive being resized to 512 with contrast enhancement applied.

So measure the distribution instead of guessing at it. This script samples
images, computes the same three statistics the safety layer uses, and reports
percentiles. The useful question is not "what is a good sharpness value" but
"what value would reject the worst 2% of my images", which is a decision about
how much you are willing to send for human review.

Run:
    python -m evaluation.calibrate_quality
    python -m evaluation.calibrate_quality --sample 800 --reject-fraction 0.05
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import CFG
from inference.safety import assess_quality
from preprocessing.fundus_preprocess import preprocess_fundus


def stats_for(path, size, use_clahe):
    """The quality statistics as the API would see them.

    The safety layer runs on the ORIGINAL image, before preprocessing, so
    measure both: the raw view is what production uses, the processed view
    shows how much the pipeline itself shifts the numbers.
    """
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    raw = assess_quality(rgb)
    proc = assess_quality(preprocess_fundus(rgb, size, use_clahe, False))
    return raw, proc


def percentiles(values, label):
    v = np.array([x for x in values if np.isfinite(x)])
    if len(v) == 0:
        print(f"  {label}: no data")
        return None
    qs = [0.5, 1, 2, 5, 10, 25, 50, 75, 95]
    print(f"  {label:<18}" + "".join(f"{q:>9}%" for q in qs))
    print(f"  {'':<18}" + "".join(f"{np.percentile(v, q):>10.1f}" for q in qs))
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(CFG.manifest_val))
    ap.add_argument("--sample", type=int, default=500)
    ap.add_argument("--reject-fraction", type=float, default=0.02,
                    help="fraction of real images you are willing to flag")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    n = min(args.sample, len(df))
    paths = df["image_path"].sample(n, random_state=0).tolist()
    print(f"sampling {n} images from {args.manifest}\n")

    raw_sharp, raw_bright, raw_frac = [], [], []
    proc_sharp = []
    flagged = {}
    unreadable = 0

    for p in tqdm(paths, desc="measuring", unit="img", dynamic_ncols=True):
        out = stats_for(p, CFG.image_size, CFG.use_clahe)
        if out is None:
            unreadable += 1
            continue
        raw, proc = out
        raw_sharp.append(raw["sharpness"])
        raw_bright.append(raw["brightness"])
        raw_frac.append(raw["fundus_fraction"])
        proc_sharp.append(proc["sharpness"])
        for issue in raw["issues"]:
            flagged[issue] = flagged.get(issue, 0) + 1

    if unreadable:
        print(f"\n{unreadable} images could not be read")

    print("\n" + "=" * 96)
    print("DISTRIBUTION OF QUALITY STATISTICS (on the original image, as the API sees it)")
    print("=" * 96)
    sharp = percentiles(raw_sharp, "sharpness")
    percentiles(raw_bright, "brightness")
    percentiles([f * 100 for f in raw_frac], "fundus area %")
    print()
    percentiles(proc_sharp, "sharpness (after preprocess)")

    print("\n" + "=" * 96)
    print("WHAT THE CURRENT THRESHOLDS DO")
    print("=" * 96)
    total = len(raw_sharp)
    if total:
        for issue, count in sorted(flagged.items(), key=lambda kv: -kv[1]):
            print(f"  {issue:<32}{count:>6} / {total}  ({count / total:>6.1%} of images)")
        if not flagged:
            print("  nothing flagged in this sample")
        print(f"\n  images failing at least one check: "
              f"{sum(1 for s, b, f in zip(raw_sharp, raw_bright, raw_frac) if s < 100 or b < 25 or b > 230 or f < 0.15)}"
              f" / {total}")

    if sharp is not None and total:
        print("\n" + "=" * 96)
        print(f"SUGGESTED SHARPNESS CUTOFF (flag the worst {args.reject_fraction:.0%})")
        print("=" * 96)
        suggested = float(np.percentile(sharp, args.reject_fraction * 100))
        current_rate = (sharp < 100).mean()
        print(f"  current cutoff     : 100      -> flags {current_rate:.1%} of real images")
        print(f"  suggested cutoff   : {suggested:.1f}      -> flags {args.reject_fraction:.0%} by construction")
        print("\n  A quality gate exists to catch unusable scans. If it flags a large")
        print("  share of ordinary images it is not protecting anyone -- it is just")
        print("  sending everything for review, which is the same as no gate at all.")
        print(f"\n  To apply:  set the sharpness cutoff in inference/safety.py to ~{suggested:.0f}")
        print("  Then eyeball the flagged images and confirm they really are bad.")
        print("  A threshold nobody has looked through is still a guess.")


if __name__ == "__main__":
    main()
