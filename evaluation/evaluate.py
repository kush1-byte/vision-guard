"""
Held-out test-set evaluation.

The numbers printed during training come from the VALIDATION set, which the
training loop also uses to pick the best checkpoint. That makes them
optimistic: the model was selected to do well on exactly those images. This
script reports the honest version, on data used for neither.

It covers three things training never measured:

  1. per-disease AUROC on the test set, with a bootstrap confidence interval
     (essential here -- some conditions have only a few dozen positives, and a
     point estimate alone hides how little that pins down)
  2. the DR SEVERITY head, which had no metric at all: confusion matrix,
     per-stage recall, and quadratic-weighted kappa -- the standard DR grading
     metric, because calling Proliferative "No DR" is far worse than calling
     Moderate "Mild", and plain accuracy cannot express that
  3. operating points: the threshold each disease needs for a target
     sensitivity, and the specificity you pay for it

Run:
    python -m evaluation.evaluate                      # test split
    python -m evaluation.evaluate --manifest datasets/val_manifest.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.metrics import (cohen_kappa_score, confusion_matrix,
                             roc_auc_score, roc_curve)
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG
from data.datasets import FundusDataset
from data.label_map import SEVERITY_CLASSES, TARGET_DISEASES
from models.vision_guard_net import VisionGuardNet
from preprocessing.transforms import val_transforms

TARGET_SENSITIVITY = 0.95      # screening errs toward catching disease


def collect(model, loader, device):
    """Run the model over a loader and return raw predictions + labels."""
    probs, labels, masks, sev_pred, sev_true = [], [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluating", unit="batch",
                          dynamic_ncols=True):
            out = model(batch["image"].to(device))
            probs.append(torch.sigmoid(out["disease_logits"]).cpu().numpy())
            sev_pred.append(torch.softmax(out["severity_logits"], 1).cpu().numpy())
            labels.append(batch["disease_labels"].numpy())
            masks.append(batch["disease_mask"].numpy())
            sev_true.append(batch["severity"].numpy())
    return (np.concatenate(probs), np.concatenate(labels), np.concatenate(masks),
            np.concatenate(sev_pred), np.concatenate(sev_true))


def bootstrap_auroc(y_true, y_score, n=2000, seed=0):
    """Percentile CI. With 20-odd positives the point estimate means little on
    its own, so report the interval alongside it."""
    rng = np.random.default_rng(seed)
    scores = []
    idx = np.arange(len(y_true))
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y_true[s])) < 2:
            continue
        scores.append(roc_auc_score(y_true[s], y_score[s]))
    if not scores:
        return None, None
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def operating_point(y_true, y_score, target_sens):
    """Lowest threshold reaching the target sensitivity, and its specificity."""
    fpr, tpr, thr = roc_curve(y_true, y_score)
    ok = np.where(tpr >= target_sens)[0]
    if len(ok) == 0:
        return None, None, None
    i = ok[0]
    return float(thr[i]), float(tpr[i]), float(1 - fpr[i])


def report_diseases(probs, labels, masks, trained):
    print("\n" + "=" * 78)
    print("PER-DISEASE PERFORMANCE (test set)")
    print("=" * 78)
    print(f"{'condition':<28}{'AUROC':>8}{'95% CI':>18}{'pos':>7}{'neg':>8}")
    print("-" * 78)

    aurocs = {}
    for i, disease in enumerate(TARGET_DISEASES):
        m = masks[:, i] == 1
        y, s = labels[m, i], probs[m, i]
        if len(np.unique(y)) < 2:
            print(f"{disease:<28}{'n/a':>8}{'(no labels)':>18}"
                  f"{int(y.sum()):>7}{int((y == 0).sum()):>8}")
            continue
        auc = roc_auc_score(y, s)
        lo, hi = bootstrap_auroc(y, s)
        aurocs[disease] = auc
        ci = f"[{lo:.3f}, {hi:.3f}]" if lo is not None else "-"
        tag = "" if (trained is None or disease in trained) else "  (UNTRAINED)"
        print(f"{disease:<28}{auc:>8.4f}{ci:>18}"
              f"{int(y.sum()):>7}{int((y == 0).sum()):>8}{tag}")

    if aurocs:
        print("-" * 78)
        print(f"{'macro average':<28}{np.mean(list(aurocs.values())):>8.4f}")
    return aurocs


def report_thresholds(probs, labels, masks):
    print("\n" + "=" * 78)
    print(f"OPERATING POINTS (threshold for {TARGET_SENSITIVITY:.0%} sensitivity)")
    print("=" * 78)
    print(f"{'condition':<28}{'threshold':>11}{'sens':>9}{'spec':>9}"
          f"{'  vs config 0.50':>18}")
    print("-" * 78)
    for i, disease in enumerate(TARGET_DISEASES):
        m = masks[:, i] == 1
        y, s = labels[m, i], probs[m, i]
        if len(np.unique(y)) < 2:
            continue
        thr, sens, spec = operating_point(y, s, TARGET_SENSITIVITY)
        if thr is None:
            print(f"{disease:<28}{'unreachable':>11}")
            continue
        # what the current fixed 0.50 threshold actually gives you
        pred = s >= CFG.refer_threshold
        sens50 = pred[y == 1].mean() if (y == 1).any() else float("nan")
        print(f"{disease:<28}{thr:>11.4f}{sens:>9.3f}{spec:>9.3f}"
              f"{sens50:>18.3f}")
    print("\n  last column = sensitivity you get today at the fixed 0.50 cutoff")


def report_severity(sev_pred, sev_true):
    known = sev_true != -1
    if known.sum() == 0:
        print("\n(no severity labels in this manifest - skipped)")
        return
    y = sev_true[known]
    p = sev_pred[known].argmax(1)

    print("\n" + "=" * 78)
    print(f"DR SEVERITY GRADING  ({known.sum()} labelled images)")
    print("=" * 78)

    acc = (y == p).mean()
    kappa = cohen_kappa_score(y, p, weights="quadratic")
    adjacent = (np.abs(y - p) <= 1).mean()
    print(f"  exact accuracy            : {acc:.4f}")
    print(f"  within one stage          : {adjacent:.4f}")
    print(f"  quadratic-weighted kappa  : {kappa:.4f}   <- the metric that matters")
    print("     (>0.8 excellent, 0.6-0.8 substantial, 0.4-0.6 moderate)")

    cm = confusion_matrix(y, p, labels=range(len(SEVERITY_CLASSES)))
    print("\n  confusion matrix  (rows = true, cols = predicted)")
    head = "".join(f"{s[:6]:>9}" for s in SEVERITY_CLASSES)
    print(f"  {'':<15}{head}")
    for i, name in enumerate(SEVERITY_CLASSES):
        row = "".join(f"{v:>9}" for v in cm[i])
        recall = cm[i, i] / cm[i].sum() if cm[i].sum() else float("nan")
        print(f"  {name:<15}{row}   recall {recall:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="datasets/test_manifest.csv")
    ap.add_argument("--checkpoint",
                    default=str(CFG.checkpoint_dir / "vision_guard_best.pt"))
    ap.add_argument("--batch-size", type=int, default=CFG.batch_size)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    device = CFG.device if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    trained = ckpt.get("trained_diseases")

    model = VisionGuardNet(ckpt.get("backbone", CFG.backbone), CFG.num_diseases,
                           CFG.num_severity, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    ds = FundusDataset(args.manifest, val_transforms(CFG.image_size),
                       CFG.image_size, CFG.use_clahe, CFG.use_denoise)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    print(f"checkpoint : {args.checkpoint}")
    print(f"  selected on val macro-AUROC {ckpt.get('macro_auroc', float('nan')):.4f}")
    print(f"  trained heads: {', '.join(trained) if trained else '(not recorded)'}")
    print(f"manifest   : {args.manifest}  ({len(ds)} images)")
    print(f"device     : {device}")

    probs, labels, masks, sev_pred, sev_true = collect(model, loader, device)

    report_diseases(probs, labels, masks, set(trained) if trained else None)
    report_severity(sev_pred, sev_true)
    report_thresholds(probs, labels, masks)

    print("\n" + "=" * 78)
    print("These are test-set numbers: this data was used for neither training")
    print("nor checkpoint selection, so they are not inflated the way the")
    print("per-epoch validation figures are.")
    print("=" * 78)


if __name__ == "__main__":
    main()
