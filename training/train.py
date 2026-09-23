"""
Training loop.

Run with:  python -m training.train

It trains VisionGuardNet on the train manifest, evaluates macro-AUROC on the val
manifest each epoch, saves the best checkpoint, and stops early if validation
stops improving.

Assumes you've already built datasets/train_manifest.csv and val_manifest.csv
(see data/prepare_manifest.py).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG
from data.datasets import FundusDataset
from data.label_map import TARGET_DISEASES
from models.vision_guard_net import VisionGuardNet
from preprocessing.transforms import train_transforms, val_transforms
from training.losses import VisionGuardLoss
from training.metrics import multilabel_auroc


def compute_pos_weight(dataset):
    """pos_weight[i] = (#negatives / #positives) for disease i, over KNOWN labels.
    Bigger weight -> the model is penalised more for missing that (rare) disease.
    """
    df = dataset.df
    weights = []
    for d in TARGET_DISEASES:
        known = df[df[d] != -1][d]
        pos = max((known == 1).sum(), 1)
        neg = max((known == 0).sum(), 1)
        weights.append(neg / pos)
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels, all_mask = [], [], []
    bar = tqdm(total=len(loader.dataset), unit="img", unit_scale=True,
               desc="  validating", leave=False, dynamic_ncols=True)
    for batch in loader:
        imgs = batch["image"].to(device)
        logits = model(imgs)["disease_logits"]
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(batch["disease_labels"].numpy())
        all_mask.append(batch["disease_mask"].numpy())
        bar.update(imgs.size(0))
    bar.close()
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    mask = np.concatenate(all_mask)
    return multilabel_auroc(probs, labels, mask)


def main():
    device = CFG.device if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")
    CFG.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_ds = FundusDataset(CFG.manifest_train, train_transforms(CFG.image_size),
                             CFG.image_size, CFG.use_clahe, CFG.use_denoise)
    val_ds = FundusDataset(CFG.manifest_val, val_transforms(CFG.image_size),
                           CFG.image_size, CFG.use_clahe, CFG.use_denoise)

    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False,
                            num_workers=CFG.num_workers, pin_memory=True)

    model = VisionGuardNet(CFG.backbone, CFG.num_diseases, CFG.num_severity,
                           CFG.pretrained, CFG.dropout).to(device)

    # Which disease heads will actually be supervised by this manifest. A column
    # that is -1 everywhere never contributes to the loss, so its head stays at
    # its random init -- inference must know not to trust those outputs.
    supervised = sorted(d for d in TARGET_DISEASES if (train_ds.df[d] != -1).any())
    unsupervised = [d for d in TARGET_DISEASES if d not in supervised]
    print(f"Supervised heads: {', '.join(supervised) or '(none)'}")
    if unsupervised:
        print(f"NOT supervised (no labels in this manifest): {', '.join(unsupervised)}")

    pos_weight = compute_pos_weight(train_ds).to(device)
    criterion = VisionGuardLoss(pos_weight=pos_weight,
                                severity_weight=CFG.severity_loss_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr,
                                  weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(CFG.use_amp and device == "cuda"))

    best_auroc, patience, start_epoch = 0.0, 0, 1

    # ---- resume, if a previous run left a last.pt behind ----
    resume_path = CFG.checkpoint_dir / "last.pt"
    if resume_path.exists():
        state = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["scaler_state"])
        best_auroc = state["best_auroc"]
        patience = state["patience"]
        start_epoch = state["epoch"] + 1
        print(f"Resuming from {resume_path}: epoch {state['epoch']} done, "
              f"best macro-AUROC {best_auroc:.4f}, patience {patience}/"
              f"{CFG.early_stop_patience}")
        if start_epoch > CFG.epochs:
            print(f"Nothing to do: {state['epoch']} epochs already completed. "
                  f"Raise CFG.epochs or delete {resume_path} to start over.")
            return

    for epoch in range(start_epoch, CFG.epochs + 1):
        model.train()
        running = 0.0
        # live image counter: how many of this epoch's images are done, plus
        # throughput and ETA. Counts IMAGES, not batches.
        bar = tqdm(total=len(train_ds), unit="img", unit_scale=True,
                   desc=f"epoch {epoch:02d}/{CFG.epochs}", dynamic_ncols=True)
        for seen_batches, batch in enumerate(train_loader, start=1):
            imgs = batch["image"].to(device)
            targets = {
                "disease_labels": batch["disease_labels"].to(device),
                "disease_mask": batch["disease_mask"].to(device),
                "severity": batch["severity"].to(device),
            }
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(CFG.use_amp and device == "cuda")):
                outputs = model(imgs)
                loss, _ = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            bar.update(imgs.size(0))
            bar.set_postfix(loss=f"{running / seen_batches:.4f}")
        bar.close()

        scheduler.step()
        metrics = evaluate(model, val_loader, device)
        avg_loss = running / max(len(train_loader), 1)
        print(f"Epoch {epoch:02d} | loss {avg_loss:.4f} | "
              f"macro-AUROC {metrics['macro']:.4f} | "
              + " ".join(f"{d}:{metrics[d]:.3f}" for d in TARGET_DISEASES
                         if metrics[d] is not None))

        if metrics["macro"] > best_auroc:
            best_auroc, patience = metrics["macro"], 0
            ckpt = CFG.checkpoint_dir / "vision_guard_best.pt"
            torch.save({"model_state": model.state_dict(),
                        "backbone": CFG.backbone,
                        "macro_auroc": best_auroc,
                        # heads that actually saw labels; the rest stay at their
                        # random init and must not be read as predictions
                        "trained_diseases": supervised}, ckpt)
            print(f"  -> saved new best ({best_auroc:.4f}) to {ckpt}")
        else:
            patience += 1

        # Resume point, written EVERY epoch (the best checkpoint alone cannot
        # resume: it may be several epochs old and carries no optimizer state).
        # Write to a temp file and replace, so a crash mid-save cannot leave a
        # truncated last.pt that would poison the next start.
        tmp = CFG.checkpoint_dir / "last.pt.tmp"
        torch.save({"model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "epoch": epoch,
                    "best_auroc": best_auroc,
                    "patience": patience,
                    "backbone": CFG.backbone,
                    "trained_diseases": supervised}, tmp)
        tmp.replace(CFG.checkpoint_dir / "last.pt")

        if patience >= CFG.early_stop_patience:
            print("Early stopping.")
            break

    print(f"Done. Best macro-AUROC: {best_auroc:.4f}")


if __name__ == "__main__":
    main()
