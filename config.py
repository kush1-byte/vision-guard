"""
Central configuration for Vision Guard.

Everything tunable lives here so you never hunt through the code to change a
path, an image size, or a threshold. Import `CFG` anywhere you need a setting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ----- paths -----
    project_root: Path = Path(__file__).parent
    data_root: Path = Path("./datasets")          # where the raw datasets live
    manifest_train: Path = Path("./datasets/train_manifest.csv")
    manifest_val: Path = Path("./datasets/val_manifest.csv")
    cache_dir: Path = Path("./datasets/preprocessed")  # deterministic-preprocessing cache
    checkpoint_dir: Path = Path("./checkpoints")

    # ----- image / preprocessing -----
    image_size: int = 512
    use_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8
    use_denoise: bool = False   # slow; leave off unless images are noisy (see notes)

    # ----- model -----
    # any timm backbone name works here; these two are the ones from the spec
    backbone: str = "efficientnet_b0"   # or "resnet50"
    pretrained: bool = True
    dropout: float = 0.3

    # ----- label schema -----
    num_diseases: int = 5
    num_severity: int = 5   # No_DR, Mild, Moderate, Severe, Proliferative

    # ----- reporting scope -----
    # Which trained heads are considered validated enough to report. The model
    # has five heads and all five received gradient, but only DR has dense
    # labels (APTOS severity + ODIR presence) and an operating point we can
    # defend. The other four learn from ODIR partial labels alone, with too few
    # positives to quote a threshold -- so they are computed but held back
    # rather than shown as if they were finished.
    #
    # Set to None to report every trained head.
    reported_diseases: tuple = ("DR",)

    # ----- training -----
    epochs: int = 30
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    severity_loss_weight: float = 0.5
    num_workers: int = 4
    use_amp: bool = True    # mixed precision on GPU
    early_stop_patience: int = 6

    # ----- inference / safety thresholds -----
    # A screening tool should err toward "send to a human". These defaults are
    # deliberately conservative; calibrate them on your validation set.
    # Image-quality gate. These are measured, not guessed: see
    # `python -m evaluation.calibrate_quality`. Laplacian variance is not
    # scale-free -- it depends on image size and on CLAHE -- so the textbook
    # "blurry below 100" flagged 86% of ordinary fundus photographs here. 8 is
    # the 5th percentile of this dataset, i.e. it flags the worst 5%.
    min_sharpness: float = 8.0
    min_brightness: float = 25.0
    max_brightness: float = 230.0
    min_fundus_fraction: float = 0.15

    refer_threshold: float = 0.50       # prob >= this  -> treat as a positive screen
    uncertain_low: float = 0.35         # the "grey zone" that gets flagged as
    uncertain_high: float = 0.65        # low-confidence
    min_top_confidence: float = 0.60    # if nothing is confidently normal/abnormal, review

    device: str = "cuda"  # falls back to cpu automatically in the code


CFG = Config()
