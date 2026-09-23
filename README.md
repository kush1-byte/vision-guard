# Vision Guard — multi-disease retinal screening

A backend, training pipeline and web UI that reads retinal **fundus** photographs
and screens for five conditions at once, plus a diabetic-retinopathy severity
grade:

| Output | Type |
|---|---|
| Diabetic Retinopathy (DR) | present / absent |
| Glaucoma | present / absent |
| Cataract | present / absent |
| Age-related Macular Degeneration (AMD) | present / absent |
| Hypertensive Retinopathy | present / absent |
| DR severity | No DR / Mild / Moderate / Severe / Proliferative |

It is **multi-label** — one eye can have several conditions — built with transfer
learning, explained with Grad-CAM, and wrapped in a safety layer whose default
answer is "send this to a human".

> ### Not a medical device
> This is a research and educational project. It must not be used to diagnose or
> treat anyone. Real clinical use would require prospective validation on the
> target population, bias and fairness testing, and regulatory clearance. The
> honest limitations are listed at the bottom of this file — read them before
> quoting any number from it.

---

## The design decision that shapes everything: partial labels

No single public dataset labels all five diseases *and* DR severity. So the
model is trained on several datasets at once, and **for any given image some
labels are known and some are unknown**.

Unknown disease labels are masked out of the loss; unknown severity uses
`ignore_index=-1`. A head only learns from images that actually say something
about its condition. This is what lets a 115k-image DR-severity dataset and a
5k-image multi-disease dataset train one model together.

Three consequences worth understanding:

1. A head with **no** labels anywhere never receives gradient. It stays at its
   random initialisation and emits values near 0.5 forever. The pipeline tracks
   which heads were genuinely supervised (`trained_diseases` in the checkpoint)
   and the API and UI refuse to present the others as predictions.
2. Negatives are only asserted where a dataset actually asserted them. A
   single-label annotation saying "glaucoma" does not mean cataract is absent,
   so the other conditions are recorded as unknown rather than guessed as 0.
3. The disease mix is wildly uneven — see Limitations.

---

## Layout

```
config.py                 every path, size and threshold in one place

data/
  label_map.py            the unified 5-disease + severity schema
  prepare_manifest.py     all source datasets -> ONE manifest csv (run once)
  datasets.py             PyTorch Dataset, handles partial labels

preprocessing/
  fundus_preprocess.py    auto-crop black border, CLAHE on green, resize
  transforms.py           runtime augmentation + ImageNet normalisation

models/
  vision_guard_net.py     timm backbone + a disease head and a severity head

training/
  losses.py               masked BCE (multi-label) + severity cross-entropy
  metrics.py              per-disease and macro AUROC
  train.py                training loop, resumable, early stopping

evaluation/
  evaluate.py             held-out TEST metrics, severity kappa, thresholds

explainability/
  gradcam.py              heatmaps of where the model looked

inference/
  safety.py               image-quality gate + confidence thresholds
  predictor.py            preprocess -> model -> safety (used by the API)

api/
  main.py                 FastAPI: UI, /predict, /explain, /health
  auth.py                 password hashing + signed session cookies
  oauth.py                Google / GitHub sign-in with an email allowlist

ui/
  index.html              the screening interface
  login.html              sign-in page
```

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
```

**On Windows this may fail**, and the error points at the wrong thing. Two pins
are needed:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "stringzilla==5.1.1" "albumentations>=1.4.18,<2"
pip install -r requirements.txt
```

* `stringzilla` is a transitive dependency of albumentations. Its newest release
  ships source-only and needs MSVC build tools; `5.1.1` has a prebuilt wheel.
* `albumentations` 2.x changes APIs this code does not use; pin to the 1.x line.
* Install the **CUDA** build of torch from pytorch.org if you have an NVIDIA GPU.
  Training is otherwise very slow.

---

## Step 1 — get the data

Download these yourself; both need free Kaggle accounts.

* **A DR-severity set** in class-folder layout (`train/0` … `train/4`), e.g.
  EyePACS/APTOS-derived collections. Gives DR presence and severity.
* **ODIR-5K** — [Ocular Disease Intelligent Recognition](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k).
  Gives Glaucoma, Cataract, AMD and Hypertensive Retinopathy.

> **ODIR has two label sources and the obvious one is wrong.** The
> `N/D/G/C/A/H/M/O` columns are **patient-level** — both eyes of a patient carry
> identical flags, so a one-eyed disease is stamped on the healthy fellow eye
> too (verified: 0 of 3358 patients differ between eyes). `prepare_manifest.py`
> uses the per-eye `labels` column instead. Using the patient-level columns
> would teach the model to call healthy eyes diseased.

## Step 2 — build the manifests

Point `DR_ROOT` and the ODIR paths at your copies in
`data/prepare_manifest.py`, then:

```bash
python -m data.prepare_manifest
```

This writes `train_manifest.csv`, `val_manifest.csv` and `test_manifest.csv`,
and prints the positive/negative/unknown count for every head. A row looks like:

```
image_path, DR, Glaucoma, Cataract, AMD, Hypertensive_Retinopathy, severity
img_001.png, 1, 0, 0, 0, -1, 2        # -1 means "unknown for this image"
```

Two leakage guards run here: augmented copies of the same eye are kept within a
single split, and ODIR is split **by patient**, never by image.

## Step 3 — train

```bash
python -u -m training.train | Tee-Object -FilePath train.log
```

Prints which heads are supervised, shows a live per-image progress bar, reports
per-disease AUROC each epoch, saves the best checkpoint, and early-stops when
validation plateaus.

**It is resumable.** `checkpoints/last.pt` is written every epoch with optimizer,
scheduler and scaler state. If the machine dies, rerun the same command and it
continues from the next epoch. Delete `last.pt` to start fresh.

## Step 4 — evaluate honestly

```bash
python -m evaluation.evaluate
```

Per-epoch AUROC comes from the validation set, which is also used to *pick* the
checkpoint — so it is optimistic. This reports the test split, used for neither,
with bootstrap confidence intervals, the severity confusion matrix and
quadratic-weighted kappa, and the threshold each disease needs for 95%
sensitivity.

## Step 5 — run the app

```bash
uvicorn api.main:app --port 8000
```

Open <http://localhost:8000>.

### Accounts

```bash
python -m api.auth add <username>      # also: list, remove
```

Passwords are stored as PBKDF2-HMAC-SHA256 with a per-user salt. Sessions are
signed HttpOnly cookies, so restarting the server does not log anyone out.

### Google / GitHub sign-in (optional)

Create an OAuth client, then `oauth_config.json`:

```json
{
  "google": { "client_id": "...", "client_secret": "..." },
  "redirect_base": "http://localhost:8000",
  "allowed_emails": ["you@example.com"]
}
```

Redirect URI: `http://localhost:8000/auth/google/callback` (register the
`127.0.0.1` form too — Google treats them as different origins).

> **The allowlist is the security boundary, not the provider.** "Sign in with
> Google" authenticates anyone on the internet with a Google account; it answers
> *who are you*, never *may you in*. An empty allowlist therefore denies
> everyone rather than admitting everyone.

Never commit `oauth_config.json`, `auth_users.json` or `.session_secret` — the
last one lets anyone forge a session. All three are in `.gitignore`.

### Endpoints

| Route | Purpose |
|---|---|
| `GET /` | the UI (redirects to `/login` when signed out) |
| `POST /predict` | image → probabilities, severity, quality, safety verdict |
| `POST /explain?disease=DR` | image → Grad-CAM heatmap PNG |
| `GET /health` | liveness, whether a model loaded, current user |

---

## Limitations

Read these before quoting any figure.

**The training mix is overwhelmingly DR.** Roughly 115k DR-severity images
against ~4.6k ODIR images. DR sees hundreds of times more supervision than the
other four conditions, and its metrics are correspondingly far more reliable.

**Three conditions rest on a few hundred examples.** Glaucoma, Cataract and AMD
have roughly 230–255 training positives each; Hypertensive Retinopathy has ~107.
Validation has 21 Hypertensive positives, which makes its AUROC statistically
noisy — a swing of several points between epochs is sampling noise, not learning.
Treat that head as indicative at best.

**Within-split augmentation inflates the numbers.** The DR source ships
pre-augmented copies of each eye. Copies of one eye never cross splits, but they
do repeat within a split, so validation and test are easier than genuinely
unseen data.

**No external validation.** Every figure comes from the same two sources the
model trained on. Performance on a different camera, clinic or population is
unmeasured and would be lower.

**Thresholds are not clinically calibrated.** The defaults in `config.py` are
starting guesses. `evaluation/evaluate.py` reports data-driven operating points,
but choosing one is a clinical decision about the cost of a miss versus a false
alarm, not a statistical one.

**The image-quality gate is heuristic.** Brightness, Laplacian-variance
sharpness and fundus-area fraction, with hand-picked cutoffs. The sharpness
threshold in particular is not calibrated for this preprocessing pipeline and
flags some clearly focused images.

**Untrained heads are shown as `n/a`, not as zero.** If a condition has no
labels in your manifest, its head is random and the UI says so rather than
printing a confident-looking number. If you add data for it, retrain and it
appears automatically.
# Vision-guard
# Vision-guard
