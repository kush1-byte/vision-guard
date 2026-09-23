# Getting Vision Guard running

Everything you need to run the web app is in this repo, including a trained
model. **You do not need to download any datasets or train anything** — that
part only matters if you want to retrain it yourself (see the end).

Tested on Windows 11 with Python 3.11. macOS/Linux work too; where the commands
differ it is noted.

---

## 1. Requirements

- **Python 3.11** — `python --version` to check. 3.12 also works; 3.13 may not,
  because some of the pinned packages have no wheels for it yet.
- **~6 GB of disk** — almost all of it PyTorch.
- **A GPU is optional.** The app runs fine on CPU (a prediction takes a second
  or two instead of milliseconds). A GPU only matters for retraining.

---

## 2. Clone and create a virtual environment

```powershell
git clone <the repo url>
cd Eye-detection

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
```

Your prompt should now start with `(.venv)`. Everything below assumes that.

---

## 3. Install the dependencies

**On Windows, run these three in order.** Plain
`pip install -r requirements.txt` fails here, and the error it prints points at
the wrong thing entirely — see the note below.

```powershell
pip install --upgrade pip

# CPU-only is fine and much smaller (~200 MB instead of ~2.7 GB):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# ...or, if you have an NVIDIA GPU and want to retrain:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install "stringzilla==5.1.1" "albumentations>=1.4.18,<2"
pip install -r requirements.txt
```

<details>
<summary>Why those two pins (worth reading if it still fails)</summary>

`albumentations` depends on `albucore`, which depends on `stringzilla`. The
newest stringzilla release ships as source only, so pip tries to compile it and
dies with:

```
error: Microsoft Visual C++ 14.0 or greater is required
```

That message suggests you need Visual Studio build tools. You don't —
`stringzilla==5.1.1` has a prebuilt Windows wheel. Pinning it avoids the
compile entirely.

The `albumentations<2` pin is separate: `requirements.txt` says `>=1.3`, so pip
would otherwise install 2.x, whose API differs from what this code uses.
</details>

---

## 4. Create a login account

The app requires you to sign in.

```powershell
python -m api.auth add yourname
```

It asks for a password twice (hidden, minimum 8 characters). Also available:
`python -m api.auth list` and `python -m api.auth remove <name>`.

Passwords are stored as salted PBKDF2 hashes — nothing is kept in plaintext.

---

## 5. Run it

```powershell
uvicorn api.main:app --port 8000
```

Open **<http://localhost:8000>** and sign in with the account you just made.

You should see `Loaded model from checkpoints\vision_guard_best.pt` in the
terminal. If it says the checkpoint is missing, you cloned without Git LFS or
the file did not download — check `checkpoints/` contains a ~16 MB
`vision_guard_best.pt`.

### Using it

Drag in a retinal fundus photograph and press **Analyze**. You get:

- probabilities for five conditions, with the 0.35–0.65 uncertainty band drawn
  behind each bar
- a diabetic-retinopathy severity stage
- an image-quality report
- a referral verdict — the app is a *screening* tool, so anything uncertain,
  positive or poor-quality goes to a human
- **Grad-CAM heatmaps** showing which part of the retina drove each score

No fundus images handy? Search "fundus photograph" or grab a few from the
public ODIR-5K dataset on Kaggle. Ordinary photographs will be rejected by the
quality gate, which is the intended behaviour.

---

## 6. Optional: Google sign-in

Only if you want it — the password login above is enough.

Create an OAuth client at <https://console.cloud.google.com/apis/credentials>
(Web application), add `http://localhost:8000/auth/google/callback` **and**
`http://127.0.0.1:8000/auth/google/callback` as authorized redirect URIs, add
your own Gmail as a Test user on the consent screen, then create
`oauth_config.json` in the project root:

```json
{
  "google": { "client_id": "...", "client_secret": "..." },
  "redirect_base": "http://localhost:8000",
  "allowed_emails": ["your.address@gmail.com"]
}
```

Restart the server and a "Continue with Google" button appears.

> `allowed_emails` is not optional. "Sign in with Google" authenticates anyone
> on the internet with a Google account — it establishes *who you are*, never
> *whether you may enter*. An empty allowlist therefore denies everyone rather
> than admitting everyone.

Never commit that file. It is already in `.gitignore`.

---

## 7. Optional: retrain it yourself

Only worth doing with an NVIDIA GPU — roughly 20 minutes per epoch on an
RTX 4060, about 6 hours in total.

You need two datasets, both free but requiring a Kaggle account:

1. A **DR-severity** set in class-folder layout (`train/0` … `train/4`)
2. **ODIR-5K** — <https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k>

Then point `DR_ROOT` and the ODIR paths in `data/prepare_manifest.py` at your
copies and run:

```powershell
python -m data.prepare_manifest          # build the manifests
python -u -m training.train              # train (resumable — see below)
python -m evaluation.evaluate            # held-out test metrics
```

Training writes `checkpoints/last.pt` every epoch, so if the machine dies you
can rerun the same command and it continues from the next epoch rather than
starting over.

---

## What the numbers mean

On a held-out test set of 12,871 images (used for neither training nor
model selection):

| | |
|---|---|
| Diabetic retinopathy AUROC | **0.9624** (95% CI 0.959–0.966) |
| DR severity, quadratic-weighted kappa | **0.8510** |
| DR severity, exact accuracy | 0.7449 |

Glaucoma, Cataract, AMD and Hypertensive Retinopathy have no test labels — the
test split comes from the DR dataset only. Their validation figures are 0.939,
0.986, 0.958 and 0.898, but those are optimistic, and each rests on 21–45
positive examples. The `Limitations` section of `README.md` is honest about
this and worth reading before quoting any of it.

**This is a research project, not a medical device.** Do not use it to diagnose
anyone.
