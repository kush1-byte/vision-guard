"""
FastAPI backend.

Endpoints:
  GET  /health              -> liveness check
  POST /predict             -> upload a fundus image, get predictions + safety verdict
  POST /explain?disease=DR  -> upload an image, get a Grad-CAM heatmap (PNG)

Run locally:
    uvicorn api.main:app --reload --port 8000

The model is loaded once at startup. Point CHECKPOINT at your trained weights.
"""
from __future__ import annotations

import io
from urllib.parse import quote

import cv2
import numpy as np
from fastapi import (Depends, FastAPI, File, Form, HTTPException, Query,
                     Request, Response, UploadFile)
from fastapi.responses import (FileResponse, JSONResponse, RedirectResponse,
                               StreamingResponse)

import secrets

from api import auth, oauth
from config import CFG
from data.label_map import TARGET_DISEASES
from inference.predictor import VisionGuardPredictor

CHECKPOINT = CFG.checkpoint_dir / "vision_guard_best.pt"
UI_FILE = CFG.project_root / "ui" / "index.html"
LOGIN_FILE = CFG.project_root / "ui" / "login.html"

app = FastAPI(title="Vision Guard API", version="1.0")
predictor: VisionGuardPredictor | None = None


@app.on_event("startup")
def load_model():
    global predictor
    try:
        predictor = VisionGuardPredictor(CHECKPOINT)
        print(f"Loaded model from {CHECKPOINT}")
    except FileNotFoundError:
        # Let the server start so /health works; /predict will report clearly.
        print(f"WARNING: checkpoint not found at {CHECKPOINT}. Train a model first.")


def _read_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def require_user(request: Request) -> str:
    """Dependency: the signed session cookie must name a valid, unexpired user."""
    user = auth.read_token(request.cookies.get(auth.COOKIE_NAME))
    if user is None:
        raise HTTPException(status_code=401, detail="Not signed in.")
    return user


# ----------------------------- auth routes -----------------------------
@app.get("/login", include_in_schema=False)
def login_page():
    if not LOGIN_FILE.exists():
        raise HTTPException(status_code=404, detail=f"Login page missing at {LOGIN_FILE}")
    return FileResponse(LOGIN_FILE)


@app.post("/auth/login")
def login(response: Response, username: str = Form(...), password: str = Form(...)):
    if not auth.any_users():
        raise HTTPException(
            status_code=503,
            detail="No accounts exist. Run: python -m api.auth add <username>")
    if not auth.check_login(username, password):
        raise HTTPException(status_code=401, detail="Wrong username or password.")
    response.set_cookie(
        auth.COOKIE_NAME, auth.make_token(username),
        max_age=auth.SESSION_TTL, httponly=True, samesite="lax",
        # secure=True once you put this behind TLS; on plain-HTTP localhost a
        # secure cookie would simply never be sent back.
        secure=False,
    )
    return {"ok": True, "user": username}


@app.post("/auth/logout")
def logout(response: Response):
    response.delete_cookie(auth.COOKIE_NAME)
    return {"ok": True}


@app.get("/auth/providers")
def providers():
    """Which social buttons the login page should render."""
    return {"providers": oauth.enabled_providers()}


@app.get("/auth/{provider}/start", include_in_schema=False)
def oauth_start(provider: str, request: Request):
    if provider not in oauth.PROVIDERS or not oauth.is_configured(provider):
        raise HTTPException(status_code=404, detail=f"{provider} sign-in is not configured.")

    # CSRF defence: a random nonce goes to the provider as `state`, and a signed
    # copy rides in a cookie. The callback must present both and they must match,
    # so a login someone else started cannot be finished in your browser.
    nonce = secrets.token_urlsafe(24)
    signed = auth.make_token(f"oauth:{provider}:{nonce}", ttl=oauth.STATE_TTL)

    target = oauth.authorize_url(provider, nonce, str(request.base_url))
    response = RedirectResponse(target, status_code=302)
    response.set_cookie(oauth.STATE_COOKIE, signed, max_age=oauth.STATE_TTL,
                        httponly=True, samesite="lax", secure=False)
    return response


@app.get("/auth/{provider}/callback", include_in_schema=False)
async def oauth_callback(provider: str, request: Request,
                         code: str | None = None, state: str | None = None,
                         error: str | None = None):
    def back(msg: str):
        return RedirectResponse(f"/login?error={quote(msg)}", status_code=302)

    if provider not in oauth.PROVIDERS or not oauth.is_configured(provider):
        return back(f"{provider} sign-in is not configured.")
    if error:
        return back(f"{provider} sign-in was cancelled ({error}).")
    if not code or not state:
        return back("Sign-in response was incomplete.")

    expected = auth.read_token(request.cookies.get(oauth.STATE_COOKIE))
    if expected != f"oauth:{provider}:{state}":
        return back("Sign-in expired or was tampered with. Please try again.")

    try:
        token = await oauth.exchange_code(provider, code, str(request.base_url))
        who = await oauth.identity(provider, token)
    except PermissionError as e:
        return back(str(e))
    except Exception:
        return back(f"Could not reach {provider}. Please try again.")

    email = who.get("email")
    if not email:
        return back(f"{provider} did not return an email address.")
    if not oauth.is_allowed(email):
        # Naming the address is the difference between a usable error and a
        # mystery; it is the user's own address, shown only to them.
        return back(f"{email} is not on the allowlist for this instance.")

    response = RedirectResponse("/", status_code=302)
    response.set_cookie(auth.COOKIE_NAME, auth.make_token(email),
                        max_age=auth.SESSION_TTL, httponly=True,
                        samesite="lax", secure=False)
    response.delete_cookie(oauth.STATE_COOKIE)
    return response


# ----------------------------- app routes ------------------------------
@app.get("/", include_in_schema=False)
def ui(request: Request):
    """The browser UI. Served from this app so it is same-origin with /predict."""
    if auth.read_token(request.cookies.get(auth.COOKIE_NAME)) is None:
        return RedirectResponse("/login", status_code=302)
    if not UI_FILE.exists():
        raise HTTPException(status_code=404, detail=f"UI not found at {UI_FILE}")
    return FileResponse(UI_FILE)


@app.get("/health")
def health(request: Request):
    """Public on purpose: the login page reads it to detect a first-run setup."""
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "has_users": auth.any_users(),
        "user": auth.read_token(request.cookies.get(auth.COOKIE_NAME)),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), user: str = Depends(require_user)):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    image = _read_image(await file.read())
    result = predictor.predict(image)
    return JSONResponse(result)


@app.post("/explain")
async def explain(disease: str = Query(...), file: UploadFile = File(...),
                  user: str = Depends(require_user)):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if disease not in TARGET_DISEASES:
        raise HTTPException(status_code=400,
                            detail=f"disease must be one of {TARGET_DISEASES}")
    # The UI hides the button, but the endpoint is reachable directly, and a
    # heatmap for a condition we are not reporting would look like a claim.
    scope = predictor.reported_diseases
    if scope is not None and disease not in scope:
        raise HTTPException(
            status_code=409,
            detail=f"{disease} is outside the validated scope; "
                   f"currently reporting: {sorted(scope)}")
    image = _read_image(await file.read())
    overlay = predictor.explain(image, disease)             # RGB uint8
    ok, buf = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode heatmap.")
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")
