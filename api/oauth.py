"""
Sign in with Google / GitHub (OAuth 2.0 authorization-code flow).

Uses `httpx`, which FastAPI already pulls in — no new dependencies. The session
cookie issued at the end is the same one `api/auth.py` mints, so password logins
and social logins share one session mechanism.

THE IMPORTANT PART — an allowlist.
    "Sign in with Google" authenticates *anyone on the internet who has a Google
    account*. It answers "who are you", never "may you in". So a provider login
    only succeeds if the verified email matches `allowed_emails` or
    `allowed_domains`. With neither configured, every provider login is refused
    and the page tells you which address was turned away. That is deliberate:
    an empty allowlist must mean "nobody", not "everybody".

Configuration — environment variables win, else `oauth_config.json` in the
project root:

    GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET
    GITHUB_CLIENT_ID / GITHUB_CLIENT_SECRET
    OAUTH_REDIRECT_BASE      e.g. http://localhost:8000

    {
      "google": { "client_id": "...", "client_secret": "..." },
      "github": { "client_id": "...", "client_secret": "..." },
      "redirect_base": "http://localhost:8000",
      "allowed_emails": ["you@gmail.com"],
      "allowed_domains": ["your-university.edu"]
    }

Neither file nor env should ever be committed — a client secret is a credential.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable

import httpx

from config import CFG

CONFIG_FILE = CFG.project_root / "oauth_config.json"
STATE_COOKIE = "vg_oauth_state"
STATE_TTL = 600           # 10 minutes to complete a round trip


# --------------------------------------------------------------------------
# provider definitions
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Provider:
    key: str
    label: str
    authorize_url: str
    token_url: str
    scope: str
    fetch_identity: Callable          # async (httpx.AsyncClient, token) -> dict
    extra_auth_params: dict


async def _google_identity(client: httpx.AsyncClient, token: str) -> dict:
    r = await client.get("https://openidconnect.googleapis.com/v1/userinfo",
                         headers={"Authorization": f"Bearer {token}"})
    r.raise_for_status()
    d = r.json()
    # Google marks unverified addresses; an unverified email proves nothing.
    if not d.get("email_verified"):
        raise PermissionError("Your Google account's email is not verified.")
    return {"email": (d.get("email") or "").lower(),
            "name": d.get("name") or d.get("email"),
            "subject": d.get("sub")}


async def _github_identity(client: httpx.AsyncClient, token: str) -> dict:
    headers = {"Authorization": f"Bearer {token}",
               "Accept": "application/vnd.github+json"}
    prof = await client.get("https://api.github.com/user", headers=headers)
    prof.raise_for_status()
    p = prof.json()

    # GitHub's /user endpoint hides the email unless it is public, so ask for
    # the address list and take the verified primary one.
    emails = await client.get("https://api.github.com/user/emails", headers=headers)
    emails.raise_for_status()
    primary = next((e for e in emails.json()
                    if e.get("primary") and e.get("verified")), None)
    if primary is None:
        raise PermissionError("No verified primary email on your GitHub account.")
    return {"email": primary["email"].lower(),
            "name": p.get("name") or p.get("login"),
            "subject": str(p.get("id"))}


PROVIDERS: dict[str, Provider] = {
    "google": Provider(
        key="google", label="Google",
        authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        scope="openid email profile",
        fetch_identity=_google_identity,
        # consent + offline are unnecessary here; we only need one identity read
        extra_auth_params={"prompt": "select_account"},
    ),
    "github": Provider(
        key="github", label="GitHub",
        authorize_url="https://github.com/login/oauth/authorize",
        token_url="https://github.com/login/oauth/access_token",
        scope="read:user user:email",
        fetch_identity=_github_identity,
        extra_auth_params={},
    ),
}


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
def _file_config() -> dict:
    if not CONFIG_FILE.exists():
        return {}
    try:
        return json.loads(CONFIG_FILE.read_text("utf-8"))
    except json.JSONDecodeError:
        return {}


def credentials(provider_key: str) -> tuple[str | None, str | None]:
    """(client_id, client_secret) from the environment, else the config file."""
    env_id = os.environ.get(f"{provider_key.upper()}_CLIENT_ID")
    env_secret = os.environ.get(f"{provider_key.upper()}_CLIENT_SECRET")
    if env_id and env_secret:
        return env_id, env_secret
    section = _file_config().get(provider_key) or {}
    return section.get("client_id"), section.get("client_secret")


def is_configured(provider_key: str) -> bool:
    return all(credentials(provider_key))


def enabled_providers() -> list[dict]:
    """What the login page should show buttons for."""
    return [{"key": p.key, "label": p.label}
            for p in PROVIDERS.values() if is_configured(p.key)]


def redirect_base(fallback: str) -> str:
    return (os.environ.get("OAUTH_REDIRECT_BASE")
            or _file_config().get("redirect_base")
            or fallback).rstrip("/")


def redirect_uri(provider_key: str, fallback_base: str) -> str:
    return f"{redirect_base(fallback_base)}/auth/{provider_key}/callback"


# --------------------------------------------------------------------------
# the allowlist — authentication is not authorization
# --------------------------------------------------------------------------
def allowlist() -> tuple[set[str], set[str]]:
    cfg = _file_config()
    env_emails = os.environ.get("OAUTH_ALLOWED_EMAILS", "")
    env_domains = os.environ.get("OAUTH_ALLOWED_DOMAINS", "")
    emails = {e.strip().lower() for e in
              [*cfg.get("allowed_emails", []), *env_emails.split(",")] if e.strip()}
    domains = {d.strip().lower().lstrip("@") for d in
               [*cfg.get("allowed_domains", []), *env_domains.split(",")] if d.strip()}
    return emails, domains


def is_allowed(email: str) -> bool:
    emails, domains = allowlist()
    if not emails and not domains:
        return False                      # empty allowlist means nobody
    email = (email or "").lower()
    if email in emails:
        return True
    return "@" in email and email.rsplit("@", 1)[1] in domains


# --------------------------------------------------------------------------
# the flow
# --------------------------------------------------------------------------
def authorize_url(provider_key: str, state: str, base: str) -> str:
    p = PROVIDERS[provider_key]
    client_id, _ = credentials(provider_key)
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri(provider_key, base),
        "response_type": "code",
        "scope": p.scope,
        "state": state,
        **p.extra_auth_params,
    }
    return str(httpx.URL(p.authorize_url, params=params))


async def exchange_code(provider_key: str, code: str, base: str) -> str:
    """Swap the one-time code for an access token, server to server over TLS."""
    p = PROVIDERS[provider_key]
    client_id, client_secret = credentials(provider_key)
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "redirect_uri": redirect_uri(provider_key, base),
        "grant_type": "authorization_code",
    }
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(p.token_url, data=data,
                              headers={"Accept": "application/json"})
        if r.status_code >= 400:
            raise PermissionError(f"{p.label} rejected the token exchange "
                                  f"(HTTP {r.status_code}).")
        token = r.json().get("access_token")
    if not token:
        raise PermissionError(f"{p.label} returned no access token.")
    return token


async def identity(provider_key: str, access_token: str) -> dict:
    p = PROVIDERS[provider_key]
    async with httpx.AsyncClient(timeout=15) as client:
        return await p.fetch_identity(client, access_token)
