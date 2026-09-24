"""
Session authentication for the Vision Guard UI.

Standard library only — no extra dependencies. Two pieces:

  1. Password storage: PBKDF2-HMAC-SHA256 with a per-user random salt. Plaintext
     passwords are never written anywhere. Users live in `auth_users.json`,
     created by the CLI at the bottom of this file or by signing up on the
     login page.

  2. Session cookie: a signed token, `base64(username|expiry).base64(hmac)`.
     The server keeps no session table — it just verifies its own signature, so
     a restart does not log everyone out. The signing key lives in
     `.session_secret` (generated on first use) or the VISION_GUARD_SECRET
     environment variable.

> **Scope.** This keeps casual visitors off a machine you control. It is not
> clinical access control: the dev server speaks plain HTTP, so on anything but
> localhost the cookie crosses the wire in the clear. Put it behind TLS before
> it leaves your machine, and do not treat this as protecting patient data.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path

from config import CFG

USER_FILE = CFG.project_root / "auth_users.json"
SECRET_FILE = CFG.project_root / ".session_secret"

COOKIE_NAME = "vg_session"
SESSION_TTL = 12 * 60 * 60      # seconds; re-login after 12 hours
_ITERATIONS = 240_000


# --------------------------------------------------------------------------
# password hashing
# --------------------------------------------------------------------------
def hash_password(password: str, salt: bytes | None = None) -> dict:
    """Return a JSON-serialisable record. Never returns the password itself."""
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ITERATIONS)
    return {"salt": salt.hex(), "hash": digest.hex(), "iterations": _ITERATIONS}


def verify_password(password: str, record: dict) -> bool:
    """Constant-time check, so a wrong guess costs the same time as a near-miss."""
    try:
        salt = bytes.fromhex(record["salt"])
        expected = bytes.fromhex(record["hash"])
        iterations = int(record.get("iterations", _ITERATIONS))
    except (KeyError, ValueError):
        return False
    got = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(got, expected)


# --------------------------------------------------------------------------
# user store
# --------------------------------------------------------------------------
def load_users() -> dict:
    if not USER_FILE.exists():
        return {}
    try:
        return json.loads(USER_FILE.read_text("utf-8"))
    except json.JSONDecodeError:
        return {}


def save_users(users: dict) -> None:
    USER_FILE.write_text(json.dumps(users, indent=2), "utf-8")


def add_user(username: str, password: str) -> None:
    username = username.strip()
    if not username or "|" in username:
        raise ValueError("username must be non-empty and must not contain '|'")
    if len(password) < 8:
        raise ValueError("password must be at least 8 characters")
    users = load_users()
    users[username] = hash_password(password)
    save_users(users)


def check_login(username: str, password: str) -> bool:
    record = load_users().get(username.strip())
    if record is None:
        # Hash anyway so a missing user and a wrong password take similar time,
        # which stops an attacker enumerating valid usernames by timing.
        hash_password(password)
        return False
    return verify_password(password, record)


def any_users() -> bool:
    return bool(load_users())


# --------------------------------------------------------------------------
# self-service sign-up
# --------------------------------------------------------------------------
# Registration is open: anyone who can reach the page can create an account.
# Set  "signup": {"enabled": false}  in oauth_config.json to turn it off.
#
# Note this is a different policy from the OAuth allowlist, which still gates
# provider logins. Password accounts are open; Google identities are not.
INSTANCE_CONFIG = CFG.project_root / "oauth_config.json"


def _instance_config() -> dict:
    if not INSTANCE_CONFIG.exists():
        return {}
    try:
        return json.loads(INSTANCE_CONFIG.read_text("utf-8"))
    except json.JSONDecodeError:
        return {}


def signup_enabled() -> bool:
    return (_instance_config().get("signup") or {}).get("enabled", True) is not False


def create_account(username: str, password: str) -> None:
    """Register a new user. Raises PermissionError or ValueError on refusal."""
    if not signup_enabled():
        raise PermissionError("Sign-up is disabled on this instance.")

    username = username.strip()
    # add_user() overwrites by design (it is the admin CLI). Here an existing
    # name must be refused outright -- otherwise signing up as someone who
    # already exists would silently reset their password.
    if username in load_users():
        raise ValueError("That username is already taken.")
    add_user(username, password)


# --------------------------------------------------------------------------
# signed session token
# --------------------------------------------------------------------------
def _secret() -> bytes:
    env = os.environ.get("VISION_GUARD_SECRET")
    if env:
        return env.encode("utf-8")
    if not SECRET_FILE.exists():
        SECRET_FILE.write_bytes(secrets.token_bytes(32))
    return SECRET_FILE.read_bytes()


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def make_token(username: str, ttl: int = SESSION_TTL) -> str:
    payload = f"{username}|{int(time.time()) + ttl}".encode("utf-8")
    sig = hmac.new(_secret(), payload, hashlib.sha256).digest()
    return f"{_b64e(payload)}.{_b64e(sig)}"


def read_token(token: str | None) -> str | None:
    """Return the username if the token is validly signed and unexpired."""
    if not token or "." not in token:
        return None
    body, sig = token.rsplit(".", 1)
    try:
        payload = _b64d(body)
        given = _b64d(sig)
    except (ValueError, base64.binascii.Error):
        return None
    expected = hmac.new(_secret(), payload, hashlib.sha256).digest()
    if not hmac.compare_digest(given, expected):
        return None
    try:
        username, expiry = payload.decode("utf-8").rsplit("|", 1)
    except ValueError:
        return None
    if int(expiry) < time.time():
        return None
    return username


# --------------------------------------------------------------------------
# CLI:  python -m api.auth add <username>
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import getpass
    import sys

    args = sys.argv[1:]
    if len(args) == 2 and args[0] == "add":
        pw = getpass.getpass("New password (min 8 chars): ")
        if pw != getpass.getpass("Repeat password: "):
            sys.exit("Passwords did not match.")
        try:
            add_user(args[1], pw)
        except ValueError as e:
            sys.exit(str(e))
        print(f"User '{args[1]}' saved to {USER_FILE}")
    elif args[:1] == ["list"]:
        users = load_users()
        print("\n".join(users) if users else f"No users yet in {USER_FILE}")
    elif len(args) == 2 and args[0] == "remove":
        users = load_users()
        if users.pop(args[1], None) is None:
            sys.exit(f"No such user: {args[1]}")
        save_users(users)
        print(f"Removed '{args[1]}'")
    else:
        print("usage:\n"
              "  python -m api.auth add <username>\n"
              "  python -m api.auth list\n"
              "  python -m api.auth remove <username>")
