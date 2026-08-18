"""Owner identity + a challenge phrase, checked entirely server-side.

The point: the secret phrase must never sit in the model's context, because
Merge is abliterated (uncensored) and a determined person could try to talk it
out of her. So we store only a salted one-way hash of the phrase; the model
gets a `verify_phrase` tool that asks us "does this attempt match?" and hears
back nothing but 'correct' / 'incorrect'. She never learns the phrase, so there
is nothing to leak.

Voice-friendly: both the stored phrase and every attempt are normalised
(lowercased, punctuation dropped, whitespace collapsed) so a phrase typed in
settings still matches the same words spoken aloud and transcribed by whisper.
"""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets

from .config import load_config, save_config


def _norm(s: str) -> str:
    """Fold a phrase to its bare words so typed and spoken forms match."""
    return " ".join(re.sub(r"[^\w\s]", "", (s or "").lower()).split())


def _hash(phrase: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", _norm(phrase).encode(), bytes.fromhex(salt), 100_000
    ).hex()


def status() -> dict:
    """What the settings UI is allowed to know — never the phrase itself."""
    ident = load_config().get("identity") or {}
    return {"enabled": bool(ident.get("enabled")),
            "owner_name": ident.get("owner_name", ""),
            "has_phrase": bool(ident.get("phrase_hash"))}


def armed() -> bool:
    """The checks actually fire only when switched on AND a phrase is set — so
    turning it on with no phrase can't lock you out."""
    ident = load_config().get("identity") or {}
    return bool(ident.get("enabled") and ident.get("phrase_hash"))


def set_identity(enabled: bool | None = None, owner_name: str | None = None,
                 phrase: str | None = None) -> dict:
    """Update the switch, owner name, and/or phrase. phrase=None leaves it
    untouched, "" clears it, anything else sets it (stored only as a salted hash)."""
    cfg = load_config()
    ident = cfg.setdefault("identity", {})
    if enabled is not None:
        ident["enabled"] = bool(enabled)
    if owner_name is not None:
        ident["owner_name"] = owner_name.strip()[:60]
    if phrase is not None:
        p = _norm(phrase)
        if p:
            salt = secrets.token_hex(16)
            ident["phrase_salt"] = salt
            ident["phrase_hash"] = _hash(p, salt)
        else:
            ident["phrase_salt"] = ""
            ident["phrase_hash"] = ""
    save_config(cfg)
    return status()


def check_phrase(attempt: str) -> bool:
    """Constant-time compare an attempt against the stored hash. False if no
    phrase is set (so 'ask for the phrase' with none configured never passes)."""
    ident = load_config().get("identity") or {}
    stored, salt = ident.get("phrase_hash", ""), ident.get("phrase_salt", "")
    if not stored or not salt:
        return False
    try:
        return hmac.compare_digest(stored, _hash(attempt, salt))
    except Exception:
        return False
