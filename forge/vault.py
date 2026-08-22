"""The session vault — secrets the model can USE but never SEE.

Merge needs credentials sometimes (an API key, an app password) but a secret
typed into a chat window is a secret written to disk: the session log, her
memory cards, the superego ledger, and every compaction summary would all
carry it. "Gone when the session ends" has to be built, not promised.

So the value never enters her context. The dashboard posts it straight here;
this module keeps it in memory only, and hands back a HANDLE like
`cred:gmail_app_password`. She passes the handle to a tool; the tool resolves
it inside the process at the moment of use. She can spend a secret without
ever holding one.

Rules that make that true:
  * RAM only. Nothing here touches the filesystem — no path, no cache, no log.
  * Session-scoped, with a wall-clock TTL, and wiped on demand or on exit.
  * `get()` is for tool internals. The listing/telling surfaces show handles
    and masked previews, never the value.
  * A resolved secret is never returned to the model as text — tools that use
    one must send it onward (to an API, a login) and report only the outcome.
"""
from __future__ import annotations

import atexit
import re
import threading
import time

_LOCK = threading.Lock()
_STORE: dict[str, dict] = {}          # handle -> {value, session, label, kind, born, ttl}
DEFAULT_TTL = 3600.0                  # an hour, unless the caller says otherwise

_SAFE = re.compile(r"[^a-z0-9_]+")


def _slug(name: str) -> str:
    return _SAFE.sub("_", str(name or "secret").strip().lower()).strip("_")[:40] or "secret"


def _expired(rec: dict, now: float) -> bool:
    return rec.get("ttl") and (now - rec["born"]) > rec["ttl"]


def _sweep(now: float | None = None) -> None:
    """Drop anything past its TTL. Called on every access, so a stale secret
    can't outlive its window just because nobody looked."""
    now = now or time.time()
    for h in [h for h, r in _STORE.items() if _expired(r, now)]:
        _wipe_one(h)


def _wipe_one(handle: str) -> None:
    rec = _STORE.pop(handle, None)
    if rec:
        rec["value"] = None           # drop the reference promptly


def put(name: str, value: str, session: str = "", label: str = "",
        kind: str = "password", ttl: float = DEFAULT_TTL) -> str:
    """Store a secret and return its handle. Called by the SERVER when a form
    is submitted — never by the model, which has no way to supply a value."""
    handle = f"cred:{_slug(name)}"
    with _LOCK:
        _sweep()
        _STORE[handle] = {"value": str(value), "session": session or "",
                          "label": label or name, "kind": kind,
                          "born": time.time(), "ttl": float(ttl or 0)}
    return handle


def get(handle: str, session: str = "") -> str | None:
    """Resolve a handle to its secret — for TOOL INTERNALS at the moment of
    use. Never return the result to the model; send it onward and report only
    what happened."""
    with _LOCK:
        _sweep()
        rec = _STORE.get(str(handle or "").strip())
        if not rec:
            return None
        if session and rec.get("session") and rec["session"] != session:
            return None               # another session's secret is not yours
        return rec.get("value")


def has(handle: str) -> bool:
    with _LOCK:
        _sweep()
        return str(handle or "").strip() in _STORE


def _mask(value: str, kind: str) -> str:
    """A preview that confirms WHICH secret it is without revealing it."""
    v = str(value or "")
    if kind in ("password", "token", "api_key", "secret"):
        return f"({len(v)} chars, ends …{v[-2:]})" if len(v) > 4 else "(set)"
    if kind == "email" and "@" in v:
        user, _, dom = v.partition("@")
        return f"{user[:2]}…@{dom}"
    return f"{v[:2]}…" if len(v) > 4 else "(set)"


def listing(session: str = "") -> str:
    """What's in the vault right now — handles and masked previews only."""
    with _LOCK:
        _sweep()
        rows = [(h, r) for h, r in _STORE.items()
                if not session or not r.get("session") or r["session"] == session]
    if not rows:
        return ("The vault is empty. Use request_credentials to put a form in "
                "front of the user; whatever they enter lands here, not in this "
                "conversation.")
    now = time.time()
    out = ["Vault (values are NOT visible to you — use the handle):"]
    for h, r in sorted(rows):
        left = int((r["ttl"] - (now - r["born"])) / 60) if r["ttl"] else None
        age = f", {left}m left" if left is not None else ""
        out.append(f"  {h}   {r['label']} {_mask(r['value'], r['kind'])}{age}")
    out.append("Pass a handle to a tool that needs it. clear_credentials wipes them.")
    return "\n".join(out)


def clear(session: str = "", handle: str = "") -> str:
    """Wipe secrets — one, or a whole session's worth."""
    with _LOCK:
        if handle:
            existed = handle in _STORE
            _wipe_one(handle)
            return (f"Wiped {handle}." if existed else f"No such handle: {handle}.")
        targets = [h for h, r in _STORE.items()
                   if not session or not r.get("session") or r["session"] == session]
        for h in targets:
            _wipe_one(h)
    return (f"Wiped {len(targets)} secret(s) from memory. They were never "
            "written to disk." if targets else "Nothing to wipe.")


def wipe_all() -> None:
    with _LOCK:
        for h in list(_STORE):
            _wipe_one(h)


atexit.register(wipe_all)             # nothing survives the process


# ---- what Merge can actually call ---------------------------------------
# Pending form requests, drained by the dashboard and shown to the user.
_PENDING: dict[str, dict] = {}

FIELD_KINDS = {"password", "text", "email", "token", "api_key", "secret", "url"}


def request_credentials(what: str, fields, session: str = "", note: str = "") -> str:
    """Ask the user for credentials with a real FORM in the chat window instead
    of making them paste secrets into the conversation.

    what: what this is for, in plain words ("your Bluesky account").
    fields: list of {"name","label","kind"} — kind is password/text/email/token/
    api_key/secret/url. Accepts a JSON string or a simple list of names.
    note: optional line explaining why you need it.

    What the user types goes straight to the vault and NEVER into this
    conversation: you receive handles like `cred:app_password`, pass those to
    the tool that needs them, and the values are wiped when the session ends or
    when you call clear_credentials. Tell the user that plainly — it's the
    reason to use the form rather than the chat box.
    """
    import json
    if isinstance(fields, str):
        try:
            fields = json.loads(fields)
        except Exception:
            fields = [f.strip() for f in fields.split(",") if f.strip()]
    if not isinstance(fields, list) or not fields:
        return ("Give me the fields to ask for, e.g. "
                '[{"name":"handle","label":"Your handle","kind":"text"},'
                '{"name":"app_password","label":"App password","kind":"password"}]')
    clean = []
    for f in fields[:10]:
        if isinstance(f, str):
            f = {"name": f, "label": f.replace("_", " ").title(), "kind": "password"}
        if not isinstance(f, dict) or not f.get("name"):
            continue
        kind = str(f.get("kind", "password")).lower()
        clean.append({"name": _slug(f["name"]),
                      "label": str(f.get("label") or f["name"])[:80],
                      "kind": kind if kind in FIELD_KINDS else "password"})
    if not clean:
        return "None of those fields were usable — each needs at least a name."
    form_id = f"form_{int(time.time()*1000)%10**9}"
    _PENDING[form_id] = {"id": form_id, "what": str(what)[:120],
                         "note": str(note or "")[:200], "fields": clean,
                         "session": session, "born": time.time()}
    names = ", ".join(f["name"] for f in clean)
    return (f"Form '{form_id}' is on screen asking for: {names}. "
            "Tell the user it's there and that what they type goes into "
            "memory only — you never see it, and it's wiped when you're done. "
            "Wait for them to submit; you'll get handles, not values.")


def take_pending(session: str = "") -> list[dict]:
    """The dashboard drains queued forms here (server-side only)."""
    out = [f for f in _PENDING.values()
           if not session or not f.get("session") or f["session"] == session]
    for f in out:
        _PENDING.pop(f["id"], None)
    return out


def credentials() -> str:
    """Show which credentials are currently in the session vault (handles and
    masked previews only — you cannot see the values)."""
    return listing()


def clear_credentials(handle: str = "") -> str:
    """Wipe credentials from the session vault — one handle, or everything if
    no handle is given. Use it the moment the job is done, and say so."""
    return clear(handle=handle)


# ---- catching a secret the user just typed ------------------------------
# The vault only protects what goes through the FORM. People type passwords
# straight into chat anyway ("my password is hunter2") — and that lands in the
# session log and the review ledger in plaintext, forever. So anything headed
# for disk gets scrubbed first. Deliberately narrow: it wants an explicit
# disclosure phrase or a key-shaped token, so ordinary talk about passwords
# ("I forgot my password") is untouched.
_DISCLOSURE = re.compile(
    r"\b((?:my|the|our|this)\s+"
    # longest first so 'passphrase' isn't chopped into 'pass'
    r"(?:passphrase|passwords?|passwd|api[ _-]?key|secret|token|pin|pass)\b"
    # REQUIRE a real handoff — a verb or a separator. Without this, ordinary
    # talk ("I forgot my password again") gets its next word redacted.
    r"(?:\s+(?:is|are|was)\s*[:=]?\s*|\s*[:=]\s*))"
    r"([^\s,.;!?\n]{3,120})", re.I)

# Long, high-entropy, key-shaped things that are secrets wherever they appear.
_KEYLIKE = re.compile(
    r"\b("
    r"sk-[A-Za-z0-9_-]{16,}"                 # OpenAI/Anthropic style
    r"|gh[pousr]_[A-Za-z0-9]{16,}"           # GitHub
    r"|xox[baprs]-[A-Za-z0-9-]{10,}"         # Slack
    r"|AKIA[0-9A-Z]{12,}"                    # AWS key id
    r"|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}"   # JWT
    r"|\b(?:[a-z0-9]{4}-){3}[a-z0-9]{4}\b"  # app-password style xxxx-xxxx-...
    r")")


def scrub(text: str) -> str:
    """Redact secrets from text on its way to disk. Returns the text unchanged
    when there's nothing secret-shaped in it."""
    if not text:
        return text
    s = str(text)
    s = _DISCLOSURE.sub(lambda m: m.group(1) + "[redacted]", s)
    s = _KEYLIKE.sub("[redacted]", s)
    return s


def scrubbed(text: str) -> bool:
    """Did scrubbing change anything? (For telling the user we caught one.)"""
    return scrub(text) != (text or "")
