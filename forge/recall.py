"""
Mid-term memory: recall, and the librarian.

Merge's memory comes in three speeds: the context window (short — lives on
the graphics card, lossy when full), the saved sessions on disk (mid —
every word ever exchanged, verbatim), and the notebook + project files
(long). This module opens the middle tier:

  * search() — lookup across every saved conversation. HYBRID: it ranks the
    librarian's index cards by MEANING (embedding cosine, via embed.py) AND
    by keyword, so a memory surfaces even when it was said in other words —
    then blends verbatim keyword hits from recent transcripts on top.
  * remember_turn() — drops the finished exchange in the librarian's
    in-tray (a queue folder). Instant; nobody waits.
  * process_queue_forever() — the librarian: the dashboard (always running)
    works the tray on a background thread, asking the little 3B to distill
    each exchange into a one-line index card AND embedding it into her
    vector space. It also keeps the working set bounded (see below).

Two things stay TRUE no matter what:
  * If the little model is down, the tray just waits — nothing is lost.
  * If the embedding organ is down, search falls back to pure keyword —
    less associative, never broken.

Pile-up guards (the disk is abundant, but the set she SEARCHES each turn is
kept bounded so it never chokes her):
  * Cards past CARD_HOT_CAP are ARCHIVED (moved out of the hot file, kept
    forever) so cosine ranking stays fast.
  * The verbatim transcript scan only reads the SESSION_SCAN_MAX most-recent
    conversations; older ones are covered by their cards.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import numpy as np

from . import embed
from .config import load_config

SESS_DIR = Path.home() / ".forge" / "sessions"
CARDS_PATH = Path.home() / ".forge" / "memory-cards.jsonl"
CARDS_ARCHIVE = Path.home() / ".forge" / "memory-cards-archive.jsonl"
VECS_PATH = Path.home() / ".forge" / "card-vectors.f32"   # raw float32, DIM per card, row-aligned to CARDS_PATH
QUEUE_DIR = Path.home() / ".forge" / "card-queue"

# Short prompt on purpose: the 3B reads ~24 tokens/sec on the CPU, so the
# card job must stay a few hundred tokens or the librarian falls behind.
CARD_PROMPT = ("One line, 25 words max: what happened in this exchange? "
               "Name the concrete things — names, files, decisions, numbers. "
               "No preamble, no quotes.")
EXCHANGE_LIMIT = 1200      # chars of (user + answer) the librarian reads

# --- pile-up guards (all tunable, single numbers) ---
CARD_HOT_CAP = 6000        # newest cards kept in the fast, searched file
CARD_ARCHIVE_SLACK = 500   # only compact once we're this far over (avoid churn)
SESSION_SCAN_MAX = 40      # most-recent transcripts the verbatim scan reads

# --- hybrid scoring weights ---
_W_SEM = 3.0               # weight on embedding cosine (0..1) for a card
_W_KW = 1.0               # weight per keyword hit
_SEM_FLOOR = 0.50          # a card with no keyword hit still counts if cosine >= this
_WS_BOOST = 0.5            # same-workspace nudge


def _words(query: str) -> list[str]:
    return [w for w in "".join(c.lower() if c.isalnum() else " "
                               for c in query).split() if len(w) >= 3]


def _snippet(text: str, words: list[str], width: int = 150) -> str:
    low = text.lower()
    at = min((low.find(w) for w in words if w in low), default=0)
    start = max(0, at - width // 3)
    piece = text[start:start + width].replace("\n", " ").strip()
    return ("…" if start else "") + piece + ("…" if start + width < len(text) else "")


# ---------------------------------------------------------------- vectors

def _load_cards() -> list[dict]:
    if not CARDS_PATH.exists():
        return []
    out = []
    for raw in CARDS_PATH.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(raw))
        except Exception:
            continue
    return out


def _load_vecs() -> np.ndarray:
    """(N, DIM) unit vectors, row-aligned to the cards file. Empty if none."""
    try:
        raw = np.fromfile(VECS_PATH, dtype="float32")
        if raw.size == 0:
            return np.empty((0, embed.DIM), dtype="float32")
        return raw.reshape(-1, embed.DIM)
    except Exception:
        return np.empty((0, embed.DIM), dtype="float32")


def _reconcile_vectors(cards: list[dict] | None = None) -> np.ndarray:
    """Make the vector file match the cards file, embedding any that are
    missing or were stored as zeros (organ was down when they were filed).
    Cheap no-op when already in sync. Returns the aligned (N, DIM) matrix."""
    cards = _load_cards() if cards is None else cards
    n = len(cards)
    vecs = _load_vecs()
    # trim any corruption where vectors outran cards
    if len(vecs) > n:
        vecs = vecs[:n]
    need = []                     # indices to (re)embed
    for i in range(n):
        if i >= len(vecs) or not np.any(vecs[i]):
            need.append(i)
    if not need:
        return vecs
    if not embed.available():
        return vecs               # try again later; keyword still works
    # grow to full height, fill the gaps in batches
    full = np.zeros((n, embed.DIM), dtype="float32")
    if len(vecs):
        full[:len(vecs)] = vecs
    for s in range(0, len(need), 128):
        chunk = need[s:s + 128]
        m = embed.embed_documents([cards[i].get("gist", "") for i in chunk])
        if m is None:
            break                 # organ dropped mid-backfill; keep what we have
        for j, i in enumerate(chunk):
            full[i] = m[j]
    try:
        tmp = VECS_PATH.with_suffix(".f32.tmp")
        full.astype("float32").tofile(tmp)
        tmp.replace(VECS_PATH)
    except Exception:
        pass
    return full


# ---------------------------------------------------------------- search

def search(query: str, workspace: str = "", limit: int = 8) -> str:
    """Every hit is something that was actually said, with when and where.
    Cards are ranked by meaning + keyword; transcripts by keyword."""
    words = _words(query)
    qvec = embed.embed_query(query) if query.strip() else None
    if not words and qvec is None:
        return "Give recall a few concrete words to look for."

    hits: list[tuple[float, float, str]] = []   # (score, when, line)

    # -- index cards: meaning (cosine) blended with keyword --
    cards = _load_cards()
    vecs = _reconcile_vectors(cards) if qvec is not None else None
    for i, c in enumerate(cards):
        gist = c.get("gist", "")
        low = gist.lower()
        kw = sum(1.0 for w in words if w in low)
        cos = float(qvec @ vecs[i]) if (qvec is not None and vecs is not None
                                        and i < len(vecs)) else 0.0
        if kw == 0 and cos < _SEM_FLOOR:
            continue
        score = _W_SEM * max(cos, 0.0) + _W_KW * kw
        if c.get("workspace") == workspace:
            score += _WS_BOOST
        when = c.get("t", 0)
        day = time.strftime("%b %d", time.localtime(when))
        folder = Path(c.get("workspace", "")).name or "?"
        why = "≈" if (kw == 0 and cos >= _SEM_FLOOR) else "·"   # ≈ = found by meaning alone
        hits.append((score, when, f"[{day} {why} {folder} · card] {gist}"))

    # -- verbatim transcripts: keyword only, bounded to recent conversations --
    if words:
        files = sorted(SESS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime,
                       reverse=True) if SESS_DIR.is_dir() else []
        for f in files[:SESSION_SCAN_MAX]:
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            folder = Path(d.get("workspace", "")).name or "?"
            boost = _WS_BOOST if d.get("workspace") == workspace else 0.0
            for e in d.get("log", []):
                if e.get("kind") not in ("user", "text"):
                    continue
                text = e.get("text", "")
                low = text.lower()
                score = sum(1.0 for w in words if w in low)
                if not score:
                    continue
                who = "user" if e["kind"] == "user" else "Merge"
                when = e.get("t", 0)
                day = time.strftime("%b %d", time.localtime(when))
                hits.append((score + boost, when,
                             f"[{day} · {folder} · {who}] {_snippet(text, words)}"))

    if not hits:
        return (f"Nothing in past conversations matches {query!r}. "
                "It may never have been said — or said in words too different "
                "even to feel related.")
    hits.sort(key=lambda h: (-h[0], -h[1]))
    return "\n".join(line for _, _, line in hits[:limit])


# ---------------------------------------------------------------- librarian

def remember_turn(user_text: str, answer_text: str,
                  workspace: str, session_id: str) -> None:
    """Drop the finished exchange in the librarian's tray. Instant — one
    small file write; the slow distilling happens elsewhere, later."""
    if not (user_text or "").strip() or not (answer_text or "").strip():
        return
    try:
        QUEUE_DIR.mkdir(parents=True, exist_ok=True)
        entry = {"t": time.time(), "workspace": workspace,
                 "session": session_id,
                 "user": user_text[:EXCHANGE_LIMIT // 2],
                 "answer": answer_text[:EXCHANGE_LIMIT // 2]}
        (QUEUE_DIR / f"{time.time_ns()}.json").write_text(
            json.dumps(entry), encoding="utf-8")
    except Exception:
        pass   # a missing card must never trouble a finished conversation


def _distill(entry: dict) -> str | None:
    """One exchange → one line, via the little model.

    Returns None when the little model is unreachable (leave the tray
    alone, try again later) and "" when it answered uselessly (drop the
    entry — retrying the same input won't get smarter)."""
    cfg = load_config()
    little = cfg.get("models", {}).get("little")
    if not little or not little.get("base_url"):
        return None
    exchange = f"User: {entry.get('user', '')}\nMerge: {entry.get('answer', '')}"
    try:
        r = httpx.post(
            little["base_url"].rstrip("/") + "/chat/completions",
            json={"model": little.get("model", "little"),
                  "messages": [{"role": "system", "content": CARD_PROMPT},
                               {"role": "user", "content": exchange}],
                  "max_tokens": 60, "temperature": 0.2},
            timeout=120.0)
        return r.json()["choices"][0]["message"]["content"].strip()
    except httpx.HTTPError:
        return None
    except Exception:
        return ""


def _append_card_and_vec(rec: dict) -> None:
    """Append one card to the hot file and its embedding to the aligned
    vector file (zeros if the organ's down — backfilled later)."""
    with CARDS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    m = embed.embed_documents([rec.get("gist", "")])
    vec = m[0] if m is not None else np.zeros(embed.DIM, dtype="float32")
    with VECS_PATH.open("ab") as vf:
        vf.write(vec.astype("float32").tobytes())


def _archive_overflow() -> int:
    """Keep the hot file at CARD_HOT_CAP: oldest cards move to the archive
    (kept forever, out of the search loop); their vectors drop in step."""
    cards = _load_cards()
    if len(cards) <= CARD_HOT_CAP + CARD_ARCHIVE_SLACK:
        return 0
    keep_from = len(cards) - CARD_HOT_CAP
    old, new = cards[:keep_from], cards[keep_from:]
    try:
        with CARDS_ARCHIVE.open("a", encoding="utf-8") as fh:
            for c in old:
                fh.write(json.dumps(c) + "\n")
        tmpc = CARDS_PATH.with_suffix(".jsonl.tmp")
        tmpc.write_text("".join(json.dumps(c) + "\n" for c in new), encoding="utf-8")
        tmpc.replace(CARDS_PATH)
        # vectors: keep the tail that matches the surviving cards
        vecs = _load_vecs()
        if len(vecs) >= len(cards):
            tail = vecs[keep_from:]
            tmpv = VECS_PATH.with_suffix(".f32.tmp")
            tail.astype("float32").tofile(tmpv)
            tmpv.replace(VECS_PATH)
        else:
            VECS_PATH.unlink(missing_ok=True)   # out of sync → rebuilt by reconcile
    except Exception:
        return 0
    return len(old)


def process_queue_once() -> int:
    """Work the tray. Returns how many cards were filed."""
    filed = 0
    for f in sorted(QUEUE_DIR.glob("*.json")) if QUEUE_DIR.is_dir() else []:
        try:
            entry = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            f.unlink(missing_ok=True)
            continue
        gist = _distill(entry)
        if gist is None:
            return filed          # librarian's asleep — leave the tray be
        if gist:
            _append_card_and_vec({
                "t": entry.get("t", time.time()),
                "workspace": entry.get("workspace", ""),
                "session": entry.get("session", ""),
                "gist": gist.splitlines()[0][:200]})
            filed += 1
        f.unlink(missing_ok=True)
    if filed:
        _archive_overflow()
    return filed


def process_queue_forever(interval: float = 30.0) -> None:
    """The dashboard runs this on a daemon thread: it's the one process
    that's always up, so cards get filed no matter which door the
    conversation came through. Also backfills any un-embedded cards when
    the organ is reachable, so memory heals itself into her vector space."""
    while True:
        try:
            process_queue_once()
            _reconcile_vectors()   # heal missing/zero vectors when organ's up (no-op when synced)
        except Exception:
            pass
        time.sleep(interval)
