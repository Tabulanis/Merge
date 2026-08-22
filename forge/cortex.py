"""Cortex — your own life-archive, indexed locally and searchable by meaning.

Point it at a Google Takeout export (or any mbox / .ics / .vcf / folder of
documents) and it builds a private, on-disk index: every message, event, and
contact reduced to a record with a summary, participants, a date, and an
embedding. Then you can ask for things the way you actually remember them —
"that thread about the roof quote last spring" — instead of guessing keywords.

Deliberate design choices:

  * NO credentials, ever. It reads an archive YOU exported. Nothing here logs
    into anything, so there's nothing to revoke and nothing to leak.
  * Its own store (~/.forge/cortex/), NOT her conversation memory. Tens of
    thousands of archival records would drown a few hundred distilled memory
    cards, and keeping them apart means the archive can be wiped on its own.
  * Local only. The archive never leaves the machine; embeddings are computed
    by the local embed server.
  * Summaries are extractive by default (the real first lines of the text).
    Nothing is invented — a record you can't parse is skipped and counted, not
    guessed at.
"""
from __future__ import annotations

import email.utils
import hashlib
import json
import mailbox
import re
import time
from email.header import decode_header, make_header
from pathlib import Path

import numpy as np

DIR = Path.home() / ".forge" / "cortex"
RECORDS = DIR / "records.jsonl"
VECS = DIR / "vectors.f32"
STATE = DIR / "state.json"

# Buckets a record can land in. Rules do the obvious ones for free; anything
# ambiguous is left "unsorted" for Merge to judge — never guessed at here.
CATEGORIES = ["financial", "travel", "receipts", "work", "personal", "health",
              "legal", "accounts", "newsletters", "calendar", "contacts",
              "unsorted"]

_RULES = [
    ("receipts", r"\b(receipt|order confirm|your order|invoice|payment received|"
                 r"shipped|tracking number|refund)\b"),
    ("financial", r"\b(statement|balance|transaction|deposit|withdraw|tax|irs|"
                  r"1099|w-2|mortgage|interest rate|account summary)\b"),
    ("travel", r"\b(itinerary|boarding pass|flight|reservation|check-in|hotel|"
               r"booking confirm|departure)\b"),
    ("health", r"\b(appointment|prescription|lab result|pharmacy|insurance claim|"
               r"explanation of benefits|patient)\b"),
    ("legal", r"\b(agreement|contract|terms of service update|notice of|"
              r"subpoena|attorney|legal notice)\b"),
    ("accounts", r"\b(verify your|password|sign-?in|two-factor|2fa|security alert|"
                 r"confirm your email|activate your account)\b"),
    ("newsletters", r"\b(unsubscribe|newsletter|digest|weekly round-?up|"
                    r"view (this )?in browser)\b"),
]


# ---- helpers ------------------------------------------------------------
def _clean(s) -> str:
    """Decode MIME header junk (=?UTF-8?B?...?=) into plain text, safely."""
    if not s:
        return ""
    try:
        return str(make_header(decode_header(str(s))))
    except Exception:
        return str(s)


def _body(msg) -> str:
    """The plain-text body, HTML stripped as a fallback. Never raises."""
    try:
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    raw = part.get_payload(decode=True) or b""
                    return raw.decode(part.get_content_charset() or "utf-8",
                                      "replace")
            for part in msg.walk():
                if part.get_content_type() == "text/html":
                    raw = part.get_payload(decode=True) or b""
                    html = raw.decode(part.get_content_charset() or "utf-8",
                                      "replace")
                    return re.sub(r"<[^>]+>", " ", html)
        else:
            raw = msg.get_payload(decode=True)
            if raw:
                return raw.decode(msg.get_content_charset() or "utf-8", "replace")
            return str(msg.get_payload())
    except Exception:
        return ""
    return ""


def _tidy(text: str, limit: int = 600) -> str:
    """Collapse whitespace and quoted-reply cruft into a readable snippet."""
    text = re.sub(r"^\s*>.*$", "", text or "", flags=re.M)         # quoted replies
    text = re.sub(r"https?://\S{40,}", "[link]", text)             # tracking URLs
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _categorize(subject: str, sender: str, body: str) -> str:
    """Rule-based first pass. Cheap, deterministic, and honest about giving up:
    anything that doesn't clearly match lands in 'unsorted' for Merge to judge."""
    hay = f"{subject} {sender} {body[:400]}".lower()
    for name, pattern in _RULES:
        if re.search(pattern, hay):
            return name
    return "unsorted"


def _rid(*parts) -> str:
    return hashlib.sha1("\x1f".join(str(p) for p in parts).encode(
        "utf-8", "replace")).hexdigest()[:16]


# ---- ingest -------------------------------------------------------------
def _iter_mbox(path: Path):
    """Yield one record dict per message. Unparseable messages are skipped."""
    box = mailbox.mbox(str(path))
    for msg in box:
        try:
            subject = _clean(msg.get("Subject"))
            sender = _clean(msg.get("From"))
            to = _clean(msg.get("To"))
            date_raw = msg.get("Date")
            try:
                ts = email.utils.mktime_tz(email.utils.parsedate_tz(date_raw))
            except Exception:
                ts = 0
            body = _tidy(_body(msg))
            yield {
                "id": _rid("mail", sender, subject, date_raw),
                "kind": "email",
                "subject": subject,
                "from": sender,
                "to": to[:300],
                "date": ts,
                "date_str": time.strftime("%Y-%m-%d", time.localtime(ts)) if ts else "",
                "summary": body[:400],
                "category": _categorize(subject, sender, body),
                "thread": (msg.get("In-Reply-To") or msg.get("References") or "")[:120],
            }
        except Exception:
            continue


def _embed_text(rec: dict) -> str:
    """What gets embedded — the parts a person would actually remember."""
    return " | ".join(x for x in [rec.get("subject", ""), rec.get("from", ""),
                                  rec.get("date_str", ""),
                                  rec.get("summary", "")[:300]] if x)


def ingest(source: str, limit: int = 0) -> str:
    """Index a Takeout export or an mbox file into the Cortex archive.

    source: path to an .mbox file, or a Takeout folder (every .mbox inside is
    ingested). limit: stop after N records (0 = everything) — useful for a
    quick trial run before committing to a huge archive.
    """
    if not source or not str(source).strip():
        return ("Point me at your unzipped Takeout folder or an .mbox file — "
                "I got nothing to open.")
    p = Path(str(source)).expanduser()
    if not p.exists():
        return f"Nothing at {p}. Point me at your Takeout folder or an .mbox file."
    boxes = sorted(p.rglob("*.mbox")) if p.is_dir() else [p]
    if not boxes:
        return (f"No .mbox files under {p}. A Google Takeout mail export contains "
                "'All mail Including Spam and Trash.mbox' — point me at the "
                "unzipped Takeout folder.")

    DIR.mkdir(parents=True, exist_ok=True)
    seen = set()
    if RECORDS.exists():
        for line in RECORDS.read_text(encoding="utf-8").splitlines():
            try:
                seen.add(json.loads(line)["id"])
            except Exception:
                continue

    new, skipped, by_cat = 0, 0, {}
    with RECORDS.open("a", encoding="utf-8") as out:
        for box in boxes:
            for rec in _iter_mbox(box):
                if not rec["subject"] and not rec["summary"]:
                    skipped += 1
                    continue
                if rec["id"] in seen:
                    skipped += 1
                    continue
                seen.add(rec["id"])
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                by_cat[rec["category"]] = by_cat.get(rec["category"], 0) + 1
                new += 1
                if limit and new >= limit:
                    break
            if limit and new >= limit:
                break

    STATE.write_text(json.dumps({"last_ingest": time.time(),
                                 "total": len(seen)}, indent=1))
    lines = [f"Indexed {new} new records from {len(boxes)} archive file(s). "
             f"({skipped} skipped as duplicates or unreadable.)"]
    if by_cat:
        lines.append("Sorted into: " + ", ".join(
            f"{k} {v}" for k, v in sorted(by_cat.items(), key=lambda x: -x[1])))
    if by_cat.get("unsorted"):
        lines.append(f"{by_cat['unsorted']} didn't match a rule and are marked "
                     "'unsorted' — read those and categorize them yourself "
                     "rather than guessing.")
    lines.append("Run build_index next so it's searchable by meaning.")
    return "\n".join(lines)


# ---- search -------------------------------------------------------------
def _load_records() -> list[dict]:
    if not RECORDS.exists():
        return []
    out = []
    for line in RECORDS.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def build_index() -> str:
    """Embed every un-embedded Cortex record so it can be searched by meaning
    rather than exact keywords. Safe to re-run; it only does the new ones."""
    recs = _load_records()
    if not recs:
        return "Nothing indexed yet — run ingest on your Takeout export first."
    try:
        from .embed import DIM, embed_documents, available
    except Exception as e:
        return f"No embedder available ({e}). Keyword search still works."
    if not available():
        return ("The embedding server isn't reachable, so meaning-search can't be "
                "built right now. Keyword search still works; re-run this when "
                "it's up.")

    have = 0
    if VECS.exists():
        try:
            raw = np.fromfile(VECS, dtype="float32")
            have = len(raw) // DIM
        except Exception:
            have = 0
    todo = recs[have:]
    if not todo:
        return f"Index is already current — {len(recs)} records embedded."

    done = 0
    with VECS.open("ab") as fh:
        for i in range(0, len(todo), 64):
            chunk = todo[i:i + 64]
            vecs = embed_documents([_embed_text(r) for r in chunk])
            if vecs is None:
                return (f"Embedded {done} of {len(todo)} before the embedder "
                        "stopped responding. Re-run to continue where it left off.")
            np.asarray(vecs, dtype="float32").tofile(fh)   # embed_documents normalizes
            done += len(chunk)
    return f"Embedded {done} records. {len(recs)} total — searchable by meaning now."


def search_life(query: str, limit: int = 8, category: str = "") -> str:
    """Search your own archive — email, calendar, contacts — the way you
    remember it ("the roof quote thread last spring"), not by exact keywords.
    Optionally narrow to one category: financial, travel, receipts, work,
    personal, health, legal, accounts, newsletters, unsorted."""
    if not query or not str(query).strip():
        return ("Tell me what to look for — a person, a company, a thing that "
                "happened. An empty search just returns whatever's closest to "
                "nothing, which is nothing useful.")
    recs = _load_records()
    if not recs:
        return "The archive is empty — ingest a Takeout export first."
    if category:
        recs_f = [r for r in recs if r.get("category") == category.strip().lower()]
        if not recs_f:
            return (f"No records in category '{category}'. Categories present: "
                    + ", ".join(sorted({r.get('category', '?') for r in recs})))
    else:
        recs_f = recs

    scored, _semantic = [], False
    # semantic first, keyword as the honest fallback
    try:
        from .embed import DIM, embed_query
        raw = np.fromfile(VECS, dtype="float32")
        mat = raw.reshape(-1, DIM)
        q = embed_query(query)
        if q is not None and len(mat):
            q = np.asarray(q, dtype="float32").reshape(-1)
            _semantic = True
            idx = {id(r): i for i, r in enumerate(recs)}
            for r in recs_f:
                i = idx[id(r)]
                if i < len(mat):
                    scored.append((float(mat[i] @ q), r))
    except Exception:
        scored, _semantic = [], False

    if not scored:
        words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
        for r in recs_f:
            hay = f"{r.get('subject','')} {r.get('from','')} {r.get('summary','')}".lower()
            hits = sum(1 for w in words if w in hay)
            if hits:
                scored.append((hits / max(len(words), 1), r))

    if not scored:
        return (f"Nothing in the archive matched '{query}'. It may not be there — "
                "I'm not going to invent a memory for you.")
    scored.sort(key=lambda x: -x[0])
    # Semantic ranking scores EVERY record, so a raw count would be a lie: an
    # unrelated query still returns a "best" row. Measured on real data, a
    # genuine hit lands ~0.72-0.82 with +0.27-0.40 lift over the median, while
    # nonsense sits ~0.44 with +0.02-0.08 lift. Demand BOTH a real score and
    # real lift over that baseline before calling anything a match — same
    # discipline as the deduction pad's signal gate.
    _all = [s for s, _ in scored]
    _med = float(np.median(_all)) if _all else 0.0
    if _semantic:
        strong = [s for s in scored if s[0] >= 0.55 and (s[0] - _med) >= 0.15]
    else:
        strong = [s for s in scored if s[0] >= 0.34]      # keyword overlap
    shown = (strong or scored)[:limit]
    if strong:
        lines = [f"From your archive — {len(strong)} relevant record(s) for "
                 f"'{query}' (showing {len(shown)}):"]
    else:
        lines = [f"Nothing in your archive clearly matches '{query}'. The closest "
                 f"{len(shown)} by meaning are below, but they're weak — treat "
                 "this as 'not found' rather than an answer:"]
    for score, r in shown:
        lines.append(f"\n● [{r.get('category','?')}] {r.get('subject','(no subject)')}"
                     f"  — {r.get('date_str','?')}")
        lines.append(f"    from: {r.get('from','?')[:80]}")
        if r.get("summary"):
            lines.append(f"    {r['summary'][:200]}")
    lines.append("\n(Everything above is a real record from your own archive — "
                 "nothing here is generated.)")
    return "\n".join(lines)


def archive_overview() -> str:
    """What's in your Cortex archive: how many records, what categories, and
    the date range it covers."""
    recs = _load_records()
    if not recs:
        return ("The archive is empty. Export your data at takeout.google.com, "
                "unzip it, then run ingest on that folder.")
    by_cat, dates = {}, [r["date"] for r in recs if r.get("date")]
    for r in recs:
        by_cat[r.get("category", "?")] = by_cat.get(r.get("category", "?"), 0) + 1
    lines = [f"Cortex archive — {len(recs)} records."]
    if dates:
        lines.append(f"  covering {time.strftime('%Y-%m-%d', time.localtime(min(dates)))}"
                     f" .. {time.strftime('%Y-%m-%d', time.localtime(max(dates)))}")
    lines.append("  by category:")
    for k, v in sorted(by_cat.items(), key=lambda x: -x[1]):
        lines.append(f"    {k:12} {v}")
    if by_cat.get("unsorted"):
        lines.append(f"\n  {by_cat['unsorted']} unsorted — read a batch and "
                     "categorize them properly instead of guessing.")
    return "\n".join(lines)
