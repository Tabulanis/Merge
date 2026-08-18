"""Writing-craft reflexes: prose statistics and canon name consistency.

Deterministic text analysis for an agent that co-writes: exact counts a
model would otherwise eyeball, and fuzzy name scanning so a misspelled
character can't quietly fork into two people across the world files.
"""
from __future__ import annotations

import re
from pathlib import Path

# Words too common to be interesting in a repetition report.
_STOPWORDS = set("""a an and are as at be but by for from had has have he her
his i if in into is it its me my not of on or she so that the their them
they this to was we were what when where which who will with you your would
could should there here than then over just like about out up down off all
been being can did do does going gone got very really said says say one two
some any no yes s t don didn't it's i'm""".split())

_WORD_RE = re.compile(r"[A-Za-z']+")


def text_stats(text: str) -> str:
    """Exact prose statistics for a draft passed in as text."""
    if not text.strip():
        return "Error: no text given — paste the draft into `text`."
    words = _WORD_RE.findall(text)
    n_words = len(words)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    dialogue_words = sum(len(_WORD_RE.findall(m))
                         for m in re.findall(r'"[^"]+"|“[^”]+”', text))
    freq: dict[str, int] = {}
    for w in words:
        lw = w.lower().strip("'")
        if len(lw) >= 4 and lw not in _STOPWORDS:
            freq[lw] = freq.get(lw, 0) + 1
    repeated = sorted(((n, w) for w, n in freq.items() if n >= 3), reverse=True)[:10]
    longest = max(sentences, key=lambda s: len(_WORD_RE.findall(s)), default="")
    minutes = n_words / 200.0
    out = [f"words:        {n_words}",
           f"sentences:    {len(sentences)}"
           + (f"  (avg {n_words / len(sentences):.1f} words each)" if sentences else ""),
           f"paragraphs:   {len(paragraphs)}",
           f"reading time: {minutes:.1f} min at 200 wpm",
           f"dialogue:     {dialogue_words} words in quotes"
           + (f" ({100 * dialogue_words / n_words:.0f}%)" if n_words else "")]
    if repeated:
        out.append("repeated words (3+ uses, most first): "
                   + ", ".join(f"{w} x{n}" for n, w in repeated))
    if longest:
        shown = longest if len(longest) <= 160 else longest[:157] + "..."
        out.append(f"longest sentence ({len(_WORD_RE.findall(longest))} words): {shown}")
    return "\n".join(out)


# Phrases that mark machine-flavored prose. Not banned words — flags to
# eyeball. Fixing them is editing for voice, which also happens to drop
# every AI-detector score, because detectors hunt these same tells.
_AI_TELLS = [
    "delve", "tapestry", "testament to", "navigate the", "navigating the",
    "in the realm of", "it's important to note", "it is important to note",
    "it's worth noting", "when it comes to", "a myriad of", "plethora",
    "moreover", "furthermore", "in conclusion", "ultimately,", "in today's",
    "ever-evolving", "ever-changing", "rich history", "gain a deeper",
    "shed light on", "play a crucial role", "a wide range of", "aims to",
    "underscore", "pivotal", "landscape of", "at its core", "on the other hand",
]


# Fiction autopilot — the phrases every model reaches for when writing a
# scene. These matter MORE than the essay tells for a co-writer: they're the
# "shivers down her spine" family, invisible one at a time, deadening in bulk.
# Regexes so pronoun and tense variants all get caught.
_FICTION_CLICHES = [(label, re.compile(rx, re.I)) for label, rx in [
    ("shiver/chill down the spine", r"(shiver|chill)s?\b.{0,20}\bspine"),
    ("sent a shiver/chill",         r"sent a (shiver|chill|jolt)"),
    ("a breath they didn't know they were holding",
                                    r"breath (she|he|they|i)\s+(did ?n'?t|had ?n'?t)\s+\w*\s*(know|realiz|realis)"),
    ("let out / released a breath", r"(let out|released|exhaled|drew) a (shaky |ragged |slow )?breath"),
    ("heart hammered/pounded/raced",r"heart\s+(hammer|pound|race|thud|thump|lurch|clench|sank|sk)\w*"),
    ("pulse quickened",             r"pulse\s+(quicken|race|pound|hammer)\w*"),
    ("breath caught/hitched",       r"breath\s+(caught|hitch|snag)\w*"),
    ("caught their breath",         r"caught (her|his|their|my) breath"),
    ("sharp intake of breath",      r"sharp intake of breath"),
    ("little did they know",        r"little did (she|he|they|i) know"),
    ("time slowed / stood still",   r"time\s+(seemed to slow|slow\w*|stood still|froze)"),
    ("eyes widened/narrowed",       r"eyes\s+(widen|narrow)\w*"),
    ("knot in their stomach",       r"(knot|tightness)\b.{0,15}stomach"),
    ("stomach churned/twisted/dropped", r"stomach\s+(churn|twist|knot|drop|lurch|clench)\w*"),
    ("jaw clenched/tightened",      r"jaw\s+(clench|tighten|tick|work)\w*"),
    ("blood ran cold",              r"blood ran cold"),
    ("hairs on the back of the neck", r"hairs?\b.{0,20}back of (her|his|their|my) neck"),
    ("barely above a whisper",      r"barely (above )?a whisper"),
    ("couldn't help but",           r"could ?n'?t help but"),
    ("whirlwind of emotion",        r"whirlwind of (emotion|feeling)"),
    ("the air was thick/heavy with", r"(the )?air\s+(was |grew |hung )\w*\s*(thick|heavy|electric)"),
    ("tears welled / a single tear", r"(tears?\s+(well|prick|stream|brim)\w*|a single tear)"),
    ("world came crashing / shattered", r"world\s+(came crashing|shatter|crumbl)\w*"),
    ("visibly relaxed/tensed",      r"visibly (relax|tens|stiffen|flinch)\w*"),
    ("a wave of <emotion>",         r"a wave of (relief|nausea|dread|emotion|anger|calm|sadness)"),
    ("breath he/she was holding",   r"the breath (she|he|they|i)('?d| had)? been holding"),
]]


def ai_tells(text: str) -> str:
    """Deterministic 'how machine-flavored is this?' meter. Measures the same
    fingerprints AI detectors use — sentence-rhythm uniformity, essay-cliche
    AND fiction-cliche density, punctuation tics — and points at the flat
    spots. Fixing what it flags is editing for voice; any detector score
    drops as a side effect."""
    if not text.strip():
        return "Error: paste the prose into `text`."
    words = _WORD_RE.findall(text)
    n_words = len(words) or 1
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    lengths = [len(_WORD_RE.findall(s)) for s in sentences] or [0]
    mean_len = sum(lengths) / len(lengths)
    var = sum((x - mean_len) ** 2 for x in lengths) / len(lengths)
    stdev = var ** 0.5
    burstiness = stdev / mean_len if mean_len else 0.0   # human prose ~0.5+

    tells = []
    low = text.lower()
    for phrase in _AI_TELLS:
        c = low.count(phrase)
        if c:
            tells.append((c, phrase))
    tells.sort(reverse=True)
    tell_total = sum(c for c, _ in tells)

    fiction = []
    for label, rx in _FICTION_CLICHES:
        hits = rx.findall(text)
        if hits:
            fiction.append((len(hits), label))
    fiction.sort(reverse=True)
    fiction_total = sum(c for c, _ in fiction)

    em_dash = text.count("—") + text.count(" - ")
    semicolons = text.count(";")

    openers: dict[str, int] = {}
    for s in sentences:
        first = (_WORD_RE.findall(s) or [""])[0].lower()
        if first:
            openers[first] = openers.get(first, 0) + 1
    repeated_openers = sorted(((n, w) for w, n in openers.items() if n >= 3),
                              reverse=True)[:5]

    triples = len(re.findall(r"\w+, \w+,? and \w+", text))
    not_just = len(re.findall(r"not (just|only) .{3,40}?,? (but|it)", low))

    # A rough 0-100 "machine flavor" score — high = flatter/more detectable.
    score = 0
    score += 30 if burstiness < 0.35 else 18 if burstiness < 0.5 else 0
    score += min(25, tell_total * 6)
    score += min(50, fiction_total * 10)   # fiction autopilot weighted heaviest
    score += min(10, (em_dash + semicolons) * 100 // n_words * 3)
    score += min(8, len(repeated_openers) * 4)
    score += min(8, (triples + not_just) * 3)
    score = min(100, score)
    verdict = ("reads human" if score < 25 else
               "some machine flavor" if score < 50 else "flat / machine-flavored")

    out = [f"MACHINE-FLAVOR SCORE: {score}/100 — {verdict}",
           f"(lower is more human; this is deterministic, not an AI detector, "
           f"but it measures what detectors measure)",
           "",
           f"sentence rhythm (burstiness): {burstiness:.2f}  "
           f"({'FLAT — vary sentence lengths' if burstiness < 0.5 else 'good variation'}); "
           f"lengths run {min(lengths)}-{max(lengths)} words, avg {mean_len:.0f}",
           f"em-dashes + semicolons: {em_dash + semicolons} "
           f"({'heavy — a classic tell' if (em_dash + semicolons) * 100 // n_words >= 2 else 'fine'})"]
    if fiction:
        out.append("FICTION AUTOPILOT (the 'shivers down her spine' family — "
                   "the ones to hunt): "
                   + ", ".join(f"{lbl} x{c}" for c, lbl in fiction[:10]))
    else:
        out.append("fiction autopilot clichés: none found")
    if tells:
        out.append("essay/AI-tell phrases: "
                   + ", ".join(f"{p} x{c}" for c, p in tells[:8]))
    else:
        out.append("essay/AI-tell phrases: none found")
    if repeated_openers:
        out.append("repeated sentence openers: "
                   + ", ".join(f"'{w}' x{n}" for n, w in repeated_openers))
    if triples or not_just:
        out.append(f"structural tics: {triples} tricolon(s) (X, Y, and Z), "
                   f"{not_just} 'not just X but Y'")
    out.append("")
    out.append("To lower the score, edit for voice: break the rhythm with a "
               "very short sentence next to a long one, cut the cliche phrases, "
               "and replace generic words with specific ones. Do NOT do this to "
               "deceive a gatekeeper who forbids AI help — only to make the "
               "writing genuinely yours.")
    return "\n".join(out)


def _levenshtein(a: str, b: str, cap: int) -> int:
    """Edit distance, giving up early past `cap` (rows only grow)."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def name_check(root: str, name: str, max_distance: int = 1) -> str:
    """Scan the workspace's text files for exact uses and near-miss
    spellings of a name. A near-miss in canon is usually a typo about to
    become a second character."""
    name = (name or "").strip()
    if not name or not _WORD_RE.fullmatch(name):
        return "Error: give one name (letters only) in `name`."
    max_distance = max(1, min(int(max_distance), 3))
    target = name.lower()
    exact = 0
    near: dict[str, list[str]] = {}
    scanned = 0
    for path in sorted(Path(root).rglob("*")):
        if path.suffix.lower() not in (".md", ".txt") or not path.is_file():
            continue
        parts = set(path.parts)
        if parts & {".git", "node_modules", ".venv", "__pycache__"}:
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        scanned += 1
        rel = str(path.relative_to(root))
        for ln_no, line in enumerate(lines, 1):
            for word in _WORD_RE.findall(line):
                lw = word.lower().strip("'")
                # A possessive is the same name, not a variant spelling.
                if lw.endswith("'s"):
                    lw = lw[:-2]
                if lw == target:
                    exact += 1
                elif (abs(len(lw) - len(target)) <= max_distance
                      and lw not in _STOPWORDS and len(lw) >= 3
                      and _levenshtein(lw, target, max_distance) <= max_distance):
                    near.setdefault(word, []).append(f"{rel}:{ln_no}")
    out = [f"'{name}' in {scanned} text files: {exact} exact use(s)."]
    if near:
        out.append(f"NEAR-MISS spellings (edit distance <= {max_distance}) — "
                   "check whether these are typos or genuinely different names:")
        for variant, locs in sorted(near.items(), key=lambda kv: -len(kv[1])):
            shown = ", ".join(locs[:5]) + (f" (+{len(locs) - 5} more)" if len(locs) > 5 else "")
            out.append(f"  {variant} x{len(locs)}: {shown}")
    else:
        out.append("No near-miss spellings found.")
    return "\n".join(out[:120])
