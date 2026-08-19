"""Bioacoustics groundwork — is there STRUCTURE in animal sound?

Honest scope: this does NOT translate or decode meaning. It answers two
testable questions about a recording of animal vocalizations:

  1. REPERTOIRE — do the sounds break into a few recurring call-types, or is
     everything unique? (segment into units -> embed each -> cluster.)
  2. PROTO-SYNTAX — do those call-types come in NON-RANDOM order? Real language
     has grammar; if the sequence of calls is more predictable than shuffled
     versions of the same calls, that's a testable fingerprint of structure —
     tested against a null so we don't fool ourselves (the market-rig rule,
     pointed at animals).

Reuses the audio nerve's embedding for the per-call vectors. numpy + scipy +
PIL only. Render lets Merge SEE the repertoire, the call timeline, and the
transition "grammar".
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.signal import stft

from . import audio_nerve as AN

RENDERS = Path.home() / "forge" / "datasets" / "bioacoustics"
BG = (13, 16, 23)
INK = (230, 236, 245)
CYAN = (59, 214, 228)
GOLD = (245, 174, 61)
# distinct call-type colors
PALETTE = [(95, 227, 136), (245, 174, 61), (225, 85, 154), (155, 141, 245),
           (59, 214, 228), (255, 138, 101), (124, 196, 84), (240, 200, 90),
           (200, 120, 255), (120, 200, 255), (255, 120, 120), (160, 220, 160)]


# ---- segmentation: split a recording into call units -------------------
def _segment(sig, sr, min_ms=60, gap_ms=80):
    """Find discrete vocalizations by energy. Returns [(start, end), ...] in
    samples. Frames above an adaptive threshold, merged across short gaps."""
    hop = int(sr * 0.010)
    win = int(sr * 0.025)
    n = 1 + (len(sig) - win) // hop if len(sig) > win else 0
    if n <= 0:
        return []
    energy = np.array([np.sqrt(np.mean(sig[i * hop:i * hop + win] ** 2))
                       for i in range(n)])
    thr = energy.mean() + 0.6 * energy.std()
    thr = max(thr, energy.max() * 0.12)
    active = energy > thr
    # merge across gaps
    gap = int(gap_ms / 10)
    units, i = [], 0
    while i < n:
        if active[i]:
            j = i
            silence = 0
            while j < n and silence <= gap:
                silence = silence + 1 if not active[j] else 0
                j += 1
            s, e = i * hop, min(len(sig), j * hop)
            if (e - s) >= sr * min_ms / 1000:
                units.append((s, e))
            i = j
        else:
            i += 1
    return units


# ---- call embedding: capture the TIME×FREQUENCY SHAPE ------------------
# The audio nerve's embedding is time-AVERAGED (built for sustained/musical
# sounds), so a rising chirp and a falling chirp look identical to it. Animal
# calls are distinguished by their CONTOUR, so a call fingerprint here is a
# small low-res log-spectrogram "sketch" (freq × time) that preserves shape —
# a sweep up vs a sweep down are visibly different sketches.
def _call_embed(sig, F=12, T=8):
    if len(sig) < 64:
        sig = np.pad(sig, (0, 64 - len(sig)))
    nper = int(np.clip(len(sig) // 8, 64, 256))
    _, _, Z = stft(sig, AN.SR, nperseg=nper, noverlap=nper // 2)
    S = np.log1p(np.abs(Z))                        # freq × frames
    fi = np.linspace(0, S.shape[0], F + 1).astype(int)
    ti = np.linspace(0, S.shape[1], T + 1).astype(int)
    grid = np.zeros((F, T))
    for a in range(F):
        for b in range(T):
            blk = S[fi[a]:fi[a + 1], ti[b]:ti[b + 1]]
            grid[a, b] = blk.mean() if blk.size else 0.0
    v = grid.ravel()
    v = v - v.mean()                               # contrast, not absolute level
    return (v / (np.linalg.norm(v) or 1)).astype(np.float32)


# ---- repertoire: cluster the units -------------------------------------
def _repertoire(vecs, max_types=12):
    """Agglomerative clustering with an automatic cut. Returns a label per
    unit (1..k). Falls back to 1 cluster if too few units."""
    if len(vecs) < 3:
        return np.ones(len(vecs), dtype=int), 1
    X = np.array(vecs)
    Z = linkage(X, method="ward")
    # Pick the number of call-types by SILHOUETTE — try each k and keep the
    # cleanest split. The old largest-merge-gap heuristic under-counted badly
    # (it merged two strictly-alternating callers into one type, erasing a
    # perfect duet), so choose k by how well-separated the clusters actually
    # are, not by where the dendrogram happens to jump.
    from sklearn.metrics import silhouette_score
    # Near-identical calls (a monotone chirp repeated) must stay ONE type —
    # otherwise segmentation jitter invents spurious clusters and fake grammar.
    X0 = np.array(vecs)
    if float(np.mean(np.var(X0, axis=0))) < 1e-4:
        return np.ones(len(vecs), dtype=int), 1
    kmax = min(max_types, len(vecs) // 3)
    best_k, best_s, best_lab = 1, -1.0, np.ones(len(vecs), dtype=int)
    for k in range(2, max(3, kmax + 1)):
        lab = fcluster(Z, t=k, criterion="maxclust")
        if lab.max() < 2:
            continue
        try:
            s = silhouette_score(X, lab)
        except Exception:
            continue
        if s > best_s:
            best_k, best_s, best_lab = int(lab.max()), s, lab
    # only accept a split if it's GENUINELY clean (silhouette well clear of a
    # noise split); else it's one type. 0.4 keeps real distinct calls apart
    # while refusing to shatter one repeated call into fake types.
    if best_s < 0.4:
        return np.ones(len(vecs), dtype=int), 1
    # RECURRENCE guard: a real call-type reappears INTERLEAVED through time
    # (A-B-A-B), while a drift/noise artifact is one clumped block (AAAA-BBBB).
    # Count runs of same label; if far fewer than chance expects, the clusters
    # are temporal blocks, not recurring types — reject the split.
    seq = list(best_lab)
    runs = 1 + sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
    _, counts = np.unique(best_lab, return_counts=True)
    p2 = float(((counts / counts.sum()) ** 2).sum())
    exp_runs = 1 + (len(seq) - 1) * (1 - p2)          # expected runs if interleaved
    if runs < 0.5 * exp_runs:
        return np.ones(len(vecs), dtype=int), 1
    return best_lab, best_k


# ---- sequence structure: is the order non-random? ----------------------
def _transition_matrix(seq, k):
    M = np.zeros((k, k))
    for a, b in zip(seq[:-1], seq[1:]):
        M[a - 1, b - 1] += 1
    return M


def _predictability(seq, k):
    """Mean conditional entropy of the next symbol given the current — LOW
    means order is predictable (structured). Bits."""
    M = _transition_matrix(seq, k)
    rows = M.sum(axis=1, keepdims=True)
    P = np.divide(M, rows, out=np.zeros_like(M), where=rows > 0)
    Ps = np.where(P > 0, P, 1.0)   # log2(1)=0, so zero-prob terms vanish
    ent = -(P * np.log2(Ps)).sum(axis=1)   # per-state entropy
    weight = rows.ravel() / (rows.sum() or 1)
    return float((ent * weight).sum())


def _structure_test(seq, k, trials=400, rng_seed_len=0):
    """Compare the real sequence's predictability to shuffled versions of the
    SAME call multiset. Lower real entropy than shuffles = real ordering
    structure. Returns (real, null_mean, p_value)."""
    if k < 2 or len(seq) < 8:
        return None
    real = _predictability(seq, k)
    arr = np.array(seq)
    # deterministic shuffles (no Math.random in this env at import; use a seeded
    # generator created HERE from the data length so runs are repeatable)
    rng = np.random.RandomState(1234 + len(seq))
    null = []
    for _ in range(trials):
        s = arr.copy(); rng.shuffle(s)
        null.append(_predictability(list(s), k))
    null = np.array(null)
    # p = fraction of shuffles at least as structured (as-low-entropy) as real
    p = float((null <= real).mean())
    return real, float(null.mean()), p


# ---- render: she sees the repertoire, timeline, and grammar ------------
def _render(sig, sr, units, labels, k, M, verdict, label):
    W, H = 900, 520
    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
    d.text((12, 10), f"\U0001f9ec bioacoustics — {label}", fill=CYAN)

    # timeline of call units, colored by type
    d.text((12, 34), "CALL TIMELINE", fill=INK)
    d.text((12, 48), "each block = one call, colored by type", fill=(120, 132, 150))
    x0, x1, ty, th = 12, W - 12, 66, 42
    total = len(sig)
    d.rectangle([x0, ty, x1, ty + th], outline=(34, 42, 58))
    for (s, e), lab in zip(units, labels):
        a = x0 + (s / total) * (x1 - x0)
        b = x0 + (e / total) * (x1 - x0)
        col = PALETTE[(lab - 1) % len(PALETTE)]
        d.rectangle([a, ty, max(a + 1, b), ty + th], fill=col)

    # repertoire legend
    d.text((12, 124), f"REPERTOIRE — {k} recurring call-type(s), "
           f"{len(units)} calls total", fill=INK)
    counts = {i: int((labels == i).sum()) for i in range(1, k + 1)}
    lx = 12
    for i in range(1, k + 1):
        col = PALETTE[(i - 1) % len(PALETTE)]
        d.rectangle([lx, 144, lx + 14, 158], fill=col)
        d.text((lx + 20, 145), f"type {i} ×{counts[i]}", fill=(180, 190, 205))
        lx += 130
        if lx > W - 120:
            lx = 12

    # transition "grammar" graph
    d.text((12, 180), "SEQUENCE GRAMMAR", fill=INK)
    d.text((12, 194), "arrows = which call tends to follow which "
           "(thicker = more often)", fill=(120, 132, 150))
    gcx, gcy, gr = W // 2, 340, 110
    pos = {}
    for i in range(k):
        ang = -np.pi / 2 + i * 2 * np.pi / max(k, 1)
        pos[i + 1] = (gcx + gr * np.cos(ang), gcy + gr * np.sin(ang))
    if M is not None and M.sum() > 0:
        mx = M.max()
        for a in range(1, k + 1):
            for b in range(1, k + 1):
                w = M[a - 1, b - 1]
                if w <= 0:
                    continue
                pa, pb = pos[a], pos[b]
                width = 1 + int(3 * w / mx)
                d.line([pa, pb], fill=(40, 90, 110), width=width)
    for i in range(1, k + 1):
        px, py = pos[i]
        col = PALETTE[(i - 1) % len(PALETTE)]
        d.ellipse([px - 12, py - 12, px + 12, py + 12], fill=col)
        d.text((px - 3, py - 6), str(i), fill=(10, 14, 20))

    # verdict
    d.text((12, 470), verdict, fill=INK)
    RENDERS.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", label)[:40]
    out = RENDERS / f"{safe}.png"
    img.save(out)
    return out


def study_calls(source: str) -> str:
    """Look for STRUCTURE in a recording of animal (or any) vocalizations:
    segment it into calls, cluster them into a repertoire of recurring types,
    and test whether the SEQUENCE of calls is non-random (a testable fingerprint
    of proto-syntax). This does NOT decode meaning — it finds whether there's a
    system there. `source` is an audio file path or a synth spec. Renders a
    picture (timeline + repertoire + grammar) and returns its path + verdict."""
    sig, label = AN._load(source, max_sec=30.0)
    if sig is None:
        return f"Couldn't hear that: {label}"
    sig = sig - float(np.mean(sig))          # strip DC bias
    sig = sig / (np.abs(sig).max() or 1)
    sr = AN.SR

    units = _segment(sig, sr)
    if len(units) < 3:
        return (f"Only found {len(units)} distinct call(s) in {label} — need "
                f"more vocalizations to look for structure. (Is this a "
                f"continuous sound rather than discrete calls?)")

    vecs = [_call_embed(sig[s:e]) for (s, e) in units]
    labels, k = _repertoire(vecs)
    seq = list(labels)
    M = _transition_matrix(seq, k)
    st = _structure_test(seq, k)

    if st is None:
        verdict = (f"{len(units)} calls, {k} type(s) — too few to test ordering "
                   "yet. Structure test needs ~8+ calls and 2+ types.")
        struct_line = verdict
    else:
        real, nullm, p = st
        ordered = p < 0.05
        struct_line = (
            f"Adjacent order is {'NON-RANDOM (structured)' if ordered else 'no more predictable than chance'}: "
            f"predictability {real:.2f} bits vs {nullm:.2f} shuffled, p={p:.3f}.")
        verdict = struct_line

    out = _render(sig, sr, units, labels, k, M, struct_line[:110], label)

    lines = [f"Studied {label}: {len(units)} calls → {k} recurring call-type(s)."]
    lines.append(struct_line)
    if st and st[2] < 0.05:
        lines.append("→ There IS non-random ordering here — a testable fingerprint "
                     "of structure (NOT meaning). Worth a closer look.")
    elif st:
        lines.append("→ No ADJACENT (call-to-call) structure beyond chance. Note this "
                     "test only sees one-step patterns — longer-range or nested "
                     "structure (every 3rd call, motifs of motifs) it cannot see, so "
                     "'no adjacent structure' is NOT 'no structure'.")
    lines.append(f"Rendered to {out} — look_at_image it to SEE the repertoire, the "
                 "call timeline, and the transition grammar.")
    lines.append("Honest limit: this finds STRUCTURE, never meaning. It cannot "
                 "tell you what a call says.")
    return "\n".join(lines)


# ======================================================================
# LANGUAGE-EQUIVALENCE — how language-LIKE is the call sequence?
# Put animal calls on the same measuring sticks linguists use, with RANDOM
# noise and real HUMAN language as the two reference poles. Never claims a
# call means anything — only where the STATISTICS sit between random and
# language on each universal property.
# ======================================================================

# A chunk of ordinary English — the "language pole". Its exact content is
# irrelevant; only the universal statistics (Zipf, entropy structure) matter.
_HUMAN_TEXT = """
the morning came slow over the hills and the light moved across the fields
where the cattle stood near the water. a man walked out to the fence and
looked at the sky and thought about the rain that had not come. the river was
low and the stones showed along the bank where the birds gathered in the
early cold. he called to the dog and the dog came running through the grass
and they went down together toward the trees. the wind turned and carried the
smell of smoke from a fire somewhere past the ridge. he thought of the year
before when the water had been high and the fields were green and the work
was good. now the ground was hard and the days were long and the town was far
and quiet. still the birds came each morning to the water and the light came
over the hills and the man walked out to the fence and looked at the sky and
waited for the rain that would come or would not come but always the morning
came slow over the hills and the light moved across the fields again.
"""


def _human_seq():
    words = re.findall(r"[a-z]+", _HUMAN_TEXT.lower())
    from collections import Counter
    ranked = [w for w, _ in Counter(words).most_common()]
    idx = {w: i + 1 for i, w in enumerate(ranked)}
    return [idx[w] for w in words]


def _zipf(seq):
    """Fit log(freq) ~ log(rank). Language ≈ slope -1 with a tight fit; random
    uniform ≈ flat. Returns (slope, r2)."""
    from collections import Counter
    freqs = sorted(Counter(seq).values(), reverse=True)
    if len(freqs) < 3:
        return 0.0, 0.0
    x = np.log(np.arange(1, len(freqs) + 1))
    y = np.log(np.array(freqs))
    A = np.vstack([x, np.ones_like(x)]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = A @ np.linalg.lstsq(A, y, rcond=None)[0]
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum() or 1
    return float(slope), float(1 - ss_res / ss_tot)


def _entropy_order(seq, order):
    """Conditional entropy of the next symbol given `order` previous ones
    (bits). Language DROPS with order (context predicts); random stays flat."""
    from collections import Counter, defaultdict
    if len(seq) <= order + 1:
        return 0.0
    ctx = defaultdict(Counter)
    for i in range(len(seq) - order):
        key = tuple(seq[i:i + order])
        ctx[key][seq[i + order]] += 1
    total = sum(sum(c.values()) for c in ctx.values())
    H = 0.0
    for c in ctx.values():
        n = sum(c.values())
        h = -sum((v / n) * np.log2(v / n) for v in c.values())
        H += (n / total) * h
    return float(H)


def _herf(seq):
    """Concentration (Herfindahl) of the bigram distribution — high when a few
    specific call-combinations dominate."""
    from collections import Counter
    if len(seq) < 3:
        return 0.0
    bg = Counter(tuple(seq[i:i + 2]) for i in range(len(seq) - 1))
    tot = sum(bg.values())
    return float(sum((n / tot) ** 2 for n in bg.values()))


def _motif_score(seq):
    """Combinatoriality, measured ALPHABET-FAIR: how much more the sequence
    concentrates on SPECIFIC call-combinations than its OWN shuffle would.
    0.5 = no more than chance; higher = reuses particular combos (language-
    like). Comparing to a shuffle of the same symbols cancels the tiny-
    repertoire problem that makes raw counts meaningless."""
    if len(seq) < 6:
        return 0.5
    real = _herf(seq)
    rng = np.random.RandomState(5)
    arr = np.array(seq); nl = []
    for _ in range(120):
        s = arr.copy(); rng.shuffle(s); nl.append(_herf(list(s)))
    nm = float(np.mean(nl)); ns = float(np.std(nl)) or 1e-9
    z = (real - nm) / ns
    return float(np.clip(0.5 + z / 6, 0, 1))


def _menzerath(units, labels, sig):
    """Menzerath's law: longer sequences are built of shorter parts. Bouts =
    call groups split by longer gaps; test whether longer bouts have shorter
    average call duration (negative correlation). Animal-only (needs durations)."""
    if len(units) < 6:
        return None
    durs = [(e - s) for s, e in units]
    gaps = [units[i + 1][0] - units[i][1] for i in range(len(units) - 1)]
    if not gaps:
        return None
    thr = float(np.median(gaps)) * 1.8
    bouts, cur = [], [0]
    for i, g in enumerate(gaps):
        if g > thr:
            bouts.append(cur); cur = [i + 1]
        else:
            cur.append(i + 1)
    bouts.append(cur)
    bouts = [b for b in bouts if len(b) >= 2]
    if len(bouts) < 3:
        return None
    lengths = np.array([len(b) for b in bouts])
    meandur = np.array([np.mean([durs[i] for i in b]) for b in bouts])
    if lengths.std() == 0 or meandur.std() == 0:
        return None
    return float(np.corrcoef(lengths, meandur)[0, 1])


def _pole(v, lo, hi):
    """Where does v sit between the random pole (lo) and language pole (hi)?
    0 = random-like, 1 = language-like, clamped."""
    if hi == lo:
        return 0.5
    return float(np.clip((v - lo) / (hi - lo), 0, 1))


def _lang_render(metrics, label, verdict):
    W, H = 820, 430
    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
    d.text((12, 10), f"\U0001f5e3 language scorecard — {label}", fill=CYAN)
    d.text((12, 30), "where the animal sits between RANDOM and HUMAN LANGUAGE on "
           "each universal property", fill=(120, 132, 150))
    rows = [("Zipf's law (word-frequency shape)", "zipf"),
            ("grammar depth (context predicts next)", "grammar"),
            ("combinatoriality (reusable motifs)", "motif")]
    y = 64
    for title, key in rows:
        m = metrics[key]
        d.text((12, y), title, fill=INK)
        bx, bw, by = 12, W - 220, y + 20
        d.rectangle([bx, by, bx + bw, by + 16], outline=(40, 48, 64))
        d.text((bx, by + 22), "random", fill=(90, 100, 116))
        d.text((bx + bw - 46, by + 22), "language", fill=(90, 100, 116))
        px = bx + int(m["pos"] * bw)
        d.ellipse([px - 7, by - 3, px + 7, by + 19], fill=GOLD)
        d.text((W - 200, by), f"animal {m['animal']:.2f}  ·  "
               f"rand {m['rand']:.2f} / lang {m['lang']:.2f}", fill=(180, 190, 205))
        y += 74
    if metrics.get("menzerath") is not None:
        d.text((12, y), f"Menzerath's law: bout-length vs call-length corr "
               f"{metrics['menzerath']:+.2f}  "
               f"({'holds (language-like)' if metrics['menzerath'] < -0.2 else 'not seen'})",
               fill=(180, 190, 205))
        y += 24
    for i, ln in enumerate(verdict.split("\n")):
        d.text((12, H - 60 + i * 14), ln, fill=INK)
    RENDERS.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", label)[:40]
    out = RENDERS / f"lang_{safe}.png"
    img.save(out)
    return out


def language_scorecard(source: str) -> str:
    """How LANGUAGE-LIKE is a recording's call sequence? Segments and clusters
    it (like study_calls), then scores the call sequence on the universal
    properties of human language — Zipf's law, grammar depth (entropy that
    drops with context), combinatoriality, Menzerath's law — against two poles:
    RANDOM and real human language. Says where the animal falls between them.
    Never claims meaning or that it IS a language. `source` = audio file path."""
    sig, label = AN._load(source, max_sec=30.0)
    if sig is None:
        return f"Couldn't hear that: {label}"
    sig = sig - float(np.mean(sig))          # strip DC bias
    sig = sig / (np.abs(sig).max() or 1)
    units = _segment(sig, AN.SR)
    if len(units) < 10:
        return (f"Only {len(units)} calls in {label} — need ~10+ to score against "
                "language. Try a longer recording.")
    vecs = [_call_embed(sig[s:e]) for (s, e) in units]
    labels, k = _repertoire(vecs)
    seq = list(labels)

    human = _human_seq()
    rng = np.random.RandomState(99)
    rand = list(rng.randint(1, max(k, 2) + 1, len(seq)))

    def bundle(s):
        slope, r2 = _zipf(s)
        h0 = _entropy_order(s, 0); h1 = _entropy_order(s, 1)
        drop = (h0 - h1) / (h0 or 1)          # fraction of uncertainty context removes
        return {"zipf": abs(slope) * r2, "grammar": drop, "motif": _motif_score(s)}

    A_, H_, R_ = bundle(seq), bundle(human), bundle(rand)
    metrics = {}
    for key in ("zipf", "grammar", "motif"):
        metrics[key] = {"animal": A_[key], "human": H_[key], "lang": H_[key],
                        "rand": R_[key], "pos": _pole(A_[key], R_[key], H_[key])}
    metrics["menzerath"] = _menzerath(units, labels, sig)

    avg_pos = np.mean([metrics[k]["pos"] for k in ("zipf", "grammar", "motif")])
    if avg_pos > 0.66:
        read = "strikingly language-like on these statistics"
    elif avg_pos > 0.4:
        read = "partway between random and language — some language-like structure"
    else:
        read = "closer to random than to language on these measures"
    verdict = (f"{len(units)} calls, {k} call-types. Overall this sits ~{avg_pos*100:.0f}% "
               f"of the way from random toward language — {read}.")

    out = _lang_render(metrics, label, verdict)
    lines = [f"LANGUAGE SCORECARD — {label} ({len(units)} calls, {k} types):",
             f"  Zipf shape:       animal {A_['zipf']:.2f}  (random {R_['zipf']:.2f} → language {H_['zipf']:.2f})  = {metrics['zipf']['pos']*100:.0f}% language-like",
             f"  grammar depth:    animal {A_['grammar']:.2f}  (random {R_['grammar']:.2f} → language {H_['grammar']:.2f})  = {metrics['grammar']['pos']*100:.0f}% language-like",
             f"  combinatoriality: animal {A_['motif']:.2f}  (random {R_['motif']:.2f} → language {H_['motif']:.2f})  = {metrics['motif']['pos']*100:.0f}% language-like"]
    if metrics["menzerath"] is not None:
        m = metrics["menzerath"]
        lines.append(f"  Menzerath's law:  corr {m:+.2f}  ({'HOLDS — a language universal appears' if m < -0.2 else 'not seen in this sample'})")
    lines.append(f"\n{verdict}")
    lines.append("A real caveat: animal repertoires are TINY (a few call-types) next to "
                 "human vocabularies (thousands) — so Zipf here is inherently limited, and "
                 "a small repertoire is itself one honest difference from language.")
    lines.append(f"Rendered to {out} — look_at_image it to SEE the scorecard.")
    lines.append("HARD LIMIT: this measures statistical resemblance, NEVER meaning. "
                 "Language-like structure is not language, and nobody can translate a call.")
    return "\n".join(lines)
