"""The audio nerve — Merge's ears, routed through her eyes.

She can see but not hear. So this turns sound into a picture of its STRUCTURE
that her vision can read: the waveform, the spectrogram (time x frequency), the
harmonic spectrum (a pitched note is a stack of overtones at integer ratios —
the symmetry of the harmonic series, made visible), and a pitch-class mandala
(the 12 tones on a circle with the harmony's geometry drawn on it — consonance
reads as symmetry, dissonance as a broken shape).

Sound in: an audio file (any format, via ffmpeg) OR a synth spec so she can
study a pure structure — "note:A4", "chord:major:C", "interval:fifth",
"harmonics:220". Picture out: one composed PNG she look_at_image's.

numpy + scipy + PIL only. No matplotlib, no librosa.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.fft import dct
from scipy.signal import stft

VEC_STORE = Path.home() / "forge" / "datasets" / "audio-nerve" / "vectors.jsonl"

SR = 22050
RENDERS = Path.home() / "forge" / "datasets" / "audio-nerve"
BG = (13, 16, 23)
INK = (230, 236, 245)
CYAN = (59, 214, 228)
GOLD = (245, 174, 61)

_NOTES = {"C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "F": 5,
          "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11}
_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_CHORDS = {"major": [0, 4, 7], "minor": [0, 3, 7], "dim": [0, 3, 6],
           "aug": [0, 4, 8], "maj7": [0, 4, 7, 11], "min7": [0, 3, 7, 10],
           "dom7": [0, 4, 7, 10], "sus4": [0, 5, 7]}
_INTERVALS = {"unison": 0, "minor2": 1, "major2": 2, "minor3": 3, "major3": 4,
              "fourth": 5, "tritone": 6, "fifth": 7, "minor6": 8, "major6": 9,
              "minor7": 10, "major7": 11, "octave": 12}


def _note_hz(name: str, default_oct: int = 4) -> float:
    m = re.match(r"([A-Ga-g][#b]?)(-?\d+)?$", name.strip())
    if not m:
        return 261.63
    pc = _NOTES.get(m.group(1).upper(), 0)
    octv = int(m.group(2)) if m.group(2) else default_oct
    octv = max(-1, min(12, octv))          # audible range; avoids 2**huge overflow
    midi = 12 * (octv + 1) + pc
    return 440.0 * 2 ** ((midi - 69) / 12)


def _tone(f0: float, t: np.ndarray, partials: int = 6) -> np.ndarray:
    """A pitched tone = fundamental + overtones at 1/n amplitude (natural-ish)."""
    return sum((1.0 / n) * np.sin(2 * np.pi * f0 * n * t) for n in range(1, partials + 1))


def _synth(spec: str, dur: float = 2.0):
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    parts = spec.split(":")
    kind = parts[0].lower()
    if kind == "note":
        sig = _tone(_note_hz(parts[1] if len(parts) > 1 else "A4"), t)
        label = f"note {parts[1] if len(parts) > 1 else 'A4'}"
    elif kind == "harmonics":
        try:
            f0 = float(parts[1]) if len(parts) > 1 else 220.0
        except (ValueError, IndexError):
            f0 = 220.0
        if not np.isfinite(f0) or f0 <= 0:
            f0 = 220.0
        sig = _tone(f0, t, partials=10)
        label = f"harmonic series on {f0:g} Hz"
    elif kind == "interval":
        name = parts[1].lower() if len(parts) > 1 else "fifth"
        root = _note_hz(parts[2] if len(parts) > 2 else "C4")
        semis = _INTERVALS.get(name, 7)
        sig = _tone(root, t) + _tone(root * 2 ** (semis / 12), t)
        label = f"interval: {name}"
    elif kind == "chord":
        quality = parts[1].lower() if len(parts) > 1 else "major"
        root = parts[2] if len(parts) > 2 else "C4"
        rf = _note_hz(root)
        sig = sum(_tone(rf * 2 ** (s / 12), t) for s in _CHORDS.get(quality, [0, 4, 7]))
        label = f"chord: {root} {quality}"
    else:
        sig = _tone(_note_hz("A4"), t)
        label = "note A4"
    env = np.minimum(1, np.minimum(t * 20, (dur - t) * 8))     # gentle fade
    out = np.nan_to_num(sig * env, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32), label


def _load(source: str, max_sec: float = 8.0):
    """Return (samples float32 mono @ SR, human label). File path -> ffmpeg
    decode; 'kind:...' -> synth."""
    _kind = source.split(":", 1)[0].lower()
    if _kind in ("note", "harmonics", "interval", "chord") and not Path(source).exists():
        return _synth(source)
    p = Path(source)
    if not p.exists():
        return None, f"no such audio: {source}"
    out = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", str(p), "-f", "f32le", "-ac", "1",
         "-ar", str(SR), "-t", str(max_sec), "-"],
        capture_output=True).stdout
    a = np.frombuffer(out, dtype="<f4").copy()
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)   # a NaN/inf sample must not poison analysis
    if a.size == 0:
        return None, f"couldn't decode any audio from {p.name} (unsupported/corrupt?)"
    if a.size < SR // 10:
        return None, (f"{p.name} is too short to analyze "
                      f"({a.size / SR:.2f}s) — need about a tenth of a second or more")
    return a, p.name


# ---- colormap (magma-ish, perceptual, no matplotlib) --------------------
_CMAP = np.array([(0, 0, 4), (40, 11, 84), (101, 21, 110), (159, 42, 99),
                  (212, 72, 66), (245, 125, 21), (250, 193, 39), (252, 255, 164)],
                 dtype=np.float32)


def _color(x):
    x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=1.0, neginf=0.0)
    x = np.clip(x, 0, 1) * (len(_CMAP) - 1)
    i = np.floor(x).astype(int); f = (x - i)[..., None]
    i2 = np.minimum(i + 1, len(_CMAP) - 1)
    return (_CMAP[i] * (1 - f) + _CMAP[i2] * f).astype(np.uint8)


# ---- panels --------------------------------------------------------------
def _spectrogram(sig, W, H):
    f, tt, Z = stft(sig, SR, nperseg=1024, noverlap=768)
    S = np.abs(Z)
    # log-frequency remap so octaves are evenly spaced (musical view)
    fmin, fmax = 40, SR / 2
    logf = np.geomspace(fmin, fmax, H)
    idx = np.clip(np.searchsorted(f, logf), 0, len(f) - 1)
    S = S[idx, :]
    S = np.log1p(S * 8)
    S = S / (S.max() or 1)
    rgb = _color(S[::-1])                       # low freq at bottom
    img = Image.fromarray(rgb).resize((W, H), Image.BILINEAR)
    return img


def _waveform(sig, W, H):
    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
    n = len(sig); step = max(1, n // W); mid = H // 2
    pts = []
    for x in range(W):
        seg = sig[x * step:(x + 1) * step]
        a = float(np.abs(seg).max()) if len(seg) else 0
        pts.append((x, mid - a * mid * 0.9)); pts.append((x, mid + a * mid * 0.9))
    for x in range(0, W):
        d.line([(x, pts[2 * x][1]), (x, pts[2 * x + 1][1])], fill=CYAN)
    return img


def _harmonic(sig, W, H):
    """Average magnitude spectrum with the overtone ladder — the symmetry of
    the harmonic series shows as evenly-spaced peaks."""
    win = np.hanning(len(sig))
    mag = np.abs(np.fft.rfft(sig * win))
    freqs = np.fft.rfftfreq(len(sig), 1 / SR)
    keep = freqs <= 2200
    mag = mag[keep]; freqs = freqs[keep]
    mag = mag / (mag.max() or 1)
    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
    base = H - 16
    for i in range(1, len(mag)):
        x = int((i / len(mag)) * W)
        y = base - mag[i] * (H - 30)
        col = GOLD if mag[i] > 0.12 else (70, 82, 100)
        d.line([(x, base), (x, y)], fill=col)
    d.line([(0, base), (W, base)], fill=(50, 60, 76))
    return img


def _chroma_vector(sig):
    f, tt, Z = stft(sig, SR, nperseg=2048, noverlap=1024)
    S = np.abs(Z).mean(axis=1)
    chroma = np.zeros(12)
    for i, fr in enumerate(f):
        if fr < 55 or fr > 4000:
            continue
        pc = int(round(12 * np.log2(fr / 16.35))) % 12
        chroma[pc] += S[i]
    return chroma / (chroma.max() or 1)


# ---- the audio embedding: a sound as a vector ----------------------------
def _mel_fb(n_filts, nfft, fmin=40, fmax=None):
    fmax = fmax or SR / 2
    hz2mel = lambda h: 2595 * np.log10(1 + h / 700)
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_filts + 2))
    bins = np.floor((nfft + 1) * pts / SR).astype(int)
    fb = np.zeros((n_filts, nfft // 2 + 1))
    for i in range(1, n_filts + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        for k in range(l, c):
            fb[i - 1, k] = (k - l) / (c - l or 1)
        for k in range(c, r):
            fb[i - 1, k] = (r - k) / (r - c or 1)
    return fb


def embed(sig) -> np.ndarray:
    """A sound → one compact, L2-normalized feature vector: pitch content
    (chroma, 12) + timbre (MFCC-like, 13) + spectral character (6). Two sounds
    that feel alike sit close in this space, so she can compare and recall
    sounds by similarity — the way her memory recalls meaning."""
    chroma = _chroma_vector(sig)                          # 12: what notes
    f, tt, Z = stft(sig, SR, nperseg=1024, noverlap=512)
    P = np.abs(Z) ** 2                                    # power: freq x frames
    Pm = P.mean(axis=1) + 1e-10                           # avg power spectrum
    # timbre: log-mel → DCT (MFCC), mean across time
    mel = _mel_fb(26, 1024) @ P
    mfcc = dct(np.log(mel + 1e-10), axis=0, norm="ortho")[:13].mean(axis=1)
    # spectral character
    centroid = float((f * Pm).sum() / Pm.sum())
    spread = float(np.sqrt(((f - centroid) ** 2 * Pm).sum() / Pm.sum()))
    cs = np.cumsum(Pm)
    rolloff = float(f[np.searchsorted(cs, 0.85 * cs[-1])])
    flatness = float(np.exp(np.log(Pm).mean()) / Pm.mean())     # tonal↔noisy
    zcr = float(np.mean(np.abs(np.diff(np.sign(sig)))) / 2)
    rms = float(np.sqrt(np.mean(sig ** 2)))
    nyq = SR / 2
    spec = np.array([centroid / nyq, spread / nyq, rolloff / nyq, flatness, zcr, rms])
    feat = np.concatenate([chroma, mfcc / 20.0, spec]).astype(np.float32)
    return feat / (np.linalg.norm(feat) or 1)


def _remember(label: str, source: str, vec: np.ndarray) -> None:
    VEC_STORE.parent.mkdir(parents=True, exist_ok=True)
    with VEC_STORE.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps({"t": time.time(), "label": label,
                             "source": source, "vec": vec.round(4).tolist()}) + "\n")


def _load_store():
    rows = []
    if VEC_STORE.exists():
        for ln in VEC_STORE.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(ln)
                r["v"] = np.array(r["vec"], dtype=np.float32)
                rows.append(r)
            except Exception:
                continue
    return rows


def match_sound(source: str, top: int = 5) -> str:
    """Hear a sound and recall the ones most like it. Embeds `source` (audio
    file path OR synth spec like 'chord:major:C') and returns the nearest
    remembered sounds by similarity — her ear-memory, searched by likeness."""
    sig, label = _load(source)
    if sig is None:
        return f"Couldn't hear that: {label}"
    q = embed(sig / (np.abs(sig).max() or 1))
    if not np.isfinite(q).all():
        return "Couldn't fingerprint that sound — its features weren't finite."
    rows = [r for r in _load_store()
            if r["v"].shape == q.shape and np.isfinite(r["v"]).all()]
    if not rows:
        return ("No sounds remembered yet — use see_sound on a few first "
                "(each one it shows also gets saved as a vector).")
    scored = sorted(((float(q @ r["v"]), r) for r in rows),
                    key=lambda x: -x[0])
    seen, out = set(), []
    for sim, r in scored:
        if r["label"] in seen:
            continue
        seen.add(r["label"])
        out.append(f"  {sim:+.3f}  {r['label']}")
        if len(out) >= top:
            break
    return (f"'{label}' is most like these remembered sounds "
            f"(cosine similarity; these fingerprints run high, ~0.5\u20131.0, so "
            f"the RANKING matters more than the raw number):\n" + "\n".join(out))


def _mandala(sig, W, H):
    """The 12 pitch classes on a circle, each lit by its energy, with the
    harmony's geometry drawn between the active tones. Consonant shapes read
    symmetric; dissonant ones read broken. This is the beauty made visible."""
    chroma = _chroma_vector(sig)
    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
    cx, cy, R = W // 2, H // 2, min(W, H) // 2 - 26
    # spokes + labels
    pos = []
    for pc in range(12):
        ang = -np.pi / 2 + pc * 2 * np.pi / 12          # C at top, clockwise
        x, y = cx + R * np.cos(ang), cy + R * np.sin(ang)
        pos.append((x, y))
        d.line([(cx, cy), (x, y)], fill=(34, 42, 58))
        lx, ly = cx + (R + 14) * np.cos(ang), cy + (R + 14) * np.sin(ang)
        d.text((lx - 4, ly - 6), _NAMES[pc], fill=(120, 132, 150))
    # the harmony polygon between active tones
    active = [i for i in range(12) if chroma[i] > 0.35]
    if len(active) >= 2:
        poly = [pos[i] for i in active]
        d.polygon(poly, outline=CYAN)
        for i in active:
            for j in active:
                if i < j:
                    d.line([pos[i], pos[j]], fill=(30, 90, 100))
    # energy blobs on each tone
    for pc in range(12):
        e = chroma[pc]
        r = 3 + e * 16
        col = _color(np.array(e))[()]
        col = tuple(int(c) for c in col)
        d.ellipse([pos[pc][0] - r, pos[pc][1] - r, pos[pc][0] + r, pos[pc][1] + r],
                  fill=col if e > 0.08 else (28, 34, 46))
    d.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(60, 70, 86))
    return img


def _label(img, text, sub=""):
    d = ImageDraw.Draw(img)
    d.text((10, 6), text, fill=INK)
    if sub:
        d.text((10, 20), sub, fill=(120, 132, 150))
    return img


def see_sound(source: str) -> str:
    """Turn a sound into a picture of its structure Merge can SEE — waveform,
    spectrogram, harmonic spectrum, and pitch-class mandala. `source` is an
    audio file path OR a synth spec: 'note:A4', 'chord:major:C',
    'interval:fifth', 'harmonics:220'. Saves a PNG and returns its path (so it
    shows in the chat and she can look_at_image it)."""
    sig, label = _load(source)
    if sig is None:
        return f"Couldn't hear that: {label}"
    sig = sig / (np.abs(sig).max() or 1)

    W = 900
    spec = _spectrogram(sig, W, 300); _label(spec, "SPECTROGRAM", "time →   ·   frequency (log) ↑")
    wave = _waveform(sig, W, 90); _label(wave, "WAVEFORM")
    harm = _harmonic(sig, W // 2, 240); _label(harm, "HARMONIC SPECTRUM", "overtones at integer ratios")
    mand = _mandala(sig, W - W // 2, 240); _label(mand, "PITCH MANDALA", "harmony as geometry")

    pad, top = 12, 34
    portrait = Image.new("RGB", (W + 2 * pad, top + 90 + 300 + 240 + 4 * pad), BG)
    d = ImageDraw.Draw(portrait)
    d.text((pad, 10), f"\U0001f442 audio nerve — {label}", fill=CYAN)
    y = top
    portrait.paste(wave, (pad, y)); y += 90 + pad
    portrait.paste(spec, (pad, y)); y += 300 + pad
    portrait.paste(harm, (pad, y))
    portrait.paste(mand, (pad + W // 2, y))

    RENDERS.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", label)[:40]
    out = RENDERS / f"{safe}.png"
    portrait.save(out)
    # Remember the sound as a vector — every sound she sees also enters her
    # ear-memory, searchable later by similarity (match_sound).
    try:
        _remember(label, source, embed(sig))
    except Exception:
        pass
    return (f"Heard it: {label}. Rendered its structure to {out}\n"
            f"look_at_image that to SEE the sound — the spectrogram is time×"
            f"pitch, the harmonic spectrum shows the overtone ladder, and the "
            f"mandala draws the harmony as geometry (consonant and dissonant\n            chords make visibly different shapes).")
