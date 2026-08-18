"""
Eyes and ears — vision, speech-in, speech-out.

The point of this module is to get non-text information *into* the agent.
Everything a language model knows normally arrives as text, which means
anything that started as a picture or a sound has to be described by a human
first. That human is the bottleneck: they see a UI bug, translate it into
words, and the agent works from the translation instead of the thing. These
helpers remove that step.

Nothing here is required. Each capability reports whether it's available and
says plainly what's missing if it isn't, so Forge runs fine on a machine with
no vision model, no microphone and no speakers.

  see()        an image -> a description, via a local vision model
  screenshot() the screen -> a PNG on disk
  listen()     audio (file or microphone) -> text, via whisper.cpp
  speak()      text -> spoken aloud
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx


# ---------------------------------------------------------------- config

@dataclass
class MediaConfig:
    """Where the eyes and ears live. Overridable from config.yaml."""
    # Primary vision is Merge's OWN native eyes (the 27B on 8085, started with
    # its mmproj) — far better than the old 7B, and she sees the real pixels
    # instead of reading a smaller model's summary. When she isn't the running
    # model (e.g. the fast 30B is up), 8085 is simply down and vision falls
    # through to the shared CPU 7B backstop below, which is always serving.
    vision_url: str = "http://127.0.0.1:8085/v1"
    vision_model: str = "qwen3.6-27b"
    vision_fallback_url: str = "http://127.0.0.1:8090/v1"
    vision_fallback_model: str = "qwen2.5-vl"
    whisper_bin: str = str(Path.home() / "whisper.cpp/build/bin/whisper-cli")
    whisper_model: str = str(Path.home() / "whisper.cpp/models/ggml-small.en.bin")
    tts_command: str = "spd-say"          # speech-dispatcher; ships with most desktops
    # A Piper voice model makes her sound like a person instead of a robot.
    # When this file exists (and piper is installed), it wins over
    # tts_command; delete or rename it to fall back to the robot.
    piper_voice: str = str(Path.home() / "forge/models/voices/en_US-amy-medium.onnx")


def load_media_config(cfg: dict) -> MediaConfig:
    m = (cfg or {}).get("media", {}) or {}
    clean = {k: v for k, v in m.items() if k in MediaConfig.__dataclass_fields__}
    # config files carry ~ paths; nothing downstream expands them
    for key in ("whisper_bin", "whisper_model"):
        if isinstance(clean.get(key), str):
            clean[key] = str(Path(clean[key]).expanduser())
    return MediaConfig(**clean)


# ---------------------------------------------------------------- vision

def _endpoints(mc: MediaConfig) -> list[tuple[str, str]]:
    """Vision endpoints in priority order: Merge's own eyes, then the 7B."""
    eps = [(mc.vision_url, mc.vision_model)]
    if getattr(mc, "vision_fallback_url", ""):
        eps.append((mc.vision_fallback_url, mc.vision_fallback_model))
    return eps


def vision_available(mc: MediaConfig) -> tuple[bool, str]:
    for url, _ in _endpoints(mc):
        try:
            r = httpx.get(f"{url.rstrip('/')}/models", timeout=3.0)
            if r.status_code == 200:
                which = "Merge's own eyes" if ":8085" in url else "the 7B backstop"
                return True, f"vision serving via {which} ({url})"
        except Exception:
            continue
    return False, ("no vision server reachable (tried Merge's eyes on 8085 and "
                   "the 7B on 8090) — start one with: ~/forge/start-model.sh vision")


def see(image_path: str, question: str, mc: MediaConfig,
        max_tokens: int = 700) -> str:
    """
    Ask a local vision model about an image.

    Images ride as base64 data URLs, which is what llama.cpp's server expects
    when it's started with an mmproj (the vision half of the model).
    """
    p = Path(image_path).expanduser()
    if not p.exists():
        return f"Error: no such image: {p}"
    if p.stat().st_size > 20_000_000:
        return f"Error: image is very large ({p.stat().st_size} bytes)."

    # Downscale before sending. A 3840x1080 desktop grab costs a lot of
    # encoder memory and tokens for detail no vision model resolves anyway;
    # 1600px on the long edge keeps UI text legible while staying cheap.
    p = _shrink(p, 1600)

    suffix = p.suffix.lower().lstrip(".") or "png"
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png",
            "webp": "webp", "gif": "gif"}.get(suffix, "png")
    b64 = base64.b64encode(p.read_bytes()).decode()

    def ask(url: str, model: str) -> str:
        body = {
            "model": model,
            "max_tokens": max_tokens,
            # Vision is a describe task, not a reasoning one. Turning thinking
            # off (on Merge's reasoning model) skips the 512-token reasoning
            # phase — faster, and it stops the budget from eating short
            # descriptions. Harmless on the 7B, which ignores the kwarg.
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
                ],
            }],
        }
        # CPU vision (the 7B) is minutes, not seconds; a short timeout just
        # turns "slow" into "broken". Merge's own GPU eyes answer in seconds.
        r = httpx.post(f"{url.rstrip('/')}/chat/completions",
                       json=body, timeout=900.0)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    # Try Merge's own eyes first, fall back to the 7B if she isn't serving.
    last = None
    for url, model in _endpoints(mc):
        try:
            return ask(url, model)
        except Exception as e:              # unreachable, timeout, HTTP error
            last = e
            continue
    return (f"Error: no vision endpoint answered (tried Merge's eyes on 8085 and "
            f"the 7B on 8090). Last error: {type(last).__name__}: {last}. "
            f"Start one with: ~/forge/start-model.sh vision")


# ------------------------------------------------------------ screenshot

def screenshot(out_path: str | None = None, region: str | None = None) -> str:
    """
    Capture the screen to a PNG and return its path.

    Tries several tools because desktops differ; whichever exists wins.
    `region` may be "x,y,w,h" to grab part of the screen.
    """
    out = Path(out_path).expanduser() if out_path else Path(
        tempfile.gettempdir()) / "forge-screenshot.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    # Pillow first: it's pure Python, needs no system package and no sudo,
    # and it grabs the whole desktop (all monitors) on X11. Only fall back to
    # external tools when it can't — mainly Wayland, which blocks direct
    # framebuffer access.
    try:
        from PIL import ImageGrab
        im = ImageGrab.grab()
        im.save(out)
        if out.exists() and out.stat().st_size > 0:
            if region:
                _crop(out, region)
            return str(out)
    except Exception:
        pass

    wayland = (os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland")
    candidates: list[list[str]] = []
    if wayland and shutil.which("grim"):
        candidates.append(["grim", str(out)])
    if shutil.which("import"):              # ImageMagick
        candidates.append(["import", "-window", "root", str(out)])
    if shutil.which("scrot"):
        candidates.append(["scrot", "-o", str(out)])
    if shutil.which("gnome-screenshot"):
        candidates.append(["gnome-screenshot", "-f", str(out)])
    if shutil.which("spectacle"):
        candidates.append(["spectacle", "-b", "-n", "-o", str(out)])
    if shutil.which("xwd") and shutil.which("convert"):
        candidates.append(["sh", "-c", f"xwd -root | convert xwd:- png:{out}"])

    if not candidates:
        return ("Error: no screenshot tool found. Install one:\n"
                "  sudo apt install imagemagick   (gives 'import')\n"
                "  sudo apt install scrot")

    errors = []
    for cmd in candidates:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if r.returncode == 0 and out.exists() and out.stat().st_size > 0:
                if region:
                    _crop(out, region)
                return str(out)
            errors.append(f"{cmd[0]}: {(r.stderr or 'failed').strip()[:80]}")
        except Exception as e:
            errors.append(f"{cmd[0]}: {type(e).__name__}")
    return "Error: every screenshot tool failed —\n  " + "\n  ".join(errors)


def _shrink(path: Path, max_edge: int) -> Path:
    """Return a path to a copy no larger than max_edge on its long side."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            if max(im.size) <= max_edge:
                return path
            scale = max_edge / max(im.size)
            small = im.resize((int(im.width * scale), int(im.height * scale)),
                              Image.LANCZOS)
            out = Path(tempfile.gettempdir()) / f"forge-shrunk-{path.stem}.png"
            small.convert("RGB").save(out)
            return out
    except Exception:
        return path


def _crop(path: Path, region: str) -> None:
    """Crop in place to 'x,y,w,h'. Silently skipped if Pillow isn't around."""
    try:
        from PIL import Image
        x, y, w, h = (int(v) for v in region.split(","))
        with Image.open(path) as im:
            im.crop((x, y, x + w, y + h)).save(path)
    except Exception:
        pass


# ---------------------------------------------------------------- speech

def speech_available(mc: MediaConfig) -> tuple[bool, str]:
    if not Path(mc.whisper_bin).exists():
        return False, f"whisper not built at {mc.whisper_bin}"
    if not Path(mc.whisper_model).exists():
        return False, f"no speech model at {mc.whisper_model}"
    return True, "speech-to-text ready"


def listen(audio_path: str, mc: MediaConfig) -> str:
    """Transcribe an audio file to text with whisper.cpp."""
    ok, why = speech_available(mc)
    if not ok:
        return f"Error: {why}"
    src = Path(audio_path).expanduser()
    if not src.exists():
        return f"Error: no such audio file: {src}"

    # whisper.cpp wants 16kHz mono WAV; convert anything else if ffmpeg is here
    wav = src
    tmp = None
    if src.suffix.lower() != ".wav":
        if not shutil.which("ffmpeg"):
            return (f"Error: {src.suffix} needs converting to WAV first, and "
                    f"ffmpeg isn't installed (sudo apt install ffmpeg).")
        tmp = Path(tempfile.mkdtemp()) / "audio.wav"
        subprocess.run(["ffmpeg", "-y", "-i", str(src), "-ar", "16000",
                        "-ac", "1", str(tmp)], capture_output=True, timeout=120)
        wav = tmp

    try:
        r = subprocess.run(
            [mc.whisper_bin, "-m", mc.whisper_model, "-f", str(wav),
             "--output-txt", "--no-timestamps", "-of", str(wav)],
            capture_output=True, text=True, timeout=600,
        )
        txt = Path(str(wav) + ".txt")
        if txt.exists():
            return txt.read_text(encoding="utf-8").strip()
        return (r.stdout or r.stderr or "").strip() or "(nothing transcribed)"
    except subprocess.TimeoutExpired:
        return "Error: transcription timed out"
    except Exception as e:
        return f"Error transcribing: {type(e).__name__}: {e}"
    finally:
        if tmp:
            shutil.rmtree(tmp.parent, ignore_errors=True)


def _piper_bin() -> str | None:
    """Piper installed next to our own interpreter, or on PATH."""
    import sys
    cand = Path(sys.executable).parent / "piper"
    if cand.exists():
        return str(cand)
    return shutil.which("piper")


def _render_wav(text: str, voice_path: Path, cap: int = 2000) -> Path | None:
    """The single place that shells out to piper: render text to a temp WAV and
    return its path (or None if piper/the voice isn't there). synth_wav (bytes
    for the browser) and speak (play locally) both go through here."""
    piper = _piper_bin()
    if not piper or not voice_path or not voice_path.exists():
        return None
    try:
        wav = Path(tempfile.mkdtemp()) / "say.wav"
        r = subprocess.run([piper, "--model", str(voice_path), "--output_file", str(wav)],
                           input=text[:cap], text=True, capture_output=True, timeout=90)
        if r.returncode == 0 and wav.exists():
            return wav
    except Exception:
        pass
    return None


VOICES_DIR = Path.home() / "forge/models/voices"

# Friendly names for the voices we ship. Anything not listed still shows up,
# labelled by its file stem, so dropping a new .onnx in the folder just works.
_VOICE_LABELS = {
    "en_US-amy-medium": "Amy — US, warm",
    "en_US-lessac-medium": "Lessac — US, clear",
    "en_US-ryan-medium": "Ryan — US, male",
    "en_GB-alan-medium": "Alan — UK, male",
    "en_GB-jenny_dioco-medium": "Jenny — UK, female",
}


def list_voices() -> list[dict]:
    """Every piper voice sitting in the voices folder, prettied up for a menu."""
    out = []
    if VOICES_DIR.exists():
        for f in sorted(VOICES_DIR.glob("*.onnx")):
            out.append({"id": f.stem, "label": _VOICE_LABELS.get(f.stem, f.stem)})
    return out


def synth_wav(text: str, voice_id: str = "") -> bytes | None:
    """Render text to WAV bytes with a named piper voice. Returns None if piper
    or the voice isn't available. voice_id is validated against the folder so a
    caller can't reach outside it."""
    valid = {v["id"] for v in list_voices()}
    if voice_id not in valid:
        voice_id = "en_US-amy-medium" if "en_US-amy-medium" in valid else next(iter(valid), "")
    if not voice_id:
        return None
    wav = _render_wav(text, VOICES_DIR / f"{voice_id}.onnx", cap=2000)
    return wav.read_bytes() if wav else None


def speak(text: str, mc: MediaConfig) -> str:
    """Say something out loud. Best-effort — silence is not an error worth failing on."""
    # Human voice first: piper renders to a wav, a system player plays it.
    voice = Path(mc.piper_voice).expanduser() if mc.piper_voice else None
    player = shutil.which("paplay") or shutil.which("aplay") or shutil.which("ffplay")
    if voice and player:
        wav = _render_wav(text, voice, cap=800)
        if wav:
            try:
                cmd = [player, str(wav)]
                if player.endswith("ffplay"):
                    cmd = [player, "-nodisp", "-autoexit", "-loglevel", "quiet", str(wav)]
                subprocess.run(cmd, capture_output=True, timeout=60)
                return "spoken"
            except Exception:
                pass   # fall through to the robot rather than stay silent
    if not shutil.which(mc.tts_command.split()[0]):
        return f"(no TTS: {mc.tts_command} not installed)"
    try:
        subprocess.run(mc.tts_command.split() + [text[:800]],
                       capture_output=True, timeout=60)
        return "spoken"
    except Exception as e:
        return f"(TTS failed: {type(e).__name__})"


def capabilities(mc: MediaConfig) -> dict:
    """What's actually usable right now — for the dashboard and /media."""
    v_ok, v_why = vision_available(mc)
    s_ok, s_why = speech_available(mc)
    # Don't infer "works" from "installed" — spectacle is present on this
    # kind of desktop and fails when actually invoked, which made the status
    # panel lie. Pillow is the one we can genuinely vouch for by import.
    try:
        from PIL import ImageGrab  # noqa: F401
        shot, shot_why = True, "ready (Pillow)"
    except Exception:
        shot = bool(shutil.which("import") or shutil.which("scrot")
                    or shutil.which("gnome-screenshot") or shutil.which("grim"))
        shot_why = "ready (external tool)" if shot else "no screenshot method available"
    return {
        "vision": {"ok": v_ok, "detail": v_why},
        "speech_in": {"ok": s_ok, "detail": s_why},
        "screenshot": {"ok": shot, "detail": shot_why},
        "speech_out": {"ok": bool(shutil.which(mc.tts_command.split()[0])),
                       "detail": mc.tts_command},
    }
