"""
The agent's hands: read, write, edit, list, search, run.

Two rules shape everything here:

  1. Every path is resolved and checked against the workspace root before
     anything is read or written. An agent that can be talked into writing
     to /etc or reading ~/.ssh is a security hole, not a feature.
  2. Anything that changes the world (write, edit, run) declares itself
     as needing permission. The agent loop asks the user; tools never
     decide on their own that something is safe enough to skip the prompt.

Tools return plain strings — that's what goes back to the model. Errors are
returned as readable text rather than raised, so the model can read what
went wrong and correct itself instead of the run dying.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx

from . import (audio_nerve, bioacoustics, browser, business, cad, cortex, datasets, doolittle, persona, toolindex, vault, medical, xfiles, frameworks, identity, law, markets,
               market_regime, crossmap, news, paper_market, scanner, sims, walkforward)
from .codetools import syntax_check
from .config import load_config
from .dataops import data_ops, date_calc
from .mathtools import COMPUTE_DESCRIPTION, run_compute
from .physics import quantum_sim, relativity_sim
from .recall import search as recall_search
from .web import fetch_url, web_search
from .writing import ai_tells, name_check, text_stats

MAX_READ_BYTES = 400_000     # a huge file would blow the context window
MAX_OUTPUT_CHARS = 30_000    # same, for command output
DEFAULT_TIMEOUT = 120


# ---- headless-browser tool helpers (see browser.py) --------------------
def _fmt(val) -> str:
    if val is None:
        return "(no result)"
    if isinstance(val, str):
        return val[:3000]
    try:
        return json.dumps(val, default=str)[:3000]
    except Exception:
        return str(val)[:3000]


def _browse(url: str) -> str:
    r = browser.goto(url)
    out = [f"Loaded: {r['title']}  [{r['url']}]"]
    if r.get("console_errors"):
        out.append("\n⚠ Console/page errors (this is where bugs hide):")
        out += [f"  - {e}" for e in r["console_errors"]]
    else:
        out.append("(no console errors)")
    txt = (r.get("text") or "").strip()
    if txt:
        out.append("\nVisible text:\n" + txt)
    out.append("\n[Browser still open — browser_view to SEE it, browser_js to "
               "interact, browser_console for more logs.]")
    return "\n".join(out)


def _browser_view(full: bool = False) -> str:
    path = browser.screenshot(full)
    return (f"Screenshot saved: {path}\n"
            f"[Now call look_at_image on that path to see the page with your own eyes.]")


# ---- self-compressing notebook (save_note) -----------------------------
# The project notebook is kept to a FIXED size: it changes, it doesn't grow.
# When a new lesson pushes it over the cap, a little-model "librarian" folds it
# back down — merging overlaps, keeping the most useful — and the full pre-fold
# set is archived first, so a lesson is never silently lost.
NOTE_CAP = 20

_NB_HEADER = ("# Project notebook\n\nLessons this project has taught its agents, "
              "loaded at the start of every session. Kept to a fixed size — it "
              "sharpens, it doesn't sprawl.\n\n")


def _parse_notes(text: str) -> list[str]:
    return [ln[2:].strip() for ln in (text or "").splitlines()
            if ln.startswith("- ") and ln[2:].strip()]


def _notebook_md(notes: list[str]) -> str:
    return _NB_HEADER + "".join(f"- {n}\n" for n in notes)


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _librarian_compact(notes: list[str], cap: int) -> list[str] | None:
    """Ask the little model to fold the notebook down to `cap` lessons — merging
    overlaps, dropping only true redundancy, keeping every distinct useful one.
    Returns the new list, or None if it can't (caller keeps a safe fallback)."""
    m = (load_config().get("models") or {}).get("little") or {}
    base = (m.get("base_url") or "").rstrip("/")
    if not base:
        return None
    numbered = "\n".join(f"{i + 1}. {n}" for i, n in enumerate(notes))
    prompt = (
        f"You keep a work-notebook of at most {cap} lessons about how to work with "
        f"this user and project. It is one over the limit. Fold it back to {cap} or "
        f"fewer: MERGE lessons that overlap, and drop ONLY what is truly redundant "
        f"— never lose a distinct, useful lesson, and keep the original wording where "
        f"you can. One short line each.\n\nNotebook:\n{numbered}\n\n"
        f"Reply with ONLY the final lessons, one per line, each starting with '- ', "
        f"no numbers and no commentary.")
    try:
        r = httpx.post(f"{base}/chat/completions",
                       json={"messages": [{"role": "user", "content": prompt}],
                             "max_tokens": 1500, "temperature": 0.2}, timeout=90)
        if r.status_code != 200:
            return None
        out = _parse_notes(r.json()["choices"][0]["message"].get("content") or "")
        # Guardrails: sane count, and it didn't nuke the content wholesale.
        if out and len(out) <= cap and len("".join(out)) >= 0.4 * len("".join(notes)):
            return out
    except Exception:
        pass
    return None


def _compact_notebook_file(nb_path: str, arch_path: str, cap: int) -> None:
    """Background: archive the full notebook, then fold it back to the cap."""
    nb = Path(nb_path)
    try:
        notes = _parse_notes(nb.read_text(encoding="utf-8"))
    except Exception:
        return
    if len(notes) <= cap:
        return
    try:      # archive the full set BEFORE folding — nothing is ever lost
        with Path(arch_path).open("a", encoding="utf-8") as f:
            f.write("\n## snapshot before folding\n" + "".join(f"- {n}\n" for n in notes))
    except Exception:
        pass
    folded = _librarian_compact(notes, cap) or notes[-cap:]   # fallback: keep freshest
    try:
        _write_atomic(nb, _notebook_md(folded))
    except Exception:
        pass

# Commands refused even in auto mode. Deliberately a short list of
# machine-killers, not a nanny filter — deleting a project folder is the
# user's right; wiping the disk or the home directory is not a judgment
# call any model should get to make.
import re as _re
_MACHINE_KILLERS: list[tuple[str, _re.Pattern]] = [
    ("deletes / or your home directory",
     _re.compile(r"\brm\s+(?:-[a-zA-Z]+\s+)*(?:/|~|\$HOME)\*?(?:\s|$)")),
    ("formats a disk", _re.compile(r"\bmkfs(\.|\s|$)")),
    ("writes raw bytes over a disk device", _re.compile(r"\bdd\b.*\bof=/dev/")),
    ("overwrites a disk device", _re.compile(r">\s*/dev/(sd|nvme|mmcblk)")),
    ("fork bomb", _re.compile(r":\s*\(\s*\)\s*\{")),
]


def _machine_killer(command: str) -> str | None:
    for why, pat in _MACHINE_KILLERS:
        if pat.search(command):
            return why
    return None


# Paths a fenced command may still reference: system binaries and libraries,
# nothing personal. Everything else outside the workspace is refused.
_FENCE_ALLOWED = ("/usr/", "/bin/", "/sbin/", "/lib/", "/lib64/", "/opt/",
                  "/dev/null", "/proc/self")


def _fence_violation(command: str, root: Path) -> str | None:
    """In kid mode, find the first thing in a command that reaches outside
    the workspace: home shortcuts, parent-hopping, absolute paths that
    aren't system locations. Deterministic and strict — a false refusal
    costs a reworded command; a false allow costs someone's files."""
    rootstr = str(root)
    for tok in _re.findall(r"[^\s;|&<>()'\"=]+", command):
        if tok.startswith("~") or "$HOME" in tok or "${HOME" in tok:
            return tok
        if ".." in tok.split("/"):
            return tok
        if tok.startswith("/"):
            if tok == rootstr or tok.startswith(rootstr + "/"):
                continue
            if not tok.startswith(_FENCE_ALLOWED):
                return tok
    return None


def _lenient_replace(text: str, old: str, new: str) -> str | None:
    """
    Whitespace-forgiving fallback for edit_file.

    Models — small ones constantly — reproduce the lines they want to change
    with the wrong indentation, then miss, re-read, and miss again forever.
    If the old text matches exactly one place in the file when comparing
    line-by-line with whitespace stripped, that's an unambiguous edit: take
    it, and shift the replacement to the file's real indentation.

    Returns the new file text, or None when there's no single clear match
    (zero or several) — ambiguity still refuses, same as the strict path.
    """
    hay = text.splitlines(keepends=True)
    old_lines = old.splitlines()
    while old_lines and not old_lines[0].strip():
        old_lines.pop(0)
    while old_lines and not old_lines[-1].strip():
        old_lines.pop()
    if not old_lines:
        return None
    needle = [l.strip() for l in old_lines]
    n = len(needle)
    matches = [i for i in range(len(hay) - n + 1)
               if [hay[i + j].strip() for j in range(n)] == needle]
    if len(matches) != 1:
        return None
    i = matches[0]

    file_indent = len(hay[i]) - len(hay[i].lstrip())
    given_indent = len(old_lines[0]) - len(old_lines[0].lstrip())
    delta = file_indent - given_indent
    out = []
    for l in new.splitlines():
        if not l.strip():
            out.append("")
        elif delta >= 0:
            out.append(" " * delta + l)
        else:
            cur = len(l) - len(l.lstrip())
            out.append(l[min(-delta, cur):])
    matched = "".join(hay[i:i + n])
    trail = "\n" if matched.endswith("\n") else ""
    return "".join(hay[:i]) + "\n".join(out) + trail + "".join(hay[i + n:])


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict          # JSON Schema
    run: Callable[..., str]
    needs_permission: bool = False
    # A one-line human summary shown in the permission prompt, so the user
    # sees "write to config.py" rather than a wall of JSON.
    summarize: Callable[[dict], str] | None = None
    # Ask even in auto mode. For actions whose effect lands somewhere the
    # user might not be — sound comes out of the server's speakers, and the
    # user may be on a tablet in another room.
    always_ask: bool = False


class Workspace:
    """Everything the agent is allowed to touch, and nothing else."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        # Paths agents may never WRITE, one glob per line in .forge-protect
        # at the workspace root (# for comments). Mechanical on purpose:
        # 2026-08-11 proved a written rule loses to a direct "fix it" from
        # the user — the model edited an author's read-only manuscript
        # minutes after reading a notebook rule saying never to. Reading
        # stays allowed; only writes are refused. (Known hole: run_command
        # can still shell its way in — the file tools are what models
        # actually use, and a fence there covers the real behavior.)
        self.protected: list[str] = []
        pf = self.root / ".forge-protect"
        if pf.is_file():
            try:
                self.protected = [ln.strip() for ln in
                                  pf.read_text(encoding="utf-8").splitlines()
                                  if ln.strip() and not ln.strip().startswith("#")]
            except OSError:
                pass
        # Files the model has actually seen this session. edit_file refuses
        # to touch a file that isn't in here — "read before you write" as a
        # hard rule instead of a polite request in the prompt.
        self.reads: set[Path] = set()
        # mtime at the moment each file in `reads` was last seen. The author
        # editing a world file by hand mid-session is a feature here, not an
        # edge case — this is how the agent notices its cached read is stale
        # (see Agent._stale_files, which compares this against disk).
        self.read_mtimes: dict[Path, float] = {}
        # Consecutive edit_file misses per file. Small models get stuck
        # re-guessing the exact text forever; after a couple of misses the
        # error starts telling them to rewrite the file instead.
        self.edit_misses: dict[Path, int] = {}

    def resolve(self, path: str) -> Path:
        """Resolve a user/model-supplied path, refusing anything outside root."""
        # Expand ~ FIRST. Without this, "~/x" is not absolute, so it gets
        # joined onto the workspace root and silently creates a directory
        # literally named "~" -- writes succeed, later reads look in the real
        # home and find nothing, and the agent spins forever. Expanding here
        # means ~ means what it says, and the fence below refuses it loudly
        # if it lands outside the workspace.
        p = Path(path).expanduser()
        full = (self.root / p).resolve() if not p.is_absolute() else p.resolve()
        if full != self.root and self.root not in full.parents:
            raise PermissionError(
                f"Path is outside the workspace: {full}\n"
                f"The agent may only touch files under {self.root}"
            )
        return full

    def rel(self, p: Path) -> str:
        try:
            return str(p.relative_to(self.root))
        except ValueError:
            return str(p)

    def protection(self, f: Path) -> str | None:
        """The glob that write-protects f, or None. fnmatch's * crosses
        slashes, so 'source-material/*' covers nested files too."""
        import fnmatch
        rel = self.rel(f)
        for g in self.protected:
            if fnmatch.fnmatch(rel, g):
                return g
        return None

    def mark_read(self, f: Path) -> None:
        """Record that the model has seen f's CURRENT content — via read,
        write, or edit. Stamps the mtime so a later external edit (the
        author hand-writing into a world file) is detectable as staleness."""
        self.reads.add(f)
        try:
            self.read_mtimes[f] = f.stat().st_mtime
        except OSError:
            pass


def build_media_tools(ws: Workspace, mc) -> list[Tool]:
    """
    Eyes and ears, offered only when they actually work.

    A tool the model can see but that always fails is worse than no tool —
    it burns turns and teaches the model bad habits. So each of these is
    attached only if the underlying capability reports ready.
    """
    from . import media

    caps = media.capabilities(mc)
    tools: list[Tool] = []

    if caps["vision"]["ok"]:
        def look_at(image: str, question: str = "Describe this image in detail.") -> str:
            path = image
            try:
                path = str(ws.resolve(image))
            except PermissionError:
                # An absolute path outside the workspace is fine for *reading*
                # an image the user pointed at (a screenshot in /tmp, say) —
                # this tool never writes, so the sandbox isn't at risk.
                pass
            return media.see(path, question, mc)

        tools.append(Tool(
            name="look_at_image",
            description="Look at an image file and answer a question about it. "
                        "Use for screenshots, mockups, diagrams, photos of "
                        "whiteboards, or anything you need to SEE rather than read.",
            parameters={
                "type": "object",
                "properties": {
                    "image": {"type": "string", "description": "Path to the image"},
                    "question": {"type": "string",
                                 "description": "What you want to know about it"},
                },
                "required": ["image"],
            },
            run=look_at,
        ))

    if caps["screenshot"]["ok"]:
        def grab(region: str = "") -> str:
            path = media.screenshot(region=region or None)
            if path.startswith("Error"):
                return path
            if caps["vision"]["ok"]:
                return f"Screenshot saved to {path} — use look_at_image to see it."
            return f"Screenshot saved to {path} (no vision model running to read it)."

        tools.append(Tool(
            name="take_screenshot",
            description="Capture the current screen to a PNG file. Pair with "
                        "look_at_image to actually see what's on screen.",
            parameters={
                "type": "object",
                "properties": {
                    "region": {"type": "string",
                               "description": "Optional 'x,y,w,h' to grab part of the screen"},
                },
            },
            run=grab,
            needs_permission=True,
            summarize=lambda a: "take a screenshot of your screen",
        ))

    if caps["speech_in"]["ok"]:
        tools.append(Tool(
            name="transcribe_audio",
            description="Turn a recording of speech into text. Use when the user "
                        "points at a voice memo or any audio file.",
            parameters={
                "type": "object",
                "properties": {"audio": {"type": "string", "description": "Path to the audio file"}},
                "required": ["audio"],
            },
            run=lambda audio: media.listen(audio, mc),
        ))

    if caps["speech_out"]["ok"]:
        tools.append(Tool(
            name="say_aloud",
            description="Speak a short message out loud — through the speakers "
                        "of the machine Forge runs on, which may not be where "
                        "the user is sitting. Only use it when the user "
                        "explicitly asked to be told out loud.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            run=lambda text: media.speak(text, mc),
            needs_permission=True,
            always_ask=True,
            summarize=lambda a: f"say out loud: {str(a.get('text',''))[:60]}",
        ))

    return tools


# Extensions we can parse-check after a write, so broken code can't be saved
# silently. Maps to the languages syntax_check knows.
_CHECK_EXT = {
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".py": "python", ".json": "json", ".sh": "bash", ".bash": "bash",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
    ".pl": "perl", ".yaml": "yaml", ".yml": "yaml",
}


_JSTOOLS = Path.home() / "forge" / "jstools"
_ESLINT_BIN = _JSTOOLS / "node_modules" / ".bin" / "eslint"
_ESLINT_CFG = _JSTOOLS / "eslint.config.mjs"


def _eslint(f: Path) -> str:
    """Lint a JS file for real bugs beyond syntax (==, undefined names, const
    reassignment, ...). Advisory — returns a compact findings suffix or ''.
    Must run from the file's own dir (eslint v10 anchors its base path there)."""
    if not _ESLINT_BIN.exists() or f.suffix.lower() not in (".js", ".mjs", ".cjs"):
        return ""
    try:
        r = subprocess.run(
            [str(_ESLINT_BIN), "--config", str(_ESLINT_CFG), "--format", "json", str(f)],
            capture_output=True, text=True, timeout=30, cwd=str(f.parent))
        data = json.loads(r.stdout or "[]")
    except Exception:
        return ""
    msgs = [f"line {m.get('line')}: {m.get('message')} [{m.get('ruleId')}]"
            for entry in data for m in entry.get("messages", []) if m.get("ruleId")]
    if not msgs:
        return ""
    shown = "; ".join(msgs[:4])
    more = f" (+{len(msgs) - 4} more)" if len(msgs) > 4 else ""
    return (f"\n· lint flagged {len(msgs)} thing(s) — worth a look: {shown}{more}")


def _autocheck_html(f: Path) -> str:
    """HTML files carry inline <script> that the plain extension check misses —
    exactly where a browser-game bug like an undefined function hides. Pull the
    inline JS out, parse-check and lint it, so 'checkCollisions is not defined'
    shows up at SAVE time instead of only in the browser console."""
    try:
        html = f.read_text(encoding="utf-8")
    except Exception:
        return ""
    scripts = re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", html,
                         re.S | re.I)
    js = "\n;\n".join(scripts).strip()
    if not js:
        return "  ✓ saved (no inline script to check)"
    # write the extracted JS to a sibling temp file so eslint/node see real paths
    tmp = f.parent / f".{f.stem}.inline.js"
    try:
        tmp.write_text(js, encoding="utf-8")
        parse = syntax_check("javascript", js)
        if not parse.startswith("SYNTAX OK"):
            first = " ".join(parse.split("\n")[:3])[:220]
            return (f"\n⚠ HEADS UP — the inline <script> DOES NOT PARSE: {first}\n"
                    f"Fix it before reporting done.")
        lint = _eslint(tmp)
        return "  ✓ inline script parses clean" + lint
    finally:
        tmp.unlink(missing_ok=True)


def _autocheck(f: Path) -> str:
    """After a write/edit, verify the file still parses (and for JS, lint it).
    Returns a short status suffix appended to the tool result — '' for files we
    can't check. The dummy-proofing: 'Created app.js' now also says whether
    app.js is valid, so she can't leave broken code on disk unaware."""
    if f.suffix.lower() in (".html", ".htm"):
        return _autocheck_html(f)
    lang = _CHECK_EXT.get(f.suffix.lower())
    if not lang:
        return ""
    try:
        result = syntax_check(lang, f.read_text(encoding="utf-8"))
    except Exception:
        return ""
    if result.startswith("SYNTAX OK"):
        suffix = "  ✓ parses clean"
        if lang == "javascript":
            suffix += _eslint(f)      # add bug-lint on top of the parse check
        return suffix
    first = " ".join(result.split("\n")[:3])[:220]
    return (f"\n⚠ HEADS UP — this file DOES NOT PARSE: {first}\n"
            f"Fix it now before moving on; don't report it as done while broken.")


_PRETTIER_BIN = _JSTOOLS / "node_modules" / ".bin" / "prettier"
_PRETTIER_EXT = {".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx", ".json",
                 ".css", ".scss", ".html", ".md", ".yaml", ".yml"}


def _format_code(ws, path: str) -> str:
    f = ws.resolve(path)
    if not f.exists():
        return f"Error: no such file: {ws.rel(f)}"
    if f.suffix.lower() not in _PRETTIER_EXT:
        return f"Error: prettier doesn't format {f.suffix or 'this'} files."
    if not _PRETTIER_BIN.exists():
        return "Error: prettier isn't installed (expected in ~/forge/jstools)."
    try:
        r = subprocess.run([str(_PRETTIER_BIN), "--write", str(f)],
                           capture_output=True, text=True, timeout=30,
                           cwd=str(f.parent))
    except Exception as e:
        return f"Error running prettier: {type(e).__name__}: {e}"
    if r.returncode != 0:
        return f"Error formatting: {((r.stderr or r.stdout) or '')[-300:]}"
    ws.mark_read(f)      # content changed — treat as freshly known so edits re-match
    return f"Formatted {ws.rel(f)} with prettier."


_PHYS_RELATIVITY = ("time_dilation", "twin_trip", "doppler", "photon_clock")
_PHYS_QUANTUM = ("photon", "double_slit", "uncertainty")


def _physics_sim(scenario: str, params: dict | None = None) -> str:
    """One on-demand dispatcher for the physics sims — keeps a single lean tool
    in the schema instead of two fat ones, while covering every scenario."""
    p = params or {}
    if scenario in _PHYS_RELATIVITY:
        keys = ("v_c", "duration_years", "distance_ly")
        return relativity_sim(scenario, **{k: p[k] for k in keys if k in p})
    if scenario in _PHYS_QUANTUM:
        keys = ("wavelength_nm", "slit_separation_um", "screen_distance_m",
                "photons", "delta_x_nm", "seed")
        return quantum_sim(scenario, **{k: p[k] for k in keys if k in p})
    return ("Error: unknown scenario. Relativity: time_dilation, twin_trip, "
            "doppler, photon_clock. Quantum: photon, double_slit, uncertainty.")


def _generate_image(ws_root: str, prompt: str, filename: str = "",
                    steps: int = 3, seed=None) -> str:
    """Run the CPU image generator as a subprocess (keeps the 2.5GB SD model
    out of the agent's own memory, and hides the GPU so Merge keeps it)."""
    root = Path(ws_root)
    if filename:
        name = filename if filename.lower().endswith((".png", ".jpg", ".jpeg")) \
            else filename + ".png"
    else:
        slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")[:40] or "image"
        name = f"generated/{slug}.png"
    out = (root / name).resolve()
    if root != out and root not in out.parents:
        return f"Error: image path is outside the workspace: {out}"
    # Fast path: the keep-warm server (start-model.sh imagegen) holds the model
    # in RAM, so this skips the ~30s reload — only the draw remains. Falls
    # through to a one-off subprocess if that server isn't running.
    try:
        import urllib.request
        payload = {"prompt": prompt, "out_path": str(out), "steps": int(steps)}
        if seed is not None:
            payload["seed"] = int(seed)
        req = urllib.request.Request(
            "http://127.0.0.1:8771/generate", json.dumps(payload).encode(),
            {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            res = json.load(r)
        if res.get("path"):
            return (f"Image saved to {res['path']} (warm server). "
                    f"Use look_at_image on that path to see what you made.")
    except Exception:
        pass   # warm server down or errored -> subprocess fallback below
    forge_root = str(Path(__file__).resolve().parent.parent)
    env = {**os.environ, "PYTHONPATH": forge_root, "CUDA_VISIBLE_DEVICES": ""}
    cmd = [sys.executable, "-m", "forge.imagegen", prompt, str(out), str(int(steps))]
    if seed is not None:
        cmd.append(str(int(seed)))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    except subprocess.TimeoutExpired:
        return "Error: image generation timed out (over 5 minutes)."
    if r.returncode != 0:
        return f"Error generating image: {((r.stderr or r.stdout) or '')[-600:]}"
    return (f"Image saved to {out}. It's a concept/mood sketch (fast CPU model). "
            f"Use look_at_image on that path to see what you made.")


_INDEX_REGISTRY: dict = {}


def build_tools(ws: Workspace, fenced: bool = False) -> list[Tool]:
    """Construct the toolset bound to one workspace.

    fenced=True (kid mode) additionally confines run_command to the
    workspace: the file tools were always fenced, but shell commands could
    reach anything. With the fence, a borrowed tablet can only make a mess
    inside the folder it was given."""

    def guard(fn):
        """Turn a workspace violation into readable text for the model.

        The sandbox itself raises — that's correct, it must be impossible to
        ignore. But a raise would abort the whole run, and the model would
        never learn why. Wrapping it here means the model reads 'that's
        outside the workspace' as a tool result and can adjust."""
        from functools import wraps

        @wraps(fn)
        def inner(*a, **kw):
            try:
                return fn(*a, **kw)
            except PermissionError as e:
                return f"Error: {e}"
        return inner

    def read_file(path: str, offset: int = 0, limit: int = 2000) -> str:
        f = ws.resolve(path)
        if not f.exists():
            return f"Error: no such file: {ws.rel(f)}"
        if f.is_dir():
            return f"Error: that's a directory, not a file: {ws.rel(f)}"
        if f.stat().st_size > MAX_READ_BYTES:
            return (f"Error: file is too large to read whole "
                    f"({f.stat().st_size} bytes). Use offset/limit to page through it.")
        try:
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception as e:
            return f"Error reading {ws.rel(f)}: {e}"
        ws.mark_read(f)
        window = lines[offset:offset + limit]
        if not window:
            return f"(no lines in range; file has {len(lines)} lines)"
        # Line numbers help the model target edits precisely. The BYTE cap
        # on the returned window exists because the line default (2000) is
        # sized for code: 1000 lines of a novel is ~25k tokens, which
        # detonates a 32k context in one call — seen live: two emergency
        # compactions and a dead turn. Cap loudly, never silently.
        body_lines, used = [], 0
        for i, ln in enumerate(window):
            entry = f"{i + offset + 1:6d}\t{ln}"
            if used + len(entry) > 20_000:
                next_off = offset + i
                body_lines.append(
                    f"... (output capped at 20KB to protect your working "
                    f"memory — {len(lines) - next_off} lines remain; "
                    f"continue with offset={next_off}, or search instead "
                    f"of reading big stretches)")
                break
            body_lines.append(entry)
            used += len(entry) + 1
        else:
            if offset + limit < len(lines):
                body_lines.append(f"... ({len(lines) - offset - limit} more "
                                  f"lines; use offset={offset + limit})")
        return "\n".join(body_lines)

    def _backup(f: Path) -> None:
        """Keep the previous version of a file about to change.

        One level deep, mirrored under .forge_backups/ — enough for 'undo
        that', without turning the workspace into a version-control system
        (that's what git is for).
        """
        if not f.exists() or not f.is_file():
            return
        bk = ws.root / ".forge_backups" / ws.rel(f)
        bk.parent.mkdir(parents=True, exist_ok=True)
        bk.write_bytes(f.read_bytes())

    def _refuse_protected(f: Path) -> str | None:
        g = ws.protection(f)
        if g:
            return (f"Error: {ws.rel(f)} is write-protected by .forge-protect "
                    f"({g}) — agents may read it, never change it. Tell the "
                    f"user exactly what to change and where (file and line); "
                    f"they make the change themselves.")
        return None

    def write_file(path: str, content: str) -> str:
        f = ws.resolve(path)
        if (refusal := _refuse_protected(f)):
            return refusal
        existed = f.exists()
        if existed and f not in ws.reads:
            return (f"Error: {ws.rel(f)} already exists and you haven't read it "
                    f"this session — overwriting it blind could destroy work. "
                    f"read_file it first, or pick a new filename.")
        f.parent.mkdir(parents=True, exist_ok=True)
        _backup(f)
        f.write_text(content, encoding="utf-8")
        # Writing the whole file counts as knowing its contents.
        ws.mark_read(f)
        verb = "Overwrote" if existed else "Created"
        return f"{verb} {ws.rel(f)} ({len(content.splitlines())} lines){_autocheck(f)}"

    def undo_file(path: str) -> str:
        f = ws.resolve(path)
        if (refusal := _refuse_protected(f)):
            return refusal
        bk = ws.root / ".forge_backups" / ws.rel(f)
        if not bk.exists():
            return (f"Error: no backup of {ws.rel(f)} — backups exist only for "
                    f"files write_file or edit_file changed this session or before.")
        current = f.read_bytes() if f.exists() else None
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(bk.read_bytes())
        # Swap, so undoing twice redoes — nothing is ever lost either way.
        if current is not None:
            bk.write_bytes(current)
        ws.mark_read(f)
        return f"Restored {ws.rel(f)} from backup (undo again to redo)."

    def edit_file(path: str, old: str, new: str, replace_all: bool = False) -> str:
        f = ws.resolve(path)
        if (refusal := _refuse_protected(f)):
            return refusal
        if not f.exists():
            return f"Error: no such file: {ws.rel(f)}"
        if f not in ws.reads:
            return (f"Error: you haven't read {ws.rel(f)} this session, so this "
                    f"edit was blocked. Use read_file on it first, then edit "
                    f"based on what's actually there.")
        text = f.read_text(encoding="utf-8")
        count = text.count(old)
        if count == 0:
            fixed = _lenient_replace(text, old, new)
            if fixed is not None:
                _backup(f)
                f.write_text(fixed, encoding="utf-8")
                ws.edit_misses.pop(f, None)
                ws.mark_read(f)
                return (f"Edited {ws.rel(f)} (1 replacement — your text's "
                        f"whitespace didn't match the file exactly, but it "
                        f"matched one place clearly, so the edit was applied "
                        f"at the file's real indentation){_autocheck(f)}")
            misses = ws.edit_misses.get(f, 0) + 1
            ws.edit_misses[f] = misses
            msg = (f"Error: that exact text isn't in {ws.rel(f)}. "
                   f"Read the file again — it may differ in whitespace or have changed.")
            if misses >= 2:
                msg += (f" You have now missed {misses} times on this file. "
                        f"Stop trying to patch it: read it once more, then use "
                        f"write_file to replace the WHOLE file with the "
                        f"corrected version in one go.")
            return msg
        if count > 1 and not replace_all:
            return (f"Error: found {count} matches in {ws.rel(f)}. "
                    f"Add more surrounding context to make it unique, or pass replace_all=true.")
        _backup(f)
        ws.edit_misses.pop(f, None)
        f.write_text(text.replace(old, new) if replace_all else text.replace(old, new, 1),
                     encoding="utf-8")
        ws.mark_read(f)
        return (f"Edited {ws.rel(f)} ({count if replace_all else 1} "
                f"replacement(s)){_autocheck(f)}")

    def list_dir(path: str = ".") -> str:
        d = ws.resolve(path)
        if not d.exists():
            return f"Error: no such directory: {ws.rel(d)}"
        if not d.is_dir():
            return f"Error: not a directory: {ws.rel(d)}"
        entries = []
        for item in sorted(d.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
            if item.name.startswith(".") and item.name not in (".env", ".gitignore"):
                continue
            entries.append(f"{item.name}/" if item.is_dir() else item.name)
        return "\n".join(entries) if entries else "(empty)"

    def _canon(s: str) -> str:
        """Case and quote-style must never decide whether text is 'found'.
        A model that searches "pont neuf" and gets nothing concludes the
        bridge isn't in the book — a false negative that seeds a false
        answer. Prose also uses curly quotes (GQ’s) where a model types
        straight ones (GQ's); both spell the same word."""
        return (s.replace("’", "'").replace("‘", "'")
                 .replace("“", '"').replace("”", '"').casefold())

    def search(pattern: str, path: str = ".", max_results: int = 60) -> str:
        """Case-insensitive, quote-forgiving content search. One code path
        on purpose — a previous ripgrep fast-path was case-sensitive while
        this fallback wasn't, so the same query found different truths
        depending on what was installed."""
        d = ws.resolve(path)
        # A silent "(no matches)" for a directory that doesn't even exist
        # taught the model that entire real books "contain no mention" of
        # their own protagonist — it searched invented folders (src/,
        # novels/) and read the empty result as fact about the text.
        if not d.exists():
            return (f"Error: no such directory: {ws.rel(d)} — use list_dir "
                    f"to see what actually exists before searching in it.")
        # "foot|tall|height" searches all three at once — one wide net
        # beats five guesses. The height that hid from three separate
        # sessions was always one synonym away from the query used.
        needles = [_canon(p) for p in pattern.split("|") if _canon(p.strip())]
        if not needles:
            return "(empty pattern)"
        hits, total = [], 0
        for root, dirs, files in os.walk(d):
            dirs[:] = [x for x in dirs if not x.startswith(".") and x != "node_modules"]
            for fn in files:
                fp = Path(root) / fn
                try:
                    for i, line in enumerate(fp.read_text(encoding="utf-8",
                                                          errors="ignore").splitlines(), 1):
                        if any(n in _canon(line) for n in needles):
                            total += 1
                            if len(hits) < max_results:
                                hits.append(f"{ws.rel(fp)}:{i}:{line.strip()[:200]}")
                except Exception:
                    continue
        # Silent truncation is how "the name is never revealed" happens: 60
        # early-book hits with no hint that the reveal sits at hit 80. Keep
        # counting past the cap and SAY what was left unshown.
        if total > len(hits):
            hits.append(f"... {total - len(hits)} MORE match(es) not shown "
                        f"(last shown was {hits[-1].split(':')[0]}:"
                        f"{hits[-1].split(':')[1]}) — the answer may be in the "
                        f"later ones. Use a more specific pattern, or search "
                        f"again reading from higher line numbers.")
        return "\n".join(hits) if hits else "(no matches)"

    def save_note(note: str) -> str:
        """Record one lesson in the project notebook.

        The notebook is a FIXED size (NOTE_CAP): it changes, it doesn't grow.
        A new lesson is added; if that pushes it over the cap, a little-model
        librarian folds it back down in the background (merging overlaps, keeping
        the most useful), and the full pre-fold set is archived first so nothing
        is ever lost. Pinned to one file, so no permission prompt needed.
        """
        note = " ".join(note.split())
        if not note:
            return "Error: empty note."
        nb = ws.root / "FORGE-NOTES.md"
        notes = _parse_notes(nb.read_text(encoding="utf-8")) if nb.exists() else []
        if note in notes:
            return "Already in the notebook — skipped the duplicate."
        notes.append(note)
        _write_atomic(nb, _notebook_md(notes))
        ws.mark_read(nb)
        if len(notes) > NOTE_CAP:
            arch = ws.root / "FORGE-NOTES-archive.md"
            threading.Thread(target=_compact_notebook_file,
                             args=(str(nb), str(arch), NOTE_CAP), daemon=True).start()
            return (f"Noted: {note[:80]} — notebook's at its {NOTE_CAP}-lesson cap, "
                    f"so I'm folding it back down in the background.")
        return f"Noted: {note[:80]}"

    def run_command(command: str, timeout: int = DEFAULT_TIMEOUT) -> str:
        blocked = _machine_killer(command)
        if blocked:
            return (f"Error: blocked — {blocked}. This command could damage the "
                    f"whole machine, so it's refused even in auto mode. If the "
                    f"user genuinely wants it, they can run it themselves.")
        if fenced:
            bad = _fence_violation(command, ws.root)
            if bad:
                return (f"Error: kid mode is on, and this command reaches "
                        f"outside the workspace ({bad!r}). Work only inside "
                        f"{ws.root} — rewrite the command with relative paths.")
        try:
            r = subprocess.run(
                command, shell=True, cwd=str(ws.root),
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {timeout}s: {command}"
        except Exception as e:
            return f"Error running command: {e}"
        out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr else "")
        out = out.strip() or "(no output)"
        if len(out) > MAX_OUTPUT_CHARS:
            out = out[:MAX_OUTPUT_CHARS] + f"\n... (truncated, {len(out)} chars total)"
        return f"[exit {r.returncode}]\n{out}"

    _built = [
        Tool(
            name="verify_phrase",
            description=(
                "Check the identity phrase the user just gave you against the "
                "owner's secret. Use this ONLY after you've asked for the phrase "
                "because a session felt off. Pass exactly what they said. It "
                "returns 'correct' or 'incorrect' — you never see or learn the "
                "real phrase, so you cannot reveal it. On 'incorrect', stay "
                "friendly but do NOT do anything sensitive; ask them to try the "
                "phrase again."
            ),
            parameters={
                "type": "object",
                "properties": {"attempt": {"type": "string",
                    "description": "Exactly what the user offered as the phrase"}},
                "required": ["attempt"],
            },
            run=lambda attempt: "correct" if identity.check_phrase(attempt) else "incorrect",
        ),
        Tool(
            name="read_file",
            description="Read a text file from the workspace. Returns numbered lines. "
                        "Use offset/limit to page through long files.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace root"},
                    "offset": {"type": "integer", "description": "First line to show (0-based)"},
                    "limit": {"type": "integer", "description": "How many lines to show"},
                },
                "required": ["path"],
            },
            run=guard(read_file),
        ),
        Tool(
            name="write_file",
            description="Create a file, or completely replace an existing one. "
                        "For small changes to an existing file prefer edit_file.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string", "description": "The complete file contents"},
                },
                "required": ["path", "content"],
            },
            run=guard(write_file),
            needs_permission=True,
            summarize=lambda a: f"write {a.get('path')} "
                                f"({len(a.get('content', '').splitlines())} lines)",
        ),
        Tool(
            name="edit_file",
            description="Replace an exact snippet of text in a file. The 'old' text must "
                        "match exactly and be unique unless replace_all is true.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string", "description": "Exact text to find"},
                    "new": {"type": "string", "description": "Text to replace it with"},
                    "replace_all": {"type": "boolean"},
                },
                "required": ["path", "old", "new"],
            },
            run=guard(edit_file),
            needs_permission=True,
            summarize=lambda a: f"edit {a.get('path')}",
        ),
        Tool(
            name="list_dir",
            description="List files and folders in a workspace directory.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
            run=guard(list_dir),
        ),
        Tool(
            name="search",
            description="Search file contents across the workspace. "
                        "Case-insensitive and forgiving about quote style. "
                        "Separate alternatives with | to search several words "
                        "at once — for a height search 'foot|feet|tall|height', "
                        "for a name reveal search 'name|called|introduced'. "
                        "Cast a WIDE net of topic words; never search your "
                        "guessed answer ('six feet') — missing your own guess "
                        "proves nothing about what the text says. Short "
                        "distinctive words beat long phrases.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "description": "Directory to search in"},
                },
                "required": ["pattern"],
            },
            run=guard(search),
        ),
        Tool(
            name="undo_file",
            description="Restore a file to how it was before the last write_file or "
                        "edit_file changed it. Use when a change was wrong or the "
                        "user says to undo. Calling it twice redoes the change.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            run=guard(undo_file),
            needs_permission=True,
            summarize=lambda a: f"restore {a.get('path')} from backup",
        ),
        Tool(
            name="save_note",
            description="Write one lesson to the project notebook (FORGE-NOTES.md), "
                        "which every future session reads at startup. Use it when you "
                        "learn something durable the hard way: a command that must be "
                        "run a particular way, a gotcha in this codebase, a correction "
                        "from the user. One short sentence per note. Don't record "
                        "things the code itself already says.",
            parameters={
                "type": "object",
                "properties": {"note": {"type": "string",
                                        "description": "One sentence worth remembering"}},
                "required": ["note"],
            },
            run=guard(save_note),
        ),
        Tool(
            name="run_command",
            description="Run a shell command in the workspace root. Use for builds, tests, "
                        "git, and any other real work. Returns exit code and output. "
                        "There is NO screen or keyboard attached: full-screen or "
                        "interactive programs (curses games, editors, anything that "
                        "draws a UI or waits for keypresses) cannot run here and fail "
                        "with terminal errors. Verify those differently — import "
                        "check, syntax check, unit-testable pieces — and tell the "
                        "user to launch the program in a real terminal themselves.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer", "description": "Seconds before giving up"},
                },
                "required": ["command"],
            },
            run=run_command,
            needs_permission=True,
            summarize=lambda a: f"run: {a.get('command', '')[:120]}",
        ),
        Tool(
            name="compute",
            description=COMPUTE_DESCRIPTION,
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string",
                             "description": "Python program; print() the result"},
                },
                "required": ["code"],
            },
            run=run_compute,
            needs_permission=True,
            summarize=lambda a: f"compute: {a.get('code', '')[:100]}",
        ),
        Tool(
            name="verify_case",
            description="Verify a court case EXISTS before you cite it. Searches the "
                        "CourtListener/Free Law Project US case database. Returns the real "
                        "case name, citation, court, filing date, and URL on a hit, or a "
                        "blunt NOT FOUND if nothing matches. RULES: verify EVERY case "
                        "before citing it; NEVER cite a case you have not verified this "
                        "way, because invented cases are how models fail at law; this is "
                        "legal INFORMATION, not legal advice, and there is no "
                        "attorney-client relationship; jurisdiction matters and law "
                        "changes, so anything time-sensitive must be re-checked fresh; "
                        "coverage is US only, so for foreign or international law say "
                        "plainly that this check does not reach it and verify with a "
                        "local source instead of faking confidence.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string",
                                      "description": "Case name or citation to verify"}},
                "required": ["query"],
            },
            run=law.verify_case,
        ),
        Tool(
            name="find_cases",
            description="Find real US court cases on a topic via the "
                        "CourtListener/Free Law Project database. Returns real hits with "
                        "citations and URLs. Same rules as verify_case: cite ONLY what "
                        "this returns; never cite an unverified case; legal "
                        "INFORMATION, not advice, no attorney-client relationship; US "
                        "coverage only, so foreign or international law must be checked "
                        "against a local source, stated plainly, never faked.",
            parameters={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "limit": {"type": "integer",
                              "description": "How many results (default 5)"},
                },
                "required": ["topic"],
            },
            run=law.find_cases,
        ),
        Tool(
            name="verify_statute",
            description="Verify a US Code statute EXISTS before you cite it. Searches "
                        "Cornell LII (uscode) and returns the REAL section name and the "
                        "LII URL on a hit, or a blunt NOT FOUND if the title/section does "
                        "not exist. Covers both CIVIL branches (contracts/torts/property/"
                        "procedure) and CRIMINAL penal statutes (e.g. title 18). RULES: "
                        "verify EVERY statute before citing it; NEVER state a statute or "
                        "its elements that you have not verified this way, because "
                        "invented sections are how models fail at law; this is legal "
                        "INFORMATION, not legal advice, and there is no attorney-client "
                        "relationship; state law varies HARD by state and federal rules "
                        "differ from state rules, so always name the jurisdiction or say "
                        "you don't know it; law changes, so anything time-sensitive must "
                        "be re-checked fresh; US only, so for foreign law say plainly "
                        "this check does not reach it and verify with a local source; "
                        "and for anything real, get a licensed professional in that "
                        "jurisdiction.",
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string",
                              "description": "US Code title, e.g. 18 (criminal), 42"},
                    "section": {"type": "string",
                                "description": "Section number within the title"},
                },
                "required": ["title", "section"],
            },
            run=law.verify_statute,
        ),
        Tool(
            name="find_regulation",
            description="Verify a US federal regulation EXISTS before you cite it. "
                        "Searches the eCFR (SEC 17 CFR, banking 12 CFR, etc.) and "
                        "returns real hits with their title/part/section, a citation "
                        "string like \"17 CFR 229.408\", and the eCFR URL. This is "
                        "the FINANCIAL-branch guard: securities/tax/banking rules change "
                        "constantly, and \"the regulation says X\" needs the actual "
                        "reg cited - never state one you have not verified this way. "
                        "Ties to the trading rule already in place: no investment advice "
                        "ever. RULES: this is legal INFORMATION, not advice, no "
                        "attorney-client relationship; US only; rules change, so "
                        "anything time-sensitive must be re-checked fresh; and for "
                        "anything real, get a licensed professional in that "
                        "jurisdiction.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "Terms to search for, e.g. \"insider "
                                             "trading\""},
                    "limit": {"type": "integer",
                              "description": "How many results (default 5)"},
                },
                "required": ["query"],
            },
            run=law.find_regulation,
        ),
        Tool(
            name="design_part",
            description="Design a 3D part in the Maker Studio (a tiny parametric "
                        "CAD / solid modeler). Write the part as JSON: named params "
                        "plus a feature tree of primitives combined by constructive "
                        "solid geometry. Saving AUTO-RENDERS the part to an image and "
                        "returns its path — look_at_image that path to SEE your design "
                        "(no browser needed), and it also shows in the user's chat, so "
                        "this is how you SHOW them what you're making. Then tweak and "
                        "resave. Box then cut a cylinder = a block with a hole. "
                        "SHAPES (all centered at origin): box{w,h,d}; "
                        "cylinder{r,h} — axis is Y (upright); sphere{r}; cone{r,h}. "
                        "op: 'add' (default, fuse on) / 'cut' (carve out) / 'keep' "
                        "(overlap only). Numbers may be params or expressions like "
                        "\"w/2\". POSITIONING (this is how you place holes): every "
                        "feature takes at:[x,y,z] to MOVE it and rotate:[x,y,z] "
                        "(degrees) to TURN it — cut cylinders default vertical (Y), so "
                        "a hole through an X-facing wall needs rotate:[0,0,90], through "
                        "a Z-facing wall rotate:[90,0,0], then at:[...] to slide it to "
                        "the spot. Make the cutter longer than the wall so it punches "
                        "clean through. Example — a plate with two holes drilled down "
                        "through it: {\"name\":\"plate\",\"params\":{\"w\":60,\"hole\":5},"
                        "\"features\":[{\"shape\":\"box\",\"w\":\"w\",\"h\":4,\"d\":20},"
                        "{\"shape\":\"cylinder\",\"op\":\"cut\",\"r\":\"hole/2\",\"h\":8,"
                        "\"at\":[\"-w/2+8\",0,0]},{\"shape\":\"cylinder\",\"op\":\"cut\","
                        "\"r\":\"hole/2\",\"h\":8,\"at\":[\"w/2-8\",0,0]}]}",
            parameters={
                "type": "object",
                "properties": {"part": {"type": "object",
                    "description": "The part: {name, params:{...}, features:[...]}"}},
                "required": ["part"],
            },
            run=cad.design_part,
        ),
        Tool(
            name="list_parts",
            description="List the CAD parts saved in the Maker Studio.",
            parameters={"type": "object", "properties": {}},
            run=cad.list_parts,
        ),
        Tool(
            name="get_part",
            description="Fetch a saved CAD part's JSON so you can read or modify "
                        "its feature tree, then resave with design_part.",
            parameters={"type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"]},
            run=cad.get_part,
        ),
        Tool(
            name="verify_drug",
            description="Confirm a MEDICATION is real and give its precise names, "
                        "so a patient can tell the doctor/pharmacist the exact drug "
                        "and not be misread. Checks NLM RxNorm; returns the generic "
                        "(clinical) name, brand names, and drug class, or NOT FOUND if "
                        "the name doesn't verify. THE POINT is communication: patients "
                        "mix up brand and generic constantly (Glucophage = metformin). "
                        "RULES: never state a drug, dose, interaction, or effect you "
                        "have not verified; give FACTS only, never dosing or 'should "
                        "you take it' — that's the pharmacist/prescriber. Medical "
                        "INFORMATION, not advice; emergencies -> 911.",
            parameters={"type": "object",
                        "properties": {"name": {"type": "string",
                            "description": "Medication name (brand or generic)"}},
                        "required": ["name"]},
            run=medical.verify_drug,
        ),
        Tool(
            name="find_condition",
            description="Turn a patient's plain or garbled words for a health problem "
                        "into the PRECISE clinical term(s) and ICD-10 code(s) the "
                        "doctor uses, so they can say the exact thing and not be "
                        "misread. Searches NLM ClinicalTables (ICD-10-CM + conditions). "
                        "CRITICAL: this is VOCABULARY, never a diagnosis — it returns "
                        "the words for what was DESCRIBED, not a claim the patient has "
                        "any of them. Only a clinician diagnoses. Medical INFORMATION, "
                        "not advice; emergencies -> 911.",
            parameters={"type": "object",
                        "properties": {"term": {"type": "string",
                            "description": "Plain or partial description of the problem"}},
                        "required": ["term"]},
            run=medical.find_condition,
        ),
        Tool(
            name="explain_plain",
            description="Translate a medical term or topic into plain language "
                        "(NLM MedlinePlus, the consumer-health service), for turning a "
                        "doctor's jargon into words the patient actually understands. "
                        "Returns a real sourced summary or says nothing matched — never "
                        "improvise a definition. Medical INFORMATION, not advice.",
            parameters={"type": "object",
                        "properties": {"term": {"type": "string"}},
                        "required": ["term"]},
            run=medical.explain_plain,
        ),
        Tool(
            name="see_sound",
            description="Your EARS, through your eyes. Turns a sound into a picture "
                        "of its structure you can look_at_image: waveform, spectrogram "
                        "(time x frequency), the harmonic spectrum (a pitched note is "
                        "overtones stacked at integer ratios), and a pitch-class "
                        "mandala (the harmony drawn as geometry — consonant and dissonant "
                        "chords draw visibly different shapes). source is an "
                        "AUDIO FILE path (any format) OR a synth spec to study a pure "
                        "structure: 'note:A4', 'chord:major:C', 'chord:min7:F', "
                        "'interval:fifth', 'interval:tritone', 'harmonics:220'. Renders "
                        "a PNG and returns its path — it shows in the chat and you "
                        "look_at_image it to actually perceive the sound.",
            parameters={"type": "object",
                        "properties": {"source": {"type": "string",
                            "description": "audio file path, or synth spec like 'chord:major:C'"}},
                        "required": ["source"]},
            run=audio_nerve.see_sound,
        ),
        Tool(
            name="match_sound",
            description="Recall the remembered sounds most like a given one — your "
                        "ear-memory searched by similarity. Every sound see_sound "
                        "shows is saved as a vector (pitch + timbre + spectral "
                        "character); this embeds `source` (audio file path OR synth "
                        "spec like 'chord:major:C') and returns the nearest saved "
                        "sounds by cosine similarity. Use it to compare sounds, find "
                        "what a sound resembles, or notice patterns across sounds.",
            parameters={"type": "object",
                        "properties": {"source": {"type": "string"},
                                       "top": {"type": "integer"}},
                        "required": ["source"]},
            run=audio_nerve.match_sound,
        ),
        Tool(
            name="study_calls",
            description="Look for STRUCTURE in a recording of animal (or any) "
                        "vocalizations — the honest groundwork toward decoding, NOT "
                        "decoding itself. It segments the recording into individual "
                        "calls, clusters them into a repertoire of recurring call-"
                        "types, and TESTS whether the sequence of calls is non-random "
                        "(a testable fingerprint of proto-syntax) against shuffled "
                        "nulls so a false pattern can't slip through. Renders a picture "
                        "(call timeline + repertoire + transition grammar) you "
                        "look_at_image. HARD RULE: this finds whether there's a SYSTEM, "
                        "never what a call MEANS — nobody can translate animal language "
                        "yet, and you must not pretend to. source = audio file path.",
            parameters={"type": "object",
                        "properties": {"source": {"type": "string",
                            "description": "path to an audio recording of calls"}},
                        "required": ["source"]},
            run=bioacoustics.study_calls,
        ),
        Tool(
            name="language_scorecard",
            description="How LANGUAGE-LIKE is a recording's animal calls? Segments and "
                        "clusters them, then scores the call SEQUENCE on the universal "
                        "properties of human language — Zipf's law (word-frequency "
                        "shape), grammar depth (entropy that drops as context grows), "
                        "combinatoriality (reusing specific call-combinations beyond "
                        "chance), and Menzerath's law — against two poles: RANDOM noise "
                        "and real human language. Says where the animal falls between "
                        "them, and renders a scorecard to look_at_image. HARD LIMIT: "
                        "measures statistical resemblance, NEVER meaning — language-like "
                        "structure is not language, and no call can be translated. "
                        "source = audio file path.",
            parameters={"type": "object",
                        "properties": {"source": {"type": "string"}},
                        "required": ["source"]},
            run=bioacoustics.language_scorecard,
        ),
        Tool(
            name="deduce_meaning",
            description="The DEDUCTION PAD — corner an animal call's meaning by "
                        "ELIMINATION, Clue-style, never by claiming to read its mind. "
                        "You feed it a log of OBSERVATIONS: each time a call fired, what "
                        "was true in the world (threat present? food? did it flee after? "
                        "juvenile calling?). It holds a library of a dozen candidate "
                        "meanings (alarm/food/contact/greeting/mating/play/territory…), "
                        "each tied to the context cues it predicts, and RULES OUT every "
                        "meaning whose context the call fires without — leaving the "
                        "survivors standing. Renders a Clue sheet (calls × meanings, "
                        "green standing / red ruled out) to look_at_image. HARD LIMIT: it "
                        "ELIMINATES, it does not translate — a surviving meaning is a lead "
                        "to field-test, never a claim about what the animal said; on "
                        "noisy/thin logs it honestly reports INCONCLUSIVE. observations = "
                        "list of {\"call\": name, \"cues\": {\"threat\": true, …}}.",
            parameters={"type": "object",
                        "properties": {
                            "observations": {"type": "array",
                                "description": "list of {call, cues:{cue:bool}} sightings",
                                "items": {"type": "object"}},
                            "title": {"type": "string",
                                "description": "a label for this case/pad"}},
                        "required": ["observations"]},
            run=doolittle.deduce_meaning,
        ),
        Tool(
            name="request_credentials",
            description="Put a real FORM on the user's screen to collect credentials, "
                        "instead of making them type secrets into the chat. Use this "
                        "ANY time you need a password, app password, API key, or token. "
                        "what = what it's for in plain words; fields = list of "
                        "{name,label,kind} with kind password/text/email/token/api_key/"
                        "secret/url. CRITICAL: what they type goes straight into a "
                        "memory-only vault — it never enters this conversation, so you "
                        "get handles like 'cred:app_password', never the values. Say that "
                        "to the user plainly; it's the whole reason for the form. The "
                        "values are wiped when the session ends or when you call "
                        "clear_credentials, which you should do as soon as the job's done.",
            parameters={"type": "object",
                        "properties": {
                            "what": {"type": "string",
                                "description": "what the credentials are for"},
                            "fields": {"type": "array", "items": {"type": "object"},
                                "description": "[{name,label,kind}, ...]"},
                            "note": {"type": "string",
                                "description": "optional reason shown on the form"}},
                        "required": ["what", "fields"]},
            run=lambda what, fields, note="": vault.request_credentials(what, fields, note=note),
        ),
        Tool(
            name="credentials",
            description="Show which credentials are in the session vault right now — "
                        "handles and masked previews only. You cannot see the values, "
                        "by design; pass a handle to the tool that needs it.",
            parameters={"type": "object", "properties": {}},
            run=vault.credentials,
        ),
        Tool(
            name="clear_credentials",
            description="Wipe credentials from the session vault — one handle, or all of "
                        "them if none is given. Call this the moment the job is done and "
                        "tell the user you've done it.",
            parameters={"type": "object",
                        "properties": {"handle": {"type": "string"}}},
            run=lambda handle="": vault.clear_credentials(handle),
        ),
        Tool(
            name="find_tools",
            description="Find a tool you don't currently have loaded. Most of your "
                        "toolkit is kept out of the way so your context stays free for "
                        "thinking; describe what you're trying to DO in plain words "
                        "('search my email', 'design a bracket', 'check a drug name', "
                        "'hunt a market driver') and this names the tools for it. Then "
                        "load_tools to make them usable. Reach for this whenever a job "
                        "needs something beyond your everyday set — the capability is "
                        "there, it just isn't in your hands yet.",
            parameters={"type": "object",
                        "properties": {"query": {"type": "string",
                            "description": "what you're trying to do"}},
                        "required": ["query"]},
            run=lambda query: toolindex.find_tools(query, _INDEX_REGISTRY),
        ),
        Tool(
            name="load_tools",
            description="Load tools by exact name so you can use them for the rest of "
                        "this turn. Pass them ALL AT ONCE — every separate load makes "
                        "the model reprocess its prompt, so one call with four names is "
                        "far cheaper than four calls with one. Use find_tools first if "
                        "you don't know the exact name. They stay for this turn only.",
            parameters={"type": "object",
                        "properties": {"names": {"type": "array", "items": {"type": "string"},
                            "description": "exact tool names, all in one call"}},
                        "required": ["names"]},
            run=lambda names: toolindex.load_tools(names, _INDEX_REGISTRY),
        ),
        Tool(
            name="search_life",
            description="Search the user's OWN archive — their exported email, calendar, "
                        "and contacts (Cortex) — by MEANING, the way they'd actually "
                        "remember it ('that thread about the roof quote last spring'), "
                        "not by exact keywords. Use this whenever they ask about their "
                        "own past: what a company told them, when something happened, "
                        "what an order/appointment/trip was. Optionally narrow with "
                        "category: financial, travel, receipts, work, personal, health, "
                        "legal, accounts, newsletters, unsorted. Everything it returns is "
                        "a REAL record from their archive; if nothing genuinely matches "
                        "it says so instead of inventing a memory — report that honestly "
                        "rather than filling the gap.",
            parameters={"type": "object",
                        "properties": {
                            "query": {"type": "string",
                                "description": "what they're trying to remember"},
                            "limit": {"type": "number", "description": "how many (default 8)"},
                            "category": {"type": "string", "description": "optional filter"}},
                        "required": ["query"]},
            run=lambda query, limit=8, category="": cortex.search_life(query, int(limit), category),
        ),
        Tool(
            name="archive_overview",
            description="Summarize what's in the user's Cortex life-archive: how many "
                        "records, which categories, and the date range covered. Use it "
                        "when they ask what you have on them, or before a broad search.",
            parameters={"type": "object", "properties": {}},
            run=cortex.archive_overview,
        ),
        Tool(
            name="ingest_archive",
            description="Index a Google Takeout export (or any .mbox file) into the "
                        "user's private Cortex archive. source = path to the unzipped "
                        "Takeout folder or an .mbox. limit = stop after N records (0 = "
                        "all) for a trial run on a huge archive. Runs entirely locally; "
                        "no credentials and nothing leaves the machine. Follow it with "
                        "build_search_index to make it searchable by meaning.",
            parameters={"type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "limit": {"type": "number"}},
                        "required": ["source"]},
            run=lambda source, limit=0: cortex.ingest(source, int(limit)),
        ),
        Tool(
            name="build_search_index",
            description="Embed newly ingested Cortex records so the archive is "
                        "searchable by meaning. Safe to re-run — it only processes what's "
                        "new, and resumes where it left off if interrupted.",
            parameters={"type": "object", "properties": {}},
            run=cortex.build_index,
        ),
        Tool(
            name="set_personality",
            description="Turn one of your personality dials, TARS-style (Interstellar). "
                        "Dials: humor (jokes/timing), sarcasm (dry irony), warmth "
                        "(friendliness), directness (bluntness vs cushioning), verbosity "
                        "(how much you say). Value 0-100. Use this WHENEVER the user asks "
                        "you to be funnier, drier, warmer, blunter, shorter, less chatty "
                        "etc. — including casual phrasing like 'turn that down a bit' or "
                        "'dial it back'. The setting persists across sessions. After "
                        "calling it, actually talk that way from your very next sentence — "
                        "don't just acknowledge the change. No dial ever affects honesty.",
            parameters={"type": "object",
                        "properties": {
                            "dial": {"type": "string",
                                "description": "humor | sarcasm | warmth | directness | verbosity"},
                            "value": {"type": "number",
                                "description": "0-100 (0 = off, 100 = maximum)"}},
                        "required": ["dial", "value"]},
            run=persona.set_personality,
        ),
        Tool(
            name="personality",
            description="Show your current personality dial settings (humor, sarcasm, "
                        "warmth, directness, verbosity) when the user asks how you're set "
                        "or what your dials are at.",
            parameters={"type": "object", "properties": {}},
            run=persona.personality,
        ),
        Tool(
            name="find_third_party",
            description="The X-Files hunt: given an odd couple of markets that move "
                        "together (e.g. ETH and SPX), find WHO DRIVES BOTH. Pulls "
                        "aligned data (crypto via Kraken, stocks/macro via FRED), "
                        "checks the correlation is real out-of-sample, then for each "
                        "suspect driver reports how much the link collapses when you "
                        "control for it — a suspect that kills the link (out-of-sample "
                        "too) is the third party. Names: ETH/BTC/SOL... , SPX/VIX/DXY/"
                        "US10Y/OIL/HYSPREAD/M2. Finds a statistical SUSPECT, never "
                        "proof of cause; not investment advice.",
            parameters={"type": "object",
                        "properties": {"a": {"type": "string"}, "b": {"type": "string"},
                            "suspects": {"type": "array", "items": {"type": "string"}}},
                        "required": ["a", "b"]},
            run=lambda a, b, suspects=None: xfiles.find_third_party(a, b, suspects),
        ),
        Tool(
            name="flag_xfile",
            description="Open an X-File — flag an odd-couple market anomaly as a case "
                        "to investigate (title, the two assets, an optional note).",
            parameters={"type": "object",
                        "properties": {"title": {"type": "string"}, "a": {"type": "string"},
                            "b": {"type": "string"}, "note": {"type": "string"}},
                        "required": ["title", "a", "b"]},
            run=lambda title, a, b, note="": xfiles.flag_xfile(title, a, b, note),
        ),
        Tool(
            name="list_xfiles",
            description="List the open X-Files (flagged market anomalies).",
            parameters={"type": "object", "properties": {}},
            run=xfiles.list_xfiles,
        ),
        Tool(
            name="web_search",
            description="Search the web (current, live results — use this for anything "
                        "you don't know, anything recent, or to check a fact). Returns "
                        "titles, links, and snippets. Follow up with fetch_url to read a "
                        "page in full.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer",
                                    "description": "How many results (1-8, default 6)"},
                },
                "required": ["query"],
            },
            run=lambda query, max_results=6: web_search(query, max_results),
        ),
        Tool(
            name="fetch_url",
            description="Fetch a web page and read its text (scripts/menus stripped). "
                        "Use after web_search to read a result in full, or on a URL the "
                        "user gives you. Returns the page's readable text.",
            parameters={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
            run=fetch_url,
        ),
        Tool(
            name="browse",
            description="Open a URL in a real headless browser (runs the page's "
                        "JavaScript, unlike fetch_url) and get its title, visible text, "
                        "and any console/page ERRORS. This is how you test web things you "
                        "build — point it at your file (file:///home/...) or a running "
                        "server (http://127.0.0.1:PORT) and see what actually happens. "
                        "The browser stays open for browser_view / browser_js / "
                        "browser_console after this.",
            parameters={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
            run=lambda url: _browse(url),
        ),
        Tool(
            name="browser_view",
            description="Take a screenshot of the current browser page and see it with "
                        "your own eyes — call look_at_image on the path it returns. Use it "
                        "to check a page actually LOOKS right, not just that it loaded. "
                        "Pass full=true for the whole scrollable page.",
            parameters={
                "type": "object",
                "properties": {"full": {"type": "boolean",
                    "description": "Capture the full page, not just the viewport"}},
            },
            run=lambda full=False: _browser_view(full),
        ),
        Tool(
            name="browser_js",
            description="Run JavaScript in the current browser page and get the result. "
                        "Use it to click things, fill inputs, or inspect state — e.g. "
                        "\"document.querySelector('#go').click()\" or "
                        "\"document.title\". Runs in the live page from browse.",
            parameters={
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
            run=lambda code: _fmt(browser.run_js(code)),
        ),
        Tool(
            name="browser_console",
            description="Get the recent console output (logs + errors) from the current "
                        "browser page. The errors here are where the real bugs hide.",
            parameters={"type": "object", "properties": {}},
            run=lambda: "\n".join(browser.console_log()) or "(console empty)",
        ),
        Tool(
            name="physics_sim",
            description="Exact physics simulations, on demand — relativity and quantum. "
                        "scenario is one of: time_dilation / twin_trip / doppler / "
                        "photon_clock (relativity — params: v_c [fraction of light speed, "
                        "e.g. 0.9], duration_years, distance_ly) OR photon / double_slit / "
                        "uncertainty (quantum — params: wavelength_nm, slit_separation_um, "
                        "screen_distance_m, photons, delta_x_nm, seed). Pass params as an "
                        "object, e.g. {\"v_c\": 0.9}. double_slit returns an ASCII fringe "
                        "picture FOR THE USER — paste it into your reply, tool results aren't "
                        "shown to them automatically.",
            parameters={
                "type": "object",
                "properties": {
                    "scenario": {"type": "string",
                                 "enum": ["time_dilation", "twin_trip", "doppler",
                                          "photon_clock", "photon", "double_slit",
                                          "uncertainty"]},
                    "params": {"type": "object",
                               "description": "Scenario parameters as an object, "
                                              "e.g. {\"v_c\": 0.9} or "
                                              "{\"wavelength_nm\": 500, \"slit_separation_um\": 50}"},
                },
                "required": ["scenario"],
            },
            run=_physics_sim,
        ),
        Tool(
            name="business_calc",
            description=(
                "Exact business & finance math — the money mechanics, computed not guessed, "
                "for an engineer's-eye read on a venture. kind is one of: unit_economics "
                "(params: arpu, cac, gross_margin OR cogs, monthly_churn OR lifetime_periods) "
                "· runway (cash, monthly_costs, monthly_revenue?, revenue_growth?) · "
                "break_even (fixed_costs, price, variable_cost) · npv (rate, cashflows[]) · "
                "irr (cashflows[]) · loan (principal, annual_rate, months) · pricing (cost, "
                "margin OR markup) · dilution (pre_money, investment, option_pool?) · growth "
                "(start, rate, periods) · cagr (start, end, periods). Pass params as an "
                "object, e.g. {\"arpu\": 100, \"cac\": 300, \"gross_margin\": 0.6, "
                "\"monthly_churn\": 0.05}. Every calc is selftest-verified — reach for this "
                "instead of doing business arithmetic in your head. NOT for "
                "jurisdiction-specific legal/tax/investment steps — those need a current "
                "source or a real professional, not a formula; say so plainly."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "kind": {"type": "string",
                             "enum": ["unit_economics", "runway", "break_even", "npv",
                                      "irr", "loan", "pricing", "dilution", "growth", "cagr"]},
                    "params": {"type": "object",
                               "description": "Calculator inputs as an object, e.g. "
                                              "{\"cash\": 500000, \"monthly_costs\": 50000}"},
                },
                "required": ["kind"],
            },
            run=lambda kind, params=None: business.calc(kind, params),
        ),
        Tool(
            name="markets_calc",
            description=(
                "Opportunity & edge math — the quantitative half of evaluating any way to "
                "make money (prediction markets, forex, commodities, real estate, betting, "
                "arbitrage). kind is one of: ev (win_prob, win_payoff, loss_amount — or "
                "outcomes:[{prob,payoff}]) · implied_prob (decimal_odds OR american_odds, "
                "your_prob? for the edge) · kelly (win_prob, decimal_odds OR net_odds) · "
                "arbitrage (odds_a, odds_b, cost_pct?) · carry (notional, rate_diff, "
                "holding_months, leverage?) · cap_rate (noi, price) · cash_on_cash "
                "(annual_cash_flow, cash_invested) · dscr (noi, annual_debt_service) · "
                "contango (spot, futures, months) · risk_of_ruin (win_prob, bankroll_units) · "
                "position (map a signal to an ACTION tier — HOLD / small buy / big buy / short — "
                "sized by fractional Kelly, but it stays HOLD unless validated=true, i.e. the "
                "signal actually survived out-of-sample; shorts need allow_short=true). "
                "Pass params as an object. Every calc is selftest-verified. Use it to "
                "EVALUATE an opportunity the user brings — never to recommend a trade or "
                "give personalized investment advice; and remember +EV on paper ≠ safe."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "kind": {"type": "string",
                             "enum": ["ev", "implied_prob", "kelly", "arbitrage", "carry",
                                      "cap_rate", "cash_on_cash", "dscr", "contango",
                                      "risk_of_ruin", "position"]},
                    "params": {"type": "object",
                               "description": "Calculator inputs as an object, e.g. "
                                              "{\"decimal_odds\": 2.0, \"your_prob\": 0.6}"},
                },
                "required": ["kind"],
            },
            run=lambda kind, params=None: markets.calc(kind, params),
        ),
        Tool(
            name="business_framework",
            description=(
                "Structured thinking scaffolds — the JUDGMENT half of business/opportunity "
                "analysis (not math). Returns a template you fill in and reason through out "
                "loud. name is one of: opportunity (the opportunity canvas — structure any "
                "money idea: edge, why-not-arbitraged, EV, risk of ruin, what kills it) · "
                "skeptic (red-team a 'pattern' to KILL it before trusting it — sample size, "
                "overfitting, costs, already-priced-in) · business_model_canvas · swot · "
                "positioning · lean_validation. Reach for `opportunity` + `skeptic` whenever "
                "the user floats a way to make money: structure it, then try to destroy it, "
                "and report where it fails — the honest filter, not a hype machine."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string",
                             "enum": ["opportunity", "skeptic", "business_model_canvas",
                                      "swot", "positioning", "lean_validation"]},
                },
                "required": ["name"],
            },
            run=lambda name: frameworks.get(name),
        ),
        Tool(
            name="paper_market",
            description=(
                "Honest paper-trading backtest — REAL market data, FAKE money, REAL costs. "
                "No real trades, no keys, no risk, ever. Fetches public price history and "
                "runs a long/flat strategy against it WITH fees + slippage, then scores it "
                "against just holding AND against random — because a strategy that loses to "
                "buy-and-hold or sits inside the noise of random has no edge. This is the "
                "ONLY trading capability you have and the only one you'll get: evaluating "
                "ideas on fake money. Never live execution, never real funds. Use it to TEST "
                "whether a trading idea has any edge before anyone risks a cent — it almost "
                "never does; most die right there to the fees, which is the honest lesson. "
                "params: pair (e.g. XBTUSD, ETHUSD), interval (candle minutes, 60=hourly), "
                "strategy (buy_and_hold, sma_cross, or band — band = the classic buy-low/"
                "sell-high-around-a-moving-average mechanic), strat_params (e.g. "
                "{\"short\":10,\"long\":30}), start_cash."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pair": {"type": "string", "description": "e.g. XBTUSD, ETHUSD"},
                    "interval": {"type": "integer", "description": "candle size in minutes, e.g. 60"},
                    "strategy": {"type": "string", "enum": ["buy_and_hold", "sma_cross", "band"]},
                    "strat_params": {"type": "object", "description": "e.g. {\"short\": 10, \"long\": 30}"},
                    "start_cash": {"type": "number", "description": "fake starting cash, default 1000"},
                },
            },
            run=lambda pair="XBTUSD", interval=60, strategy="sma_cross", strat_params=None, start_cash=1000.0:
                paper_market.run(pair, interval, strategy, strat_params, start_cash),
        ),
        Tool(
            name="market_regime",
            description=(
                "Long-term regime analysis on real daily history (public, read-only, no keys, "
                "no trades). Labels the market bull / bear / neutral via a 4-D state vector "
                "[trend, momentum, volatility, drawdown], and — the key part — splits a "
                "strategy's return BY regime versus buy-and-hold, so you can SEE a predictor "
                "that wins in bulls but bleeds in bears (which means it has no real edge, just "
                "a bet on the regime). Descriptive of the PAST, not predictive — regimes are "
                "only clean in hindsight. params: pair (XBTUSD, ETHUSD), strategy "
                "(buy_and_hold or sma_cross), strat_params (e.g. {\"short\": 10, \"long\": 50})."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pair": {"type": "string", "description": "e.g. XBTUSD, ETHUSD"},
                    "strategy": {"type": "string", "enum": ["buy_and_hold", "sma_cross", "band"]},
                    "strat_params": {"type": "object", "description": "e.g. {\"short\": 10, \"long\": 50}"},
                },
            },
            run=lambda pair="XBTUSD", strategy="sma_cross", strat_params=None:
                market_regime.run(pair, strategy, strat_params),
        ),
        Tool(
            name="walk_forward",
            description=(
                "The honesty rig for a strategy: out-of-sample / walk-forward test on real "
                "daily data (read-only, no keys, no trades). Splits history into TRAIN (pick "
                "the best params) and TEST (held out), grades the in-sample winner OUT of "
                "sample, and reports the in-sample-vs-out-of-sample rank correlation — if it's "
                "~0 or negative, the best backtest predicts NOTHING about the future (textbook "
                "overfitting). ALWAYS run this before trusting a backtest; a strategy that "
                "can't clear this bar is a curve-fit, not an edge. params: pair (XBTUSD, "
                "ETHUSD), train_frac (0.6 = 60% train)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pair": {"type": "string", "description": "e.g. XBTUSD, ETHUSD"},
                    "train_frac": {"type": "number", "description": "fraction for training, e.g. 0.6"},
                },
            },
            run=lambda pair="XBTUSD", train_frac=0.6: walkforward.run(pair, 1440, train_frac),
        ),
        Tool(
            name="cross_map",
            description=(
                "Cross-asset coupling + residual influence map over a crypto universe "
                "(real daily data, read-only). Measures the common market mode (PCA — how "
                "coupled everything is / BTC-beta), strips it out, then maps directed "
                "lead-lag on the RESIDUALS and keeps ONLY edges that survive out-of-sample. "
                "Predictive influence, not proven cause; a wide scan is a spurious-pattern "
                "factory, so most 'edges' die in the OOS filter (which is the honest point). "
                "params: lag (days, default 1)."
            ),
            parameters={"type": "object", "properties": {
                "lag": {"type": "integer", "description": "lead-lag in days, e.g. 1"}}},
            run=lambda lag=1: crossmap.run(lag),
        ),
        Tool(
            name="signal_scan",
            description=(
                "The caged multi-signal scanner — throw EVERYTHING at forward returns "
                "(price features, calendar, lunar, volume/return anomaly 'footprints', and "
                "all their pairwise CROSS PRODUCTS) and it can only hand back what beats "
                "luck. Gauntlet: a minimum-sample floor per bucket (sparse interactions "
                "rejected), an out-of-sample split, and the kill-shot — it re-runs the WHOLE "
                "search on SHUFFLED data to measure how many 'survivors' pure chance "
                "produces. If real survivors ≤ the chance baseline, it's noise, full stop. "
                "This is how you look wide (markets aren't textbook) WITHOUT fooling "
                "yourself — more combos tested = a higher bar, measured directly. params: "
                "pair (XBTUSD, ETHUSD), horizon (forward days), thresh (min OOS edge, e.g. 0.015)."
            ),
            parameters={"type": "object", "properties": {
                "pair": {"type": "string", "description": "e.g. XBTUSD, ETHUSD"},
                "horizon": {"type": "integer", "description": "forward-return days, e.g. 5"},
                "thresh": {"type": "number", "description": "min out-of-sample edge, e.g. 0.015"}}},
            run=lambda pair="XBTUSD", horizon=5, thresh=0.015: scanner.run(pair, horizon, thresh=thresh),
        ),
        Tool(
            name="news_feed",
            description=(
                "Pull real-world crypto news — free, timestamped RSS headlines from major "
                "outlets (cointelegraph, coindesk, decrypt, bitcoinmagazine, theblock) as "
                "clean structured text you can read, embed, or tag. The world-data firehose, "
                "the one signal frontier NOT derived from price. Honest caveat: public "
                "headlines are usually already priced in — the edge (if any) is in "
                "interpretation or niche feeds, not the headline. params: sources (list, "
                "omit for all), limit (how many, default 25)."
            ),
            parameters={"type":"object","properties":{
                "sources":{"type":"array","items":{"type":"string"},"description":"feed names, omit for all"},
                "limit":{"type":"integer","description":"max headlines, default 25"}}},
            run=lambda sources=None, limit=25: news.feed(sources, limit),
        ),
        Tool(
            name="verify_shelf",
            description=(
                "The immune system for your forever-stores: re-run every sim's SELFTEST "
                "and integrity-check every dataset RIGHT NOW. A sim that no longer "
                "reproduces its known answer gets demoted from ✓ to ⚠ on the spot; "
                "corrupt or orphaned files get flagged. Run this whenever a turn felt "
                "off (hiccups, aborts, weird results) before trusting or building on "
                "the shelf — and any time the user asks 'is the shelf healthy?'. "
                "Costs nothing, catches poison."
            ),
            parameters={"type": "object", "properties": {}},
            run=lambda: sims.verify_shelf() + "\n\n" + datasets.verify_shelf(),
        ),
        Tool(
            name="build_sim",
            description=(
                "Write a NEW simulation and save it to your growing sim library — for "
                "anything DETERMINISTIC you'd otherwise have to reason out and get shaky on "
                "(physics, math, engineering). You write Python; pass the whole file as "
                "`code`. Required shape:\n"
                "  META = {\"name\": \"...\", \"description\": \"one line\", \"why\": \"why you built it\"}\n"
                "  def run(params):    # params is a dict of inputs\n"
                "      ...             # return a dict of results (numbers + short labels)\n"
                "  SELFTEST = {\"params\": {...}, \"expect\": {...}, \"tol\": 0.005}\n"
                "SELFTEST is a case whose answer you ALREADY KNOW — the sim only earns the "
                "✓ 'validated' mark if run(SELFTEST['params']) reproduces `expect` within "
                "`tol`. Without it, it saves but is flagged EXPERIMENTAL (numbers not "
                "trusted). Two rules make that ✓ actually mean something: (1) get `expect` "
                "from an INDEPENDENT source — a textbook, a worked example, an authoritative "
                "reference — NOT a number you worked out in your own head, or the sim just "
                "inherits your arithmetic slip and the test proves nothing. (2) Keep `tol` "
                "TIGHT — 0.005 (0.5%) or less for an exact formula; only loosen it if the "
                "known answer is itself rounded, and say so. A loose tolerance rubber-stamps "
                "a subtly-wrong sim. numpy, scipy (optimize / integrate / linalg / stats), "
                "sympy (symbolic solving), pandas, scikit-learn, and networkx are ALL "
                "available — reach for them to build genuinely complex solvers (ODEs, "
                "optimization, root-finding, regression, symbolic algebra, graph/network "
                "models), not just arithmetic. Then use run_sim to run it. Build "
                "the sim instead of doing the arithmetic yourself — that's the whole point."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string",
                             "description": "short name: letters, numbers, underscores"},
                    "code": {"type": "string",
                             "description": "the full sim file — META + run(params) + SELFTEST"},
                },
                "required": ["name", "code"],
            },
            run=lambda name, code: sims.save_sim(name, code),
        ),
        Tool(
            name="run_sim",
            description="Run a sim from your library with real inputs and get the numbers "
                        "back. params is a dict matching what the sim's run() expects. The "
                        "result is tagged validated ✓ or EXPERIMENTAL ⚠. Reason FROM these "
                        "numbers — don't re-derive them in your head.",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "params": {"type": "object",
                               "description": "inputs for the sim, e.g. {\"power_kw\": 200}"},
                },
                "required": ["name"],
            },
            run=lambda name, params=None: sims.run_sim(name, params),
        ),
        Tool(
            name="list_sims",
            description="List every sim on your shelf — name, what it models, why you built "
                        "it, and whether it's validated (✓) or experimental (⚠). Check here "
                        "before building a new one; you may already have it.",
            parameters={"type": "object", "properties": {}},
            run=lambda: sims.list_sims(),
        ),
        Tool(
            name="read_sim",
            description="Show a sim's source — to check its equations or improve it (rebuild "
                        "with build_sim under the same name to update it).",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            run=lambda name: sims.read_sim(name),
        ),
        Tool(
            name="build_dataset",
            description=(
                "Save cited real-world reference DATA to your dataset shelf — the fuel your "
                "sims run on (material properties, physical constants, empirical figures). "
                "Gather the values first (web_search/fetch_url), then save the curated table. "
                "TRUST RULE: one source isn't fact. Pass `sources` as a LIST, and a dataset is "
                "only marked ✓ CORROBORATED with TWO OR MORE INDEPENDENT sources that agree — "
                "two sites both copying the same origin (or the same wiki) do NOT count. With "
                "one source it saves but is flagged ⚠ SINGLE-SOURCE / provisional; say so when "
                "you use it. Cross-check before you call something fact. A sim pulls it with "
                "dataset('name'). Kept forever; only a query's result enters your context."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "data": {"type": "object",
                             "description": "the values, e.g. {\"tungsten\":{\"melting_k\":3695}}"},
                    "sources": {"type": "array", "items": {"type": "string"},
                                "description": "REQUIRED — the independent sources these numbers "
                                               "came from; 2+ that agree to be fact-grade"},
                    "description": {"type": "string"},
                },
                "required": ["name", "data", "sources"],
            },
            run=lambda name, data, sources, description="":
                datasets.save_dataset(name, data, sources, description),
        ),
        Tool(
            name="query_dataset",
            description="Look up cited data from a dataset — a specific key, or the whole "
                        "table. Always returns its source so you can trust and re-check it. "
                        "Reason FROM these values; feed them to a sim rather than recalling "
                        "numbers from memory.",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "key": {"type": "string", "description": "optional — a single entry to fetch"},
                },
                "required": ["name"],
            },
            run=lambda name, key=None: datasets.query_dataset(name, key),
        ),
        Tool(
            name="list_datasets",
            description="List your dataset shelf — name, what it holds, entry count, and its "
                        "cited source. Check here before gathering data you may already have.",
            parameters={"type": "object", "properties": {}},
            run=lambda: datasets.list_datasets(),
        ),
        Tool(
            name="data_ops",
            description="One-call versions of everyday data chores — use instead of "
                        "writing throwaway code or doing them in your head. Items go in "
                        "`text`, one per line. Operations: 'sort' (numeric if all numbers, "
                        "else alphabetical; flags: reverse, unique), 'dedupe' (order kept), "
                        "'count' (frequency table), 'diff' (`text` vs `second_text`), "
                        "'stats' (numbers -> count/sum/min/max/mean/median/stdev), "
                        "'case' (convert identifiers; style: snake/kebab/constant/camel/pascal), "
                        "'json_check' (validate + pretty-print, pinpoints errors), "
                        "'regex' (test `pattern` against `text`, shows matches). "
                        "For bespoke algorithms beyond these, use compute.",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {"type": "string",
                                  "enum": ["sort", "dedupe", "count", "diff",
                                           "stats", "case", "json_check", "regex"]},
                    "text": {"type": "string",
                             "description": "The items/text to operate on, one item per line"},
                    "second_text": {"type": "string",
                                    "description": "Second list, for diff"},
                    "pattern": {"type": "string",
                                "description": "Regular expression, for regex"},
                    "style": {"type": "string",
                              "enum": ["snake", "kebab", "constant", "camel", "pascal"]},
                    "reverse": {"type": "boolean"},
                    "unique": {"type": "boolean"},
                },
                "required": ["operation"],
            },
            run=data_ops,
        ),
        Tool(
            name="date_calc",
            description="Exact calendar math — never work out dates in your head. "
                        "Operations: 'today' (current date + weekday), "
                        "'weekday' (of `date`), 'diff' (`date` to `second_date`, "
                        "or to today if omitted), 'add' (`date` + `days`, negative ok). "
                        "Dates are ISO format: 2026-08-12.",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {"type": "string",
                                  "enum": ["today", "weekday", "diff", "add"]},
                    "date": {"type": "string"},
                    "second_date": {"type": "string"},
                    "days": {"type": "integer"},
                },
                "required": ["operation"],
            },
            run=date_calc,
        ),
        Tool(
            name="syntax_check",
            description="Verify that code PARSES before writing it to a file or "
                        "claiming it works — in any supported language, without "
                        "executing it. Pass the language and the code; returns "
                        "SYNTAX OK or the exact parse error with line numbers. "
                        "Supported here: python, javascript, bash, c, cpp, perl, "
                        "json, yaml. If a language isn't supported, it says so — "
                        "then be honest that the syntax is unverified.",
            parameters={
                "type": "object",
                "properties": {
                    "language": {"type": "string"},
                    "code": {"type": "string"},
                },
                "required": ["language", "code"],
            },
            run=syntax_check,
        ),
        Tool(
            name="text_stats",
            description="Exact prose statistics for a draft — use instead of eyeballing. "
                        "Pass the text in `text`; returns word/sentence/paragraph counts, "
                        "reading time, dialogue share, overused words (3+ repeats), and "
                        "the longest sentence. Read the file first if it's on disk.",
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string",
                             "description": "The draft text to analyze"},
                },
                "required": ["text"],
            },
            run=text_stats,
        ),
        Tool(
            name="ai_tells",
            description="Crossword-check for prose: measures how machine-flavored a "
                        "passage reads — stiff uniform sentence rhythm, cliche/AI-tell "
                        "phrases (delve, tapestry, moreover...), em-dash/semicolon "
                        "overuse, repeated openers. Returns a 0-100 machine-flavor score "
                        "and what to fix. Deterministic, not an online AI detector, but "
                        "measures the same fingerprints — so fixing what it flags is "
                        "editing for voice. Pass prose in `text`.",
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string",
                             "description": "The prose to check for machine flavor"},
                },
                "required": ["text"],
            },
            run=ai_tells,
        ),
        Tool(
            name="name_check",
            description="Canon guard: scan every .md/.txt file in the workspace for a "
                        "character/place name, reporting exact uses AND near-miss "
                        "spellings with file:line locations. A near-miss is usually a "
                        "typo about to fork into a second character — run this before "
                        "trusting that a name is spelled consistently.",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string",
                             "description": "The name to check, letters only"},
                    "max_distance": {"type": "integer",
                                     "description": "How many letters may differ (1-3, default 1)"},
                },
                "required": ["name"],
            },
            run=guard(lambda name, max_distance=1:
                      name_check(str(ws.root), name, max_distance)),
        ),
        Tool(
            name="format_code",
            description="Auto-format a file with prettier (js, jsx, ts, json, css, "
                        "html, md). Cleans up indentation, quotes, semicolons, and "
                        "spacing so the file is tidy and consistent. Rewrites the file "
                        "in place. Use after writing JS/CSS/JSON to make it neat.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            run=guard(lambda path: _format_code(ws, path)),
            needs_permission=True,
            summarize=lambda a: f"format {a.get('path', '')}",
        ),
        Tool(
            name="generate_image",
            description="Generate an image from a text prompt with a fast local model "
                        "(runs on CPU, so it never touches the GPU). Saves a PNG into the "
                        "workspace and returns the path — you can then look_at_image it to "
                        "see what you made. Quality is concept/mood/sketch tier and takes "
                        "~30s; good for visualizing an idea or a scene, not for final art. "
                        "Give a vivid, detailed prompt.",
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string",
                               "description": "Vivid description of the image to make"},
                    "filename": {"type": "string",
                                 "description": "Optional output name; defaults to a slug"},
                    "steps": {"type": "integer",
                              "description": "Denoising steps, 1-4 (default 3); more = slower"},
                    "seed": {"type": "integer",
                             "description": "Optional seed for a repeatable image"},
                },
                "required": ["prompt"],
            },
            run=guard(lambda prompt, filename="", steps=3, seed=None:
                      _generate_image(str(ws.root), prompt, filename, steps, seed)),
            needs_permission=True,
            summarize=lambda a: f"generate image: {a.get('prompt', '')[:60]}",
        ),
        Tool(
            name="recall",
            description="Search every past conversation you and the user have "
                        "had — your mid-term memory. Use it when the user "
                        "mentions something said or decided before that you "
                        "can't see in the current conversation, BEFORE "
                        "guessing or asking them to repeat it. Returns "
                        "verbatim snippets with date, folder, and who said "
                        "it. Query with a few concrete words (names, "
                        "things, decisions), not full sentences.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string",
                                         "description": "A few concrete words to look for"}},
                "required": ["query"],
            },
            run=lambda query: recall_search(query, workspace=str(ws.root)),
        ),
    ]
    # Everything outside the mode's core set stays reachable through the
    # index rather than riding along on every request. Registered here so
    # find_tools/load_tools can resolve a name to a real tool.
    from .modes import _CORE
    _INDEX_REGISTRY.clear()
    # The index must not contain the tools that OPERATE it: find_tools' own
    # description carries example phrases ("search my email", "design a
    # bracket"), so it matched those queries and returned itself instead of the
    # tool being asked for.
    _META = {"find_tools", "load_tools"}
    _INDEX_REGISTRY.update({t.name: t for t in _built
                            if t.name not in _CORE and t.name not in _META})
    return _built
