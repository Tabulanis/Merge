"""Syntax checking across languages — deterministic "does this even parse?"

One call, any language with a checker on the machine. This is the coding
reflex: verify code parses BEFORE claiming it works or writing it to disk,
instead of eyeballing syntax from memory. Checkers only parse — nothing is
executed.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

_TIMEOUT = 20

# language -> (file suffix, command builder). Commands parse only, never run.
_CHECKERS = {
    "python":     (".py",   lambda p: ["python3", "-m", "py_compile", p]),
    "javascript": (".js",   lambda p: ["node", "--check", p]),
    "bash":       (".sh",   lambda p: ["bash", "-n", p]),
    "c":          (".c",    lambda p: ["gcc", "-fsyntax-only", "-Wall", p]),
    "cpp":        (".cpp",  lambda p: ["g++", "-fsyntax-only", "-Wall", p]),
    "perl":       (".pl",   lambda p: ["perl", "-c", p]),
}
_ALIASES = {"py": "python", "js": "javascript", "node": "javascript",
            "sh": "bash", "shell": "bash", "c++": "cpp", "cxx": "cpp"}


def syntax_check(language: str, code: str) -> str:
    lang = _ALIASES.get(language.strip().lower(), language.strip().lower())

    # json/yaml parse in-process — no external binary needed
    if lang == "json":
        try:
            json.loads(code)
            return "SYNTAX OK (json)"
        except json.JSONDecodeError as e:
            return f"SYNTAX ERROR (json): {e.msg} at line {e.lineno}, column {e.colno}"
    if lang == "yaml":
        try:
            import yaml  # optional dependency; report honestly if absent
        except ImportError:
            return "No yaml checker available on this machine (PyYAML not installed)."
        try:
            yaml.safe_load(code)
            return "SYNTAX OK (yaml)"
        except yaml.YAMLError as e:
            return f"SYNTAX ERROR (yaml): {str(e)[:300]}"

    if lang not in _CHECKERS:
        known = ", ".join(sorted(list(_CHECKERS) + ["json", "yaml"]))
        return (f"No checker for {language!r}. Available here: {known}. "
                f"Say so honestly rather than eyeballing the syntax.")

    suffix, build = _CHECKERS[lang]
    if not shutil.which(build("x")[0]):
        return (f"The {lang} checker ({build('x')[0]}) is not installed on "
                f"this machine — cannot verify, and say so rather than guess.")

    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False,
                                     encoding="utf-8") as tf:
        tf.write(code)
        tmp = tf.name
    try:
        proc = subprocess.run(build(tmp), capture_output=True, text=True,
                              timeout=_TIMEOUT)
        out = ((proc.stdout or "") + (proc.stderr or "")).replace(tmp, "<code>")
        if proc.returncode == 0:
            # perl/gcc may still print warnings on success — pass them along
            extra = out.strip()
            return f"SYNTAX OK ({lang})" + (f"\nwarnings:\n{extra[:1500]}" if extra
                                            and "syntax OK" not in extra else "")
        return f"SYNTAX ERROR ({lang}):\n{out.strip()[:2000]}"
    except subprocess.TimeoutExpired:
        return f"Checker timed out after {_TIMEOUT}s — file may be enormous."
    finally:
        Path(tmp).unlink(missing_ok=True)
        # py_compile drops a __pycache__ next to the temp file
        pyc = Path(tmp).parent / "__pycache__"
        if lang == "python" and pyc.is_dir():
            shutil.rmtree(pyc, ignore_errors=True)
