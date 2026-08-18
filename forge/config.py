"""
Config: one YAML file the CLI reads and the dashboard writes.

That shared file is the whole trick behind "change the model in the browser,
the terminal picks it up." No sockets, no message bus — the CLI re-reads
config at the start of every turn, so a change saved in the dashboard takes
effect on your next message.

Lives at ~/.forge/config.yaml so it's per-user, not per-project.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(os.environ.get("FORGE_HOME", Path.home() / ".forge"))
CONFIG_PATH = CONFIG_DIR / "config.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    # Which entry in `models` the agent talks to right now. The dashboard
    # changes this; the CLI honors it on the next turn.
    "active_model": "claude",

    # Kid mode: chat only. Locks every settings route in the dashboard,
    # confines run_command to the workspace, and pins new chats to
    # ~/Playground. Flip it here in the file, at the computer — on purpose,
    # nothing reachable from a tablet can change it.
    "kid_mode": False,

    # Named model configs. Add as many as you like — local ones get a
    # base_url pointing at whatever's serving them.
    "models": {
        "claude": {
            "provider": "anthropic",
            "model": "claude-opus-5",
            "max_tokens": 8000,
            # Leave api_key empty to use the ANTHROPIC_API_KEY environment
            # variable instead of storing a secret in a plain text file.
            "api_key": "",
        },
        "fable": {
            "provider": "anthropic",
            "model": "claude-fable-5",
            "max_tokens": 8000,
            "api_key": "",
        },
        "local": {
            "provider": "openai-compat",
            "model": "qwen2.5-coder:14b",
            "base_url": "http://localhost:11434/v1",
            "max_tokens": 4096,
            "api_key": "",
        },
        # A small CPU-only model for short-prompt side jobs (quick
        # classifications; memory summaries when the main model is a paid
        # API — set agent.summarizer_model to its name to enable that).
        # Measured honestly: CPU prompt reading is ~24 tok/s, so anything
        # long-prompt (full agent work, big summaries) belongs on the GPU
        # or API model instead.
        "little": {
            "provider": "openai-compat",
            "model": "qwen2.5-3b",
            "base_url": "http://127.0.0.1:8083/v1",
            "max_tokens": 2048,
            "api_key": "",
        },
    },

    "agent": {
        # Hard stop on tool-call rounds per message, so a confused model
        # can't spin forever burning tokens.
        "max_steps": 80,
        # "ask"  — prompt before every world-changing tool (default, safest)
        # "auto" — run everything without asking (fast, for throwaway dirs)
        # "deny" — read-only; refuse all writes and commands
        "permission_mode": "ask",
        # The sealed reviewer: judges final answers against the evidence
        # before "done". True/False; superego_model names which model
        # judges (empty = the active model, with the sealed prompt).
        "superego": True,
        "superego_model": "",
    },

    # host 127.0.0.1 keeps this machine-only. Set 0.0.0.0 to reach it from a
    # phone or tablet on the same network — but then set a token too, because
    # this thing runs shell commands and anyone on the LAN could reach it.
    "server": {"host": "127.0.0.1", "port": 8770, "token": ""},

    # Who she belongs to, and a challenge phrase for when a session smells off.
    # Off by default — arm it when you're ready. The phrase is stored ONLY as a
    # salted hash (see identity.py), never in plaintext here, never in her prompt.
    "identity": {"enabled": False, "owner_name": "", "phrase_hash": "", "phrase_salt": ""},

    # Eyes and ears. Every one of these is optional — Forge runs fine
    # without any of them, and tools only appear when they actually work.
    "media": {
        # Primary vision is Merge's own sighted brain (8085); the shared CPU 7B
        # (8090) is the fallback when she's not the active model. Keep these in
        # step with the MediaConfig defaults in media.py.
        "vision_url": "http://127.0.0.1:8085/v1",
        "vision_model": "qwen3.6-27b",
        "vision_fallback_url": "http://127.0.0.1:8090/v1",
        "vision_fallback_model": "qwen2.5-vl",
        "whisper_bin": "~/whisper.cpp/build/bin/whisper-cli",
        "whisper_model": "~/whisper.cpp/models/ggml-small.en.bin",
        "tts_command": "spd-say",
        "piper_voice": "~/forge/models/voices/en_US-amy-medium.onnx",
    },
}


def _deep_merge(base: dict, over: dict) -> dict:
    """Fill in anything the user's config file leaves out."""
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(base[k], v) if (k in base and isinstance(base[k], dict)
                                             and isinstance(v, dict)) else v
    return out


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        save_config(DEFAULT_CONFIG)
        return dict(DEFAULT_CONFIG)
    try:
        loaded = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise yaml.YAMLError(f"expected a mapping, got {type(loaded).__name__}")
    except yaml.YAMLError as e:
        # A hand-edit gone wrong shouldn't brick every Forge tool at once.
        # Keep the broken file for repair, start fresh, and say so loudly.
        import sys
        broken = CONFIG_PATH.with_name("config.yaml.broken")
        try:
            CONFIG_PATH.replace(broken)
        except OSError:
            broken = None
        save_config(DEFAULT_CONFIG)
        print(f"WARNING: {CONFIG_PATH} was not valid YAML ({e}).\n"
              + (f"Your old file is saved at {broken} — " if broken else "")
              + "Forge regenerated a default config so it can keep working. "
                "Your model list will need re-adding (dashboard → Add a model).",
              file=sys.stderr)
        return dict(DEFAULT_CONFIG)
    merged = _deep_merge(DEFAULT_CONFIG, loaded)
    # `models` is the user's list, not a set of defaults to top up. Merging it
    # meant a deleted model came straight back on the next load — and worse,
    # a stale default could be suggested as a fix when it points at something
    # that was never running here.
    if isinstance(loaded.get("models"), dict):
        merged["models"] = loaded["models"]
    return merged


def save_config(cfg: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def active_model_config(cfg: dict) -> dict:
    """The config block for whichever model is currently selected."""
    name = cfg.get("active_model", "claude")
    models = cfg.get("models", {})
    if name not in models:
        raise SystemExit(
            f"active_model is {name!r} but there's no such entry under models. "
            f"Available: {', '.join(models) or '(none)'}"
        )
    return models[name]

# --- pipelines ------------------------------------------------------

PIPELINES_PATH = CONFIG_DIR / "pipelines.yaml"


def load_pipelines() -> dict:
    """
    Orchestration recipes, from the user's copy in ~/.forge.

    On first run the shipped examples are copied there, so editing them is
    safe — a package upgrade can't overwrite the user's own recipes.
    """
    if not PIPELINES_PATH.exists():
        shipped = Path(__file__).parent / "pipelines.yaml"
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        if shipped.exists():
            PIPELINES_PATH.write_text(shipped.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            PIPELINES_PATH.write_text("pipelines: {}\n", encoding="utf-8")
    try:
        data = yaml.safe_load(PIPELINES_PATH.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise SystemExit(f"pipelines.yaml is not valid YAML: {PIPELINES_PATH}\n{e}")
    return data.get("pipelines", {}) or {}


def save_pipelines(pipelines: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    PIPELINES_PATH.write_text(
        yaml.safe_dump({"pipelines": pipelines}, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
