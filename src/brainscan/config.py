"""Paths and settings. Everything reads env vars so the npm launcher, the
install scripts and a plain `pip install` all behave the same way."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# .env in the current directory wins, then ~/.brainscan/.env (npm installs).
load_dotenv(Path.cwd() / ".env")
load_dotenv(Path.home() / ".brainscan" / ".env")

DEFAULT_MODEL = "unsloth/gemma-3-1b-it"


def default_model() -> str:
    return os.environ.get("BRAINSCAN_MODEL") or DEFAULT_MODEL


def artifacts_dir() -> Path:
    p = Path(os.environ.get("BRAINSCAN_ARTIFACTS") or Path.cwd() / "artifacts")
    p.mkdir(parents=True, exist_ok=True)
    return p


def hf_home() -> Path:
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"])
    return Path.home() / ".cache" / "huggingface"


def synced_folder(path: Path) -> str | None:
    """Return the sync provider's name if `path` sits inside a synced folder.
    Multi-GB weights inside OneDrive/Dropbox/iCloud get uploaded and corrupt."""
    s = str(path).lower()
    for name in ("onedrive", "dropbox", "icloud", "google drive"):
        if name in s:
            return name
    return None
