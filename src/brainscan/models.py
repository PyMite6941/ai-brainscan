"""Find models already on this machine and say which ones we can open up.

Only Hugging Face safetensors/bin checkpoints expose hidden states. Ollama /
llama.cpp (GGUF) only hand back tokens, so there is nothing to scan; we list
them with a warning rather than silently hiding them.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import scan_cache_dir


@dataclass
class LocalModel:
    repo_id: str
    size_gb: float
    has_weights: bool
    architecture: str | None
    n_layers: int | None
    d_model: int | None
    scannable: bool
    note: str = ""


def _read_config(snapshot: Path) -> dict:
    cfg = snapshot / "config.json"
    if not cfg.exists():
        return {}
    try:
        return json.loads(cfg.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def text_config(cfg: dict) -> dict:
    """Multimodal configs (gemma-3-4b+) nest the language model under text_config."""
    return cfg.get("text_config") or cfg


def local_models() -> list[LocalModel]:
    try:
        cache = scan_cache_dir()
    except Exception:  # noqa: BLE001 - no cache yet on first run
        return []
    out: list[LocalModel] = []
    for repo in cache.repos:
        if repo.repo_type != "model":
            continue
        files: set[str] = set()
        snapshot = None
        for rev in repo.revisions:
            files |= {f.file_name for f in rev.files}
            snapshot = rev.snapshot_path
        has_weights = any(f.endswith((".safetensors", ".bin")) for f in files)
        cfg = _read_config(snapshot) if snapshot else {}
        tcfg = text_config(cfg)
        arch = (cfg.get("architectures") or [None])[0]
        is_lm = bool(arch) and ("CausalLM" in arch or "ConditionalGeneration" in arch or "LMHead" in arch)
        note = ""
        if not has_weights:
            note = "metadata only, weights not downloaded (brainscan download)"
        elif not is_lm:
            note = "not a causal language model"
        out.append(
            LocalModel(
                repo_id=repo.repo_id,
                size_gb=round(repo.size_on_disk / 1e9, 2),
                has_weights=has_weights,
                architecture=arch,
                n_layers=tcfg.get("num_hidden_layers") or tcfg.get("n_layer"),
                d_model=tcfg.get("hidden_size") or tcfg.get("n_embd"),
                scannable=has_weights and is_lm,
                note=note,
            )
        )
    return sorted(out, key=lambda m: (not m.scannable, m.repo_id.lower()))


def ollama_models() -> list[str]:
    """Names of Ollama models, if Ollama is installed. Reported, never offered."""
    root = Path(os.environ.get("OLLAMA_MODELS") or Path.home() / ".ollama" / "models")
    lib = root / "manifests" / "registry.ollama.ai" / "library"
    if not lib.exists():
        return []
    return sorted(f"{m.name}:{t.name}" for m in lib.iterdir() if m.is_dir() for t in m.iterdir())


def download(repo_id: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id,
            allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model", "*.txt"],
        )
    )
