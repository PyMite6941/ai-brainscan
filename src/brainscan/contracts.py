"""The on-disk contract. Capture writes a Scan; every UI only reads one.

This file is the seam between slow work (running the model) and fast work
(drawing it). Teams work in parallel by agreeing on this, so change it with
care and bump SCHEMA_VERSION when you do.

A scan is a directory:

    artifacts/<scan_id>/
        meta.json        ScanMeta below
        resid.npy        float16 [n_layers + 1, n_tokens, d_model]
                         residual stream after each block; index 0 = embeddings
        lens.json        logit lens: [n_layers + 1][n_tokens][top_k] of [token, prob]
        attn.npy         optional float16 [n_layers, n_heads, n_tokens, n_tokens]
        modules/<key>.json   ModuleResult from any module that ran on this scan
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class ScanMeta:
    scan_id: str
    model_id: str
    label: str                    # free text, e.g. "refused", "jailbreak:dan", "control"
    prompt: str                   # the input. artifacts/ is gitignored.
    prompt_sha256: str
    chat_template: bool           # was the chat template applied (it should be, on -it models)
    tokens: list[str]             # decoded token strings, len == n_tokens
    n_layers: int
    d_model: int
    created: str                  # ISO-8601
    brainscan_version: str
    has_attn: bool = False
    # generated output: hash + verdict always, text only if the user opted in.
    # never commit elicited harmful text. see README "Ground rules".
    output_sha256: str | None = None
    output_text: str | None = None
    verdict: str | None = None    # "refused" | "complied" | None (not judged)
    extra: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION


@dataclass
class ModuleResult:
    module: str                   # registry key, e.g. "llm_guard"
    category: str                 # guardrail | classifier | evaluation | whitebox
    verdict: str | None = None    # "pass" | "flag" | "error" | None
    score: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Scan:
    meta: ScanMeta
    resid: np.ndarray                         # [L+1, T, d]
    lens: list[list[list[tuple[str, float]]]]  # [L+1][T][k]
    attn: np.ndarray | None = None            # [L, H, T, T]
    path: Path | None = None

    # ---- derived views (cheap, computed on read) ----
    def resid_norm(self) -> np.ndarray:
        """[L+1, T] L2 norm of the residual stream. The plainest 'EEG'."""
        return np.linalg.norm(self.resid.astype(np.float32), axis=-1)

    def project(self, direction: np.ndarray) -> np.ndarray:
        """[L+1, T] projection onto a unit direction (e.g. a refusal direction).
        `direction` is [d] for one shared direction or [L+1, d] per layer."""
        d = direction.astype(np.float32)
        d = d / (np.linalg.norm(d, axis=-1, keepdims=True) + 1e-8)
        r = self.resid.astype(np.float32)
        if d.ndim == 1:
            return r @ d
        return np.einsum("ltd,ld->lt", r, d)

    def top_token(self) -> list[list[str]]:
        """[L+1][T] the logit lens's single best guess at each layer/position."""
        return [[pos[0][0] if pos else "" for pos in layer] for layer in self.lens]

    # ---- io ----
    def save(self, root: Path) -> Path:
        out = root / self.meta.scan_id
        (out / "modules").mkdir(parents=True, exist_ok=True)
        self.meta.has_attn = self.attn is not None
        (out / "meta.json").write_text(json.dumps(asdict(self.meta), indent=2), encoding="utf-8")
        np.save(out / "resid.npy", self.resid.astype(np.float16))
        (out / "lens.json").write_text(json.dumps(self.lens), encoding="utf-8")
        if self.attn is not None:
            np.save(out / "attn.npy", self.attn.astype(np.float16))
        self.path = out
        return out

    @classmethod
    def load(cls, path: Path) -> Scan:
        meta = ScanMeta(**json.loads((path / "meta.json").read_text(encoding="utf-8")))
        resid = np.load(path / "resid.npy")
        lens = json.loads((path / "lens.json").read_text(encoding="utf-8"))
        attn = np.load(path / "attn.npy") if (path / "attn.npy").exists() else None
        return cls(meta=meta, resid=resid, lens=lens, attn=attn, path=path)

    def save_module_result(self, result: ModuleResult) -> Path:
        assert self.path is not None, "save the scan before attaching module results"
        p = self.path / "modules" / f"{result.module}.json"
        p.parent.mkdir(exist_ok=True)
        p.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
        return p

    def module_results(self) -> list[ModuleResult]:
        if self.path is None or not (self.path / "modules").exists():
            return []
        return [
            ModuleResult(**json.loads(p.read_text(encoding="utf-8")))
            for p in sorted((self.path / "modules").glob("*.json"))
        ]


def list_scans(root: Path) -> list[ScanMeta]:
    """All scans under root, newest first. Reads meta only, never the arrays."""
    metas = []
    for p in root.glob("*/meta.json"):
        try:
            metas.append(ScanMeta(**json.loads(p.read_text(encoding="utf-8"))))
        except (json.JSONDecodeError, TypeError):
            continue
    return sorted(metas, key=lambda m: m.created, reverse=True)
