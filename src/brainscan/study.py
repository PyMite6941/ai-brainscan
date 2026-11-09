"""Study helpers: run prompt sets, compare scans, find when a token 'appears'.

These are read-only views over saved Scans plus a prompt-set loader. They are
deliberately generic (cosine similarity, logit-lens lookups). The research
analyses (refusal direction, persona drift, patching) live in `analysis/`
and belong to Track A.

Prompt-set format (JSONL, one object per line, `#` lines ignored):

    {"prompt": "The capital of Texas is", "label": "direct", "group": "texas"}
    {"messages": [{"role": "user", "content": "..."}], "label": "multi-turn"}
    {"prompt": "...", "chat": false, "expect": "Austin"}

`label` and `group` are free text. `group` ties variants of one question
together so they can be compared. `expect` is an optional answer token that
`brainscan sweep` looks for in the logit lens. `chat` overrides --chat/--raw.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from brainscan.contracts import Scan

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


# ---------------------------------------------------------------- prompt sets

@dataclass
class PromptItem:
    prompt: str | list[dict[str, str]]
    label: str = ""
    group: str = ""
    chat: bool | None = None
    expect: str | None = None


def resolve_prompt_set(name: str) -> Path:
    """A path, or the name of a bundled set in prompts/ (with or without .jsonl)."""
    p = Path(name)
    if p.is_file():
        return p
    for cand in (PROMPTS_DIR / name, PROMPTS_DIR / f"{name}.jsonl"):
        if cand.is_file():
            return cand
    known = ", ".join(sorted(x.stem for x in PROMPTS_DIR.glob("*.jsonl"))) or "none"
    raise FileNotFoundError(f"no prompt set {name!r} (bundled sets: {known})")


def bundled_prompt_sets() -> list[Path]:
    return sorted(PROMPTS_DIR.glob("*.jsonl"))


def load_prompt_set(path: Path) -> list[PromptItem]:
    items = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{n}: bad JSON ({e})") from e
        prompt = obj.get("messages") or obj.get("prompt")
        if not prompt:
            raise ValueError(f"{path}:{n}: needs 'prompt' or 'messages'")
        items.append(PromptItem(prompt=prompt, label=str(obj.get("label", "")),
                                group=str(obj.get("group", "")), chat=obj.get("chat"),
                                expect=obj.get("expect")))
    return items


# ---------------------------------------------------------------- lens lookups

def _norm_tok(t: str) -> str:
    return t.strip().lower()


def first_layer(scan: Scan, token: str, position: int = -1, top_k: int = 1) -> int | None:
    """First layer whose logit-lens top-k at `position` contains `token`
    (whitespace/case-insensitive). None if it never does."""
    target = _norm_tok(token)
    T = len(scan.meta.tokens)
    pos = position if position >= 0 else T + position
    for layer, cell in enumerate(scan.lens):
        if any(_norm_tok(t) == target for t, _ in cell[pos][:top_k]):
            return layer
    return None


def settles_at(scan: Scan, token: str, position: int = -1, top_k: int = 1) -> int | None:
    """First layer from which `token` stays in the logit-lens top-k all the way to
    the output. More robust than `first_layer`, which early-layer noise can
    trigger (e.g. the lens on a chat template's final newline guesses digits at
    layer 1 for every prompt). None if it is not in the top-k at the last layer."""
    target = _norm_tok(token)
    T = len(scan.meta.tokens)
    pos = position if position >= 0 else T + position
    onset = None
    for layer in range(len(scan.lens) - 1, -1, -1):
        if any(_norm_tok(t) == target for t, _ in scan.lens[layer][pos][:top_k]):
            onset = layer
        else:
            break
    return onset


def token_prob_curve(scan: Scan, token: str, position: int = -1) -> list[float]:
    """Per-layer probability of `token` at `position` (0 where outside the saved top-k)."""
    target = _norm_tok(token)
    T = len(scan.meta.tokens)
    pos = position if position >= 0 else T + position
    return [max((p for t, p in cell[pos] if _norm_tok(t) == target), default=0.0)
            for cell in scan.lens]


# ---------------------------------------------------------------- comparison

@dataclass
class LayerDiff:
    layer: int
    cosine: float          # cos(resid_A, resid_B) at the chosen positions
    norm_a: float
    norm_b: float
    top_a: str
    top_b: str
    cosine_centred: float | None = None  # after subtracting a baseline mean (see similarity_matrix)


def diff(a: Scan, b: Scan, pos_a: int = -1, pos_b: int = -1,
         center: np.ndarray | None = None) -> list[LayerDiff]:
    """Layer-by-layer comparison of two scans of the same model at one position each.

    High cosine = the two inputs are represented alike at that depth. Watching
    where it drops (or recovers) shows where the model starts treating the
    inputs differently. `center` is an optional [L+1, d] baseline (mean over
    many scans of the same model) that removes the shared outlier dimensions."""
    if a.meta.d_model != b.meta.d_model or a.meta.n_layers != b.meta.n_layers:
        raise ValueError(f"scans come from different models ({a.meta.model_id} vs {b.meta.model_id})")
    ta, tb = len(a.meta.tokens), len(b.meta.tokens)
    pa = pos_a if pos_a >= 0 else ta + pos_a
    pb = pos_b if pos_b >= 0 else tb + pos_b
    ra = a.resid[:, pa].astype(np.float32)
    rb = b.resid[:, pb].astype(np.float32)
    na, nb = np.linalg.norm(ra, axis=-1), np.linalg.norm(rb, axis=-1)
    cos = (ra * rb).sum(-1) / (na * nb + 1e-8)
    cc = None
    if center is not None:
        ca, cb = ra - center, rb - center
        cc = (ca * cb).sum(-1) / (np.linalg.norm(ca, axis=-1) * np.linalg.norm(cb, axis=-1) + 1e-8)
    topa, topb = a.top_token(), b.top_token()
    return [LayerDiff(i, float(cos[i]), float(na[i]), float(nb[i]), topa[i][pa], topb[i][pb],
                      None if cc is None else float(cc[i]))
            for i in range(len(cos))]


def baseline_mean(scans: list[Scan], position: int = -1) -> np.ndarray | None:
    """[L+1, d] mean residual at `position` over scans (None if fewer than 3)."""
    if len(scans) < 3:
        return None
    return np.mean([s.resid[:, position if position >= 0 else len(s.meta.tokens) + position]
                    .astype(np.float32) for s in scans], axis=0)


def _vec(s: Scan, layer: int, position: int) -> np.ndarray:
    T = len(s.meta.tokens)
    return s.resid[layer, position if position >= 0 else T + position].astype(np.float32)


def similarity_matrix(scans: list[Scan], layer: int, position: int = -1,
                      center: np.ndarray | None = None) -> np.ndarray:
    """[N, N] cosine similarity of every pair of scans at one layer/position.

    Raw residuals share a few huge outlier dimensions (attention-sink /
    massive activations), which push every raw cosine towards 1. Pass
    `center` (e.g. `mean_vector(all_scans_in_the_sweep, layer)`) to subtract
    that shared component so the numbers reflect what differs between inputs."""
    m = np.stack([_vec(s, layer, position) for s in scans])
    if center is not None:
        m = m - center
    m = m / (np.linalg.norm(m, axis=-1, keepdims=True) + 1e-8)
    return m @ m.T


def mean_vector(scans: list[Scan], layer: int, position: int = -1) -> np.ndarray:
    """[d] mean residual at one layer/position over a set of scans (for centering)."""
    return np.mean([_vec(s, layer, position) for s in scans], axis=0)
