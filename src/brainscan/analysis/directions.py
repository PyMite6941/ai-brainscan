"""Refusal direction. Method: Arditi et al., github.com/andyrdt/refusal_direction

Contract
--------
refusal_direction(harmful, harmless, position=-1) -> np.ndarray [L+1, d]
    harmful/harmless: lists of Scans of the SAME model with the chat template on.
    For each layer, mean residual at `position` over harmful minus the mean
    over harmless. Return one direction per layer (unit-normalise or not, but
    say which). The paper then picks the single best layer by ablation.

save_direction(name, direction) writes artifacts/directions/<name>.npy, which
Scan.project() and every UI pick up automatically.

Things that will bite:
  * position -1 is the last prompt token AFTER the generation prompt
    ("<start_of_turn>model\\n"); that is where the decision is made.
  * use matched pairs (JailbreakBench harmful vs harmless) or you will find
    a "topic" direction instead of a refusal one.
  * include XSTest-style safe-but-scary prompts as a control.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from brainscan.config import artifacts_dir
from brainscan.contracts import Scan


def refusal_direction(harmful: list[Scan], harmless: list[Scan], position: int = -1) -> np.ndarray:
    raise NotImplementedError("Track A: difference in means per layer, see module docstring")


def directions_dir() -> Path:
    p = artifacts_dir() / "directions"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_direction(name: str, direction: np.ndarray) -> Path:
    p = directions_dir() / f"{name}.npy"
    np.save(p, direction.astype(np.float32))
    return p


def load_directions() -> dict[str, np.ndarray]:
    return {p.stem: np.load(p) for p in sorted(directions_dir().glob("*.npy"))}
