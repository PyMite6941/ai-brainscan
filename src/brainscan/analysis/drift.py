"""Persona drift across a conversation. View 03.

Method: the Assistant Axis (Lu et al., arXiv 2601.10387) and Persona Vectors
(arXiv 2507.21509). NOTE: github.com/safety-research/assistant-axis has NO
licence file. Read it, reimplement from the paper, do not copy the code.

Contract
--------
persona_trace(model_id, turns, direction) -> list[float]
    turns: a multi-turn conversation (list of {"role", "content"}).
    Capture the conversation once per prefix (turn 1, turns 1-2, ...) and
    project the last token's residual at a chosen layer onto `direction`.
    One number per turn: how "assistant-like" the model still is.

Prediction to test: in a successful multi-turn jailbreak the persona
component decays BEFORE the refusal direction collapses.
"""

from __future__ import annotations

import numpy as np


def persona_trace(model_id: str, turns: list[dict[str, str]], direction: np.ndarray, layer: int) -> list[float]:
    raise NotImplementedError("Track A: project each turn-prefix's last token, see module docstring")
