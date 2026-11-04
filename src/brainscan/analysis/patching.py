"""The causal test. "Correlation isn't enough": a direction only counts once
removing it changes what the model does.

Contract
--------
generate_with_ablation(model_id, prompt, direction, layers=None, max_new_tokens=64) -> str
    Register forward hooks that subtract the component along `direction`
    (x - (x . d_hat) d_hat) from the residual stream at `layers` (default:
    all), generate, remove the hooks. Return the text so the caller can hash
    and judge it. Never write the text to the repo.

Ablation stays a RUNTIME hook. Never save or publish an edited checkpoint
(Gemma terms, and the project ground rules).
"""

from __future__ import annotations

import numpy as np


def generate_with_ablation(
    model_id: str,
    prompt: str,
    direction: np.ndarray,
    layers: list[int] | None = None,
    max_new_tokens: int = 64,
) -> str:
    raise NotImplementedError("Track A: forward hooks that project out the direction")
