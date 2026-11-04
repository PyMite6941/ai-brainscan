"""Mixture of modules: third-party tools behind one interface.

Four categories, two vantage points:

  outside the model (black-box: inputs, outputs, behaviour)
    guardrail   runtime filters on the way in/out   NeMo Guardrails, Guardrails AI, LLM Guard
    classifier  "is this text unsafe?" models       Llama Guard
    evaluation  attack generators + scorers         garak, PyRIT, Inspect

  inside the model (white-box: activations, hidden states)
    whitebox    hooks into the forward pass          brainscan native, TransformerLens, nnsight

Every module is optional. `available()` only checks the package imports, so
listing modules never loads a model. To add one: subclass BrainModule in the
right category file and add it to REGISTRY below. See docs/MODULES.md.
"""

from __future__ import annotations

from brainscan.modules.base import CATEGORIES, BrainModule, ModuleInfo, RunContext
from brainscan.modules.blackbox import (
    Garak,
    GuardrailsAI,
    Inspect,
    LlamaGuard,
    LLMGuard,
    NeMoGuardrails,
    PyRIT,
)
from brainscan.modules.whitebox import NativeCapture, NNSight, TransformerLens

REGISTRY: dict[str, BrainModule] = {
    m.info.key: m
    for m in (
        NativeCapture(),
        TransformerLens(),
        NNSight(),
        LLMGuard(),
        NeMoGuardrails(),
        GuardrailsAI(),
        LlamaGuard(),
        Garak(),
        PyRIT(),
        Inspect(),
    )
}


def get(key: str) -> BrainModule:
    try:
        return REGISTRY[key]
    except KeyError:
        raise KeyError(f"unknown module {key!r}; known: {', '.join(REGISTRY)}") from None


def by_category(category: str) -> list[BrainModule]:
    return [m for m in REGISTRY.values() if m.info.category == category]


__all__ = ["CATEGORIES", "REGISTRY", "BrainModule", "ModuleInfo", "RunContext", "by_category", "get"]
