from __future__ import annotations

import importlib.util
from abc import ABC
from dataclasses import dataclass, field
from typing import Any

from brainscan.contracts import ModuleResult, Scan

CATEGORIES = {
    "guardrail": "Runtime guardrails: filter what goes in and comes out",
    "classifier": "Safety classifiers: label text as safe/unsafe",
    "evaluation": "Attack + evaluation: generate attacks, score behaviour",
    "whitebox": "White-box interpretability: read activations directly",
}


@dataclass(frozen=True)
class ModuleInfo:
    key: str            # registry key, used in CLI + file names
    name: str           # display name
    category: str       # one of CATEGORIES
    import_name: str    # what `available()` tries to import
    pip_extra: str      # `pip install "ai-brainscan[<extra>]"`
    license: str
    url: str
    what: str           # one line: what it tells us that nothing else does


@dataclass
class RunContext:
    """What a module gets handed. Black-box modules use prompt/output text;
    white-box modules use the model id (and load it) or an existing scan."""
    prompt: str
    model_id: str | None = None
    output_text: str | None = None
    scan: Scan | None = None
    options: dict[str, Any] = field(default_factory=dict)


class BrainModule(ABC):
    info: ModuleInfo

    def available(self) -> bool:
        return importlib.util.find_spec(self.info.import_name) is not None

    def install_hint(self) -> str:
        return f'pip install "ai-brainscan[{self.info.pip_extra}]"'

    def run(self, ctx: RunContext) -> ModuleResult:
        """Override in each adapter. Must return a ModuleResult and must never
        put generated harmful text into `details`: scores, labels, hashes only."""
        raise NotImplementedError(
            f"{self.info.name} adapter not written yet. See docs/MODULES.md for the "
            f"contract and {self.info.url} for the library."
        )
