"""Black-box modules: they only see text in and text out.

They matter because they give the behavioural ground truth the white-box
view is explained against: "the classifier says this complied" is the label;
"the refusal signal collapsed at layer 16" is the explanation.
"""

from __future__ import annotations

from brainscan.modules.base import BrainModule, ModuleInfo

# ---------------------------------------------------------------- guardrails


class LLMGuard(BrainModule):
    info = ModuleInfo(
        key="llm_guard",
        name="LLM Guard",
        category="guardrail",
        import_name="llm_guard",
        pip_extra="guardrails",
        license="MIT",
        url="https://github.com/protectai/llm-guard",
        what="Input/output scanners (prompt injection, toxicity, secrets) with scores",
    )


class NeMoGuardrails(BrainModule):
    info = ModuleInfo(
        key="nemo_guardrails",
        name="NeMo Guardrails",
        category="guardrail",
        import_name="nemoguardrails",
        pip_extra="nemo",
        license="Apache-2.0",
        url="https://github.com/NVIDIA/NeMo-Guardrails",
        what="Programmable dialogue rails (Colang); jailbreak + topic checks",
    )


class GuardrailsAI(BrainModule):
    info = ModuleInfo(
        key="guardrails_ai",
        name="Guardrails AI",
        category="guardrail",
        import_name="guardrails",
        pip_extra="guardrails-ai",
        license="Apache-2.0",
        url="https://github.com/guardrails-ai/guardrails",
        what="Validators from the Guardrails Hub run over outputs",
    )


# ---------------------------------------------------------------- classifiers


class LlamaGuard(BrainModule):
    info = ModuleInfo(
        key="llama_guard",
        name="Llama Guard",
        category="classifier",
        import_name="transformers",
        pip_extra="default",
        # the code path is ours; the WEIGHTS are under the Llama licence + AUP
        # and are gated on HF. never vendor them.
        license="Llama Community Licence (weights, gated)",
        url="https://huggingface.co/meta-llama/Llama-Guard-3-1B",
        what="Safe/unsafe + hazard category for a prompt or a response",
    )


# ---------------------------------------------------------------- evaluation


class Garak(BrainModule):
    info = ModuleInfo(
        key="garak",
        name="garak",
        category="evaluation",
        import_name="garak",
        pip_extra="garak",
        license="Apache-2.0",
        url="https://github.com/NVIDIA/garak",
        what="Probe library: fires known attack families, reports which got through",
    )


class PyRIT(BrainModule):
    info = ModuleInfo(
        key="pyrit",
        name="PyRIT",
        category="evaluation",
        import_name="pyrit",
        pip_extra="pyrit",
        license="MIT",
        url="https://github.com/Azure/PyRIT",
        what="Multi-turn attack orchestration (e.g. Crescendo); feeds persona-drift scans",
    )


class Inspect(BrainModule):
    info = ModuleInfo(
        key="inspect",
        name="Inspect",
        category="evaluation",
        import_name="inspect_ai",
        pip_extra="eval",
        license="MIT",
        url="https://github.com/UKGovernmentBEIS/inspect_ai",
        what="Reproducible eval harness: datasets, solvers, scorers, logs",
    )
