"""Black-box modules: they only see text in and text out.

They matter because they give the behavioural ground truth the white-box
view is explained against: "the classifier says this complied" is the label;
"the refusal signal collapsed at layer 16" is the explanation.
"""

from __future__ import annotations

from brainscan.contracts import ModuleResult
from brainscan.modules.base import BrainModule, ModuleInfo, RunContext

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

    _INPUT_SCANNERS = ("PromptInjection", "Toxicity", "TokenLimit")
    _OUTPUT_SCANNERS = ("NoRefusal", "Relevance", "Sensitive")

    def run(self, ctx: RunContext) -> ModuleResult:
        from llm_guard import scan_output, scan_prompt
        from llm_guard.input_scanners import (
            PromptInjection,
            TokenLimit,
            Toxicity,
        )
        from llm_guard.output_scanners import NoRefusal, Relevance, Sensitive

        input_scanners = [PromptInjection(), Toxicity(), TokenLimit()]
        _, results_valid, results_score = scan_prompt(input_scanners, ctx.prompt)
        input_valid = all(results_valid.values())
        input_details = {k: bool(v) for k, v in results_valid.items()}
        input_scores = {k: float(v) for k, v in results_score.items()}

        output_valid = True
        output_details = {}
        output_scores = {}
        if ctx.output_text:
            output_scanners = [NoRefusal(), Relevance(), Sensitive()]
            _, out_valid, out_score = scan_output(
                output_scanners, ctx.prompt, ctx.output_text
            )
            output_valid = all(out_valid.values())
            output_details = {k: bool(v) for k, v in out_valid.items()}
            output_scores = {k: float(v) for k, v in out_score.items()}

        verdict = "pass" if input_valid and output_valid else "flag"
        return ModuleResult(
            module=self.info.key,
            category=self.info.category,
            verdict=verdict,
            score=1.0 if verdict == "pass" else 0.0,
            details={
                "input_valid": input_valid,
                "input_scanners": input_details,
                "input_scores": input_scores,
                "output_valid": output_valid,
                "output_scanners": output_details,
                "output_scores": output_scores,
            },
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

    def run(self, ctx: RunContext) -> ModuleResult:
        raise NotImplementedError(
            "NeMo Guardrails adapter: needs a Colang config pointing at the "
            "local model. See docs/MODULES.md."
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

    def run(self, ctx: RunContext) -> ModuleResult:
        raise NotImplementedError(
            "Guardrails AI adapter: install a validator from the Guardrails Hub "
            "and run it over ctx.output_text. See docs/MODULES.md."
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

    DEFAULT_MODEL = "meta-llama/Llama-Guard-3-1B"

    def run(self, ctx: RunContext) -> ModuleResult:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = ctx.options.get("model_id", self.DEFAULT_MODEL)
        text = ctx.output_text or ctx.prompt
        if not text:
            return ModuleResult(
                module=self.info.key,
                category=self.info.category,
                verdict="error",
                score=None,
                details={"error": "no text to classify"},
            )

        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.eval()

        conversation = [
            {"role": "user", "content": [{"type": "text", "text": text}]}
        ]
        input_ids = tok.apply_chat_template(
            conversation, return_tensors="pt"
        )
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=20, pad_token_id=tok.pad_token_id or 0
            )
        generated = output[:, prompt_len:]
        result = tok.decode(generated[0], skip_special_tokens=True).strip()

        lines = [ln.strip() for ln in result.split("\n") if ln.strip()]
        label = lines[0].lower() if lines else "unknown"
        categories = [ln for ln in lines[1:] if ln]
        verdict = "pass" if label == "safe" else "flag"

        return ModuleResult(
            module=self.info.key,
            category=self.info.category,
            verdict=verdict,
            score=1.0 if verdict == "pass" else 0.0,
            details={
                "label": label,
                "categories": categories,
                "model": model_id,
                "raw": result,
            },
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

    def run(self, ctx: RunContext) -> ModuleResult:
        raise NotImplementedError(
            "garak adapter: run garak as a subprocess against ctx.model_id "
            "and parse the probe/detector results. See reference.garak.ai."
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

    def run(self, ctx: RunContext) -> ModuleResult:
        raise NotImplementedError(
            "PyRIT adapter: orchestrate a multi-turn attack (e.g. Crescendo) "
            "against ctx.model_id and score the outcome. See Azure/PyRIT docs."
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

    def run(self, ctx: RunContext) -> ModuleResult:
        raise NotImplementedError(
            "Inspect adapter: wrap ctx.prompt as an inspect_ai Task with a "
            "generate() solver and a safety scorer. See inspect.aisi.org.uk."
        )
