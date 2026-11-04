# Modules

Every third-party tool sits behind `brainscan.modules.base.BrainModule`:

```python
class MyTool(BrainModule):
    info = ModuleInfo(key="my_tool", name="My Tool", category="classifier",
                      import_name="my_tool", pip_extra="my-tool", license="MIT",
                      url="https://...", what="one line: what it tells us")

    def run(self, ctx: RunContext) -> ModuleResult:
        ...
        return ModuleResult(module="my_tool", category="classifier",
                            verdict="flag", score=0.93, details={"hazard": "S2"})
```

Then add it to `REGISTRY` in `src/brainscan/modules/__init__.py` and an extra in `pyproject.toml`.

## Rules for adapters

- `available()` must stay cheap: it only checks the import. Never load a model at import time.
- `details` holds scores, labels, categories and hashes. **Never generated harmful text.**
- Attach results to a scan with `scan.save_module_result(result)`; the UIs show them automatically.
- Heavy libraries pin conflicting dependencies (garak, nemoguardrails, pyrit especially). Install one per venv
  if pip cannot resolve them together; they only need to agree on the `ModuleResult` JSON.

## Status

| key | category | adapter | notes |
|---|---|---|---|
| `native` | whitebox | done | the capture engine |
| `transformer_lens` | whitebox | todo | activation patching; check Gemma 3 support in your TL version |
| `nnsight` | whitebox | todo | tracing/editing; can run big models remotely on NDIF |
| `llm_guard` | guardrail | todo | input + output scanners |
| `nemo_guardrails` | guardrail | todo | needs an LLM config; point it at the local model |
| `guardrails_ai` | guardrail | todo | hub validators need `guardrails hub install` |
| `llama_guard` | classifier | todo | `meta-llama/Llama-Guard-3-1B`, gated; replaces the keyword verdict |
| `garak` | evaluation | todo | run probes against the local HF model, import hit/miss per probe |
| `pyrit` | evaluation | todo | Crescendo multi-turn conversations become input for persona-drift scans |
| `inspect` | evaluation | todo | wrap sweeps as reproducible Inspect tasks |
