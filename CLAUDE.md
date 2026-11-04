# CLAUDE.md: ai-brainscan

Interpretability instrument (team research project). Vision doc: claude.ai artifact "Tracing the Refusal"
(75b00274-2a61-4677-b09c-6ad747b5004d). Successor to `../AI Guardrail Research/` (its `MEMORY.md` holds the
research decisions: Gemma 3 1B, never Ollama, Gradio for teams, licence notes).

## Boundaries
- `src/brainscan/analysis/` (refusal direction, drift, patching) is **Track A research code: the team/Matt writes
  it.** Leave the `NotImplementedError` stubs alone unless Matt asks for an implementation.
- Claude builds: UIs (`ui/`), module adapters (`modules/`), scripts, npm launcher, packaging, docs, tests.
- Lower-level work goes to Hermes (`hermes --in <repo> --yolo -z "Read the task brief at <path>..."`); Claude
  writes the brief, reviews, runs tests, commits.

## Architecture
capture (`capture.py`, slow) writes `artifacts/<scan_id>/` per `contracts.py`; every UI only reads it.
Change `contracts.py` => bump `SCHEMA_VERSION`.

## Commands
```
.venv\Scripts\python.exe -m pytest -q          # tests (gpt2 test needs gpt2 in HF cache)
scripts\run.ps1 [web|tui|capture ...]          # run from the venv
node bin/brainscan.js <args>                   # npm launcher (uses .venv if present, else ~/.brainscan/venv)
```

## Gotchas
- HF `hidden_states[-1]` is already final-normed: the logit lens must not normalise it again.
- Gemma 3 4B+ nests the LM (`model.language_model`); `_final_norm()` handles known paths.
- `artifacts/` and generated text are never committed.
