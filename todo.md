# ai-brainscan — todo

Source of truth for open work: `docs/DEVLOG.md` ("Open" section). Track A (`src/brainscan/analysis/`) is Matt's — leave the `NotImplementedError` stubs alone unless he asks.

## npm / GitHub setup
- [x] npm: `files` whitelists `src/brainscan/**/*.py` so `__pycache__`/`egg-info` never ship (npm ignores `.npmignore` inside `files` entries); `npm pack` → 33 files, clean
- [x] commit + push to `PyMite6941/ai-brainscan` main
- [x] GitHub: branch protection on main (require PR + 1 review, dismiss stale, resolve conversation; no status checks — no CI yet; admins can bypass)
- [ ] npm: 0.1.0 is already on the registry (with the pyc/egg-info junk). Bump to 0.1.1 and `npm publish --access public` — needs `npm login` with `@ai-clyde` org access

## Verification
- [x] `python -m pytest -q` green (23 passed — gpt2 test needs gpt2 in HF cache)
- [x] `python -m ruff check src tests` clean
- [x] `node bin/brainscan.js doctor` works from the venv (fixed: Rich was eating `[extra]` in install hints)

## From DEVLOG open (in priority order)
- [ ] Review Hermes output: `ui/tui.py`, `tests/` — read both; tests pass. Close if nothing wrong.
- [ ] Feasibility test on Gemma 3 1B (logit lens Dallas→Texas→Austin hop; throwaway difference-in-means direction) — heavy, needs model download
- [ ] Black-box module adapters (`modules/`, Claude track): nemo_guardrails, guardrails_ai, garak, pyrit, inspect — each needs its heavy lib installed
- [ ] Gemma Scope 2 SAE features (View 02) and circuit-tracer graphs (View 04) — planned, not started
