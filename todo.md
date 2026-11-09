# ai-brainscan — todo

Source of truth for open work: `docs/DEVLOG.md` ("Open" section). Track A (`src/brainscan/analysis/`) is Matt's — leave the `NotImplementedError` stubs alone unless he asks.

## npm / GitHub setup
- [x] npm: `files` whitelists `src/brainscan/**/*.py` so `__pycache__`/`egg-info` never ship (npm ignores `.npmignore` inside `files` entries); `npm pack` → 33 files, clean
- [x] commit + push to `PyMite6941/ai-brainscan` main
- [x] GitHub: branch protection on main (require PR + 1 review, dismiss stale, resolve conversation; no status checks — no CI yet; admins can bypass)
- [ ] npm: 0.1.0 is on the registry (with pyc/egg-info junk). 0.2.0 is bumped and packs clean (38 files) — `npm login` (`@ai-clyde` org access), then `npm publish --access public`

## Study tooling (done 2026-11-09)
- [x] `brainscan sweep <set>` / `prompts` / `diff` / `show --find` / label lookup; `src/brainscan/study.py`
- [x] bundled prompt sets: multihop, framing, language, refusal_probe (`src/brainscan/prompts/`)
- [x] web UI: Sweep tab; Compare gets cosine-per-layer (raw + centred) and token tracking — verified in browser
- [ ] next: `sweep refusal_probe -g 32` to get refused/complied labels → input for Track A's `refusal_direction`
- [ ] TUI doesn't know about sweeps/diff yet

## Verification
- [x] `python -m pytest -q` green (23 passed — gpt2 test needs gpt2 in HF cache)
- [x] `python -m ruff check src tests` clean
- [x] `node bin/brainscan.js doctor` works from the venv (fixed: Rich was eating `[extra]` in install hints)

## From DEVLOG open (in priority order)
- [ ] Review Hermes output: `ui/tui.py`, `tests/` — read both; tests pass. Close if nothing wrong.
- [ ] Feasibility test on Gemma 3 1B — logit-lens half **done**: "Texas" is top-1 at layers 18-20, "Austin" wins at 26. Remaining: throwaway difference-in-means direction (Track A)
- [ ] Black-box module adapters (`modules/`, Claude track): nemo_guardrails, guardrails_ai, garak, pyrit, inspect — each needs its heavy lib installed
- [ ] Gemma Scope 2 SAE features (View 02) and circuit-tracer graphs (View 04) — planned, not started
