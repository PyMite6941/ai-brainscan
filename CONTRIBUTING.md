# Contributing

All changes go through a pull request with at least one review. `main` is protected.

1. Branch from `main`: `git switch -c your-name/short-topic`.
2. Install from your clone (`scripts/install.*`), then `scripts/run.* doctor`.
3. Keep to your track's folder to avoid conflicts:

| Track | Folder |
|---|---|
| A: capture & core research | `src/brainscan/capture.py`, `src/brainscan/analysis/` |
| B: attacks & data | `data/` (prompt sets), sweep scripts |
| C: features & meaning | `src/brainscan/modules/` (SAE / Neuronpedia adapters) |
| D: the instrument | `src/brainscan/ui/` |

4. If you change `src/brainscan/contracts.py`, bump `SCHEMA_VERSION` and say so in the PR: every track reads it.
5. `python -m pytest -q` and `python -m ruff check src tests` must pass.
6. Never commit weights, `artifacts/`, `.env`, or generated harmful text.
