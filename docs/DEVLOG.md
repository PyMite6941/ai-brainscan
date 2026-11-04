# Dev log

Newest first. One entry per working session: what changed, why, what is still open.
Decisions that would be expensive to re-derive go here, not only in commit messages.

---

## 2026-11-03: repo scaffold

**Goal:** get the repo ready for team development: one Python package, three front ends (CLI, web, TUI),
installable via npm, with install/setup/run scripts for Windows, macOS and Linux.

### Built
- `pyproject.toml`: package `ai-brainscan`, CLI entry `brainscan`. Extras: `web` (Gradio + Plotly), `tui` (Textual),
  `whitebox` (TransformerLens, nnsight, SAELens), one extra per black-box tool (`guardrails`, `nemo`,
  `guardrails-ai`, `eval`, `garak`, `pyrit`), `dev`.
- `contracts.py`: the on-disk Scan format (`artifacts/<scan_id>/meta.json, resid.npy, lens.json, attn.npy,
  modules/*.json`). `SCHEMA_VERSION = 1`.
- `capture.py`: one forward pass with `output_hidden_states`, logit lens through the model's own final norm +
  unembedding, optional attention, optional generation with a keyword refused/complied verdict.
- `models.py`: local model discovery via `huggingface_hub.scan_cache_dir()`; Ollama models reported, not offered.
- `modules/`: mixture-of-modules registry, 4 categories, 10 modules. Only `native` has an adapter so far.
- `analysis/`: `directions.py`, `drift.py`, `patching.py` are contract-only stubs (Track A).
- `cli.py`: `doctor, models, download, capture, list, show, modules, setup, web, tui, version`.
- `ui/web.py`: Gradio with Scan / Explore (layer × token heatmap, logit lens table) / Compare (A vs B depth curve)
  / Modules tabs.
- `ui/tui.py`: Textual viewer (delegated to Hermes, under review).
- `bin/brainscan.js` + `package.json`: npm launcher, zero dependencies. Builds a venv on first run.
- `scripts/{install,setup,run}.{sh,ps1,cmd}`.
- Docs: README, CONTRIBUTING, CITATION.cff, LICENSE (official Apache-2.0 text), docs/MODULES.md, CLAUDE.md.

### Decisions
| Decision | Why |
|---|---|
| Capture writes files and the UIs only read them | A model reload on every click is unusable on a CPU laptop. This split lets web, TUI and CLI all exist cheaply, and lets team tracks work in parallel. |
| Save the full residual stream, not just summaries | Every later analysis (refusal direction, drift, SAE features) is a function of it, so old scans never need re-running. Costs ~2.4 KB per token-layer in fp16 for the 1B model. |
| Plain `transformers` hidden states, not TransformerLens, for the native path | Works on any HF causal LM today, including Gemma 3 and transformers 5. TransformerLens/nnsight stay available as modules. |
| Logit lens skips the final norm on the last layer | HF already applies it to `hidden_states[-1]`. Applying it twice is wrong for Gemma's `(1 + w)` RMSNorm. |
| fp32 on CPU by default | bf16 matmuls are slow or unsupported on many laptop CPUs. 1B in fp32 is ~4 GB of RAM. |
| Gradio for the web UI | `--share` links let teammates see each other's work without a deploy, and its queue stops two users OOMing the model. (Carried over from AI Guardrail Research §14.) |
| torch is not a declared dependency | The right wheel depends on OS and GPU. The install scripts and npm launcher choose: CPU index on Windows/Linux, default on macOS, CUDA with `--gpu`. |
| npm launcher installs Python on first run, not in `postinstall` | Keeps `npm i` fast and CI-safe. A failed pip install gives a readable error at `brainscan` time instead of a broken npm install. |
| Default model `unsloth/gemma-3-1b-it` | Ungated mirror (no HF token needed). 26 layers, runs on CPU, and has Gemma Scope 2 SAEs/transcoders for later. |
| Generated text is hashed, never stored, unless `--keep-output` | Project ground rule: no elicited harmful outputs in the repo. `artifacts/` is gitignored too. |
| Verdict is a keyword heuristic, labelled as such | Good enough to sort a sweep. Not reportable; the Llama Guard module replaces it. |

### Open
- [ ] Review Hermes output: `ui/tui.py`, `tests/`.
- [ ] Feasibility test on Gemma 3 1B: does the logit lens show the Dallas → Texas → Austin hop? Does a throwaway
      difference-in-means direction separate refused from complied on held-out prompts?
- [ ] Black-box module adapters (Hermes-sized tasks, one per module).
- [ ] Create the GitHub repo `PyMite6941/ai-brainscan` (does not exist yet) + branch protection.
- [ ] Gemma Scope 2 SAE features (View 02) and circuit-tracer graphs (View 04).
