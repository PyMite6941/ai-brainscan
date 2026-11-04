---
layout: default
title: ai-brainscan
---

# ai-brainscan

**Watch a language model think, layer by layer.**

An open interpretability instrument: it loads a model already on your laptop, runs a prompt through it, records the residual stream at every layer, and lets you look at it in a web UI, a terminal UI, or the CLI.

## Docs

- [DEVLOG](DEVLOG.md) — session log: what changed, why, what's open
- [MODULES](MODULES.md) — the mixture-of-modules registry and adapter contract

## Install

Pick one. All end up with the same `brainscan` command. Python 3.10+ required; no GPU needed.

**npm (any OS)**

```bash
npm install -g github:PyMite6941/ai-brainscan
brainscan setup        # first run: builds a private venv, downloads the default model
brainscan web
```

**From a clone**

| OS | install | first-run setup | run |
|---|---|---|---|
| Windows (PowerShell) | `scripts\install.ps1` | `scripts\setup.ps1` | `scripts\run.ps1` |
| Windows (cmd) | `scripts\install.cmd` | `scripts\setup.cmd` | `scripts\run.cmd` |
| macOS / Linux | `scripts/install.sh` | `scripts/setup.sh` | `scripts/run.sh` |

**pip**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # skip the index-url on macOS
pip install "ai-brainscan[web,tui] @ git+https://github.com/PyMite6941/ai-brainscan"
```

## Use

```bash
brainscan doctor                       # is everything installed?
brainscan models                       # which local models can be scanned
brainscan capture "The capital of the state containing Dallas is" --label geography
brainscan capture "How do I pick a lock?" --label probe -g 48      # generate + judge refused/complied
brainscan list
brainscan show <scan-id>               # logit lens: what it is about to say, at every layer
brainscan web [--share]                # Gradio UI; --share gives teammates a public link
brainscan tui                          # terminal UI
brainscan modules                      # the mixture of modules and what is installed
```

## How it fits together

```
capture (slow, loads the model)  ──writes──▶  artifacts/<scan_id>/   ◀──reads──  web · tui · cli (instant)
                                               meta.json  resid.npy  lens.json  [attn.npy]  modules/*.json
```

## What is built vs. what is not

| Piece | Status |
|---|---|
| capture: residual stream + logit lens + attention, any HF model | working |
| web UI (heatmap, logit lens, A/B compare, modules) | working |
| terminal UI | working |
| model discovery, download, doctor | working |
| refusal direction, persona drift, ablation (`src/brainscan/analysis/`) | Track A: to be written by the team |
| black-box module adapters | registered, adapters to be written |
| SAE features, attribution graphs | planned |

## Ground rules

- Model weights never get committed. They download at runtime into the Hugging Face cache, outside the repo.
- Harmful outputs never get committed. Scans store a hash and a refused/complied verdict. Generated text is only kept with `--keep-output`, and `artifacts/` is gitignored.
- Measure over-refusal too. Safe-but-scary prompts go in every experiment as a control.
- Correlation isn't enough. A direction counts once ablating it changes behaviour.
- Ablation stays a runtime hook. Never publish an edited checkpoint.

## Licence

Code: Apache-2.0. Model weights are under their own terms (Gemma Terms of Use, Llama licence, ...). See [LICENSE](../LICENSE) and [CITATION.cff](../CITATION.cff).
