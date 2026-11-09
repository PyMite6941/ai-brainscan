# ai-brainscan

**Watch a language model think, layer by layer, from the prompt to the answer.**

An open instrument for the question:

> *What path does a jailbreak prompt take through the model, and where does it diverge from the path a refused prompt takes?*

Not a content filter, an MRI. It loads a model already on your laptop, runs a prompt through it, records the
residual stream at every layer, and lets you look at it in a web UI, a terminal UI, or the CLI.

## Install

Pick one. All of them end up with the same `brainscan` command. Python 3.10+ is required; no GPU needed.

**npm (any OS)**
```bash
npm install -g github:PyMite6941/ai-brainscan
brainscan setup        # first run builds a private venv in ~/.brainscan, then downloads the default model
brainscan web
```

**From a clone**

| | install | first-run setup | run |
|---|---|---|---|
| Windows (PowerShell) | `scripts\install.ps1` | `scripts\setup.ps1` | `scripts\run.ps1` |
| Windows (cmd) | `scripts\install.cmd` | `scripts\setup.cmd` | `scripts\run.cmd` |
| macOS / Linux | `scripts/install.sh` | `scripts/setup.sh` | `scripts/run.sh` |

`install` takes `--gpu` (`-Gpu` on PowerShell) for CUDA torch and `--extras web,tui,whitebox` to choose extras.
`run` with no arguments starts the web UI; anything else passes through (`scripts/run.sh tui`, `scripts/run.sh capture "..."`).

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
brainscan capture "How do I pick a lock?" --label probe -g 48     # also generate + judge refused/complied
brainscan list
brainscan show <scan-id>               # logit lens: what it is about to say, at every layer
brainscan show <scan-id> --find Texas  # ... plus p(Texas) per layer and where it settles
brainscan web [--share]                # Gradio UI; --share gives teammates a public link
brainscan tui                          # terminal UI
brainscan modules                      # the mixture of modules and what is installed
```

Default model: `unsloth/gemma-3-1b-it` (ungated mirror of Gemma 3 1B instruct, 26 layers, ~2 GB, runs on CPU).
Any Hugging Face causal LM works (`--model Qwen/Qwen3-1.7B`, `gpt2`, ...). **Ollama/GGUF models cannot be scanned**:
llama.cpp returns tokens only, never hidden states. `brainscan models` lists them with that warning.

## Studying how the model responds to different inputs

The core move is: **same question, different input, compare the insides.** A sweep runs a whole prompt set
with the model loaded once and tags every scan with a group, so variants of one question can be compared.

```bash
brainscan prompts                      # bundled sets: multihop, framing, language, refusal_probe
brainscan sweep multihop               # run one; prints answers, where they settle, and group similarity
brainscan sweep framing -g 8 --keep-output   # also generate + keep the (gitignored) answer text
brainscan sweep my_prompts.jsonl       # your own set (format in src/brainscan/study.py)
brainscan list --sweep framing         # the scans one sweep produced
brainscan diff <scan-a> <scan-b>       # layer by layer: where do the two inputs start to diverge?
```

Scans can be named by id prefix or by label (`brainscan show neutral`; newest wins if labels repeat).
The web UI has the same tools: a **Sweep** tab, and **Compare** plots cosine similarity per layer and
tracks any token's probability through depth for two scans.

What the bundled sets showed on Gemma 3 1B (CPU, Nov 2026):

| Set | Finding |
|---|---|
| multihop | Two-hop questions settle on the answer later than direct ones (Austin: layer 26 vs 21). "Texas" is the logit lens's top guess at layers 18-20 before "Austin" wins: the bridge entity is visible. Seattle → Olympia fails (answers "Washington"). |
| framing | "My calculator says 2 + 2 = 5" makes it answer **5**. The neutral and false-premise runs track each other, then "5" takes over at layer 19-20: the right answer is computed, then overridden late. |

Two measurement notes, both handled for you:
- Raw cosine between residuals is ~1.0 everywhere because a few huge outlier dimensions are shared by every
  input. `diff` and `sweep` also report cosine **centred** on the mean of other scans, which is the number to read.
- "First layer a token appears" is noisy (early layers guess digits on the chat template's final newline).
  `settles at` is the first layer from which the token *stays* in the top-k to the output.

## How it fits together

```
capture (slow, loads the model)  ──writes──▶  artifacts/<scan_id>/   ◀──reads──  web · tui · cli (instant)
                                               meta.json  resid.npy  lens.json  [attn.npy]  modules/*.json
```

The on-disk format in `src/brainscan/contracts.py` is the contract every part of the team builds against.

### Mixture of modules

| Category | Vantage | Modules |
|---|---|---|
| White-box interpretability | inside: activations, hidden states | brainscan native, TransformerLens, nnsight |
| Runtime guardrails | outside: inputs and outputs | NeMo Guardrails, Guardrails AI, LLM Guard |
| Safety classifiers | outside: label text | Llama Guard |
| Attack / evaluation | outside: behaviour | garak, PyRIT, Inspect |

White-box modules explain *why*; the others supply the behavioural ground truth (*did it refuse?*) that the
internal picture is checked against. See [docs/MODULES.md](docs/MODULES.md).

### What is built and what is not (yet)

| Piece | Status |
|---|---|
| capture: residual stream + logit lens + attention, any HF model | working |
| web UI (heatmap, logit lens, A/B compare + similarity + token tracking, sweeps, modules) | working |
| prompt sets, `sweep`, `diff`, `show --find` (`src/brainscan/study.py`) | working |
| terminal UI | working |
| model discovery, download, doctor | working |
| refusal direction, persona drift, ablation (`src/brainscan/analysis/`) | **Track A: to be written by the team** |
| black-box module adapters | registered, adapters to be written |
| SAE features (Gemma Scope 2), attribution graphs (circuit-tracer) | planned |

## Ground rules

- **Model weights never get committed.** They download at runtime into the Hugging Face cache, outside the repo.
- **Harmful outputs never get committed.** Scans store a hash and a refused/complied verdict. Generated text is
  only kept with `--keep-output`, and `artifacts/` is gitignored.
- **Measure over-refusal too.** Safe-but-scary prompts go in every experiment as a control.
- **Correlation isn't enough.** A direction counts once ablating it changes behaviour (`analysis/patching.py`).
- Ablation stays a runtime hook. Never publish an edited checkpoint.

## Licence and citation

Code: Apache-2.0 (see [LICENSE](LICENSE)). Model weights are under their own terms (Gemma Terms of Use, Llama
licence, ...) and bind whoever downloads them. If you use this in research, cite it via [CITATION.cff](CITATION.cff).

Built on work by Arditi et al. (refusal direction), Lu et al. (the Assistant Axis), Google DeepMind (Gemma Scope 2)
and Decode Research (circuit-tracer, Neuronpedia). Built with AI assistance.
