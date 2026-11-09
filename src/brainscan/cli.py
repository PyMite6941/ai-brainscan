"""`brainscan` command. Every other entry point (npm, run scripts) ends here."""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from brainscan import __version__, config

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="ai-brainscan: watch a language model think, layer by layer.",
)
console = Console()


def _scan_path(scan: str) -> Path:
    p = Path(scan)
    if p.is_dir():
        return p
    root = config.artifacts_dir()
    if (root / scan).is_dir():
        return root / scan
    matches = [d for d in root.iterdir() if d.is_dir() and d.name.startswith(scan)]
    if len(matches) == 1:
        return matches[0]
    raise typer.BadParameter(f"no unique scan matching {scan!r} in {root}")


@app.command()
def version():
    """Print the version."""
    console.print(__version__)


@app.command()
def doctor():
    """Check the install: python, torch, extras, modules, model cache."""
    from brainscan.models import local_models, ollama_models
    from brainscan.modules import REGISTRY

    ok = True
    t = Table(title=f"ai-brainscan {__version__}", show_header=False)
    t.add_row("python", f"{sys.version.split()[0]} ({platform.system()} {platform.machine()})")
    try:
        import torch

        accel = "cuda" if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu only"
        )
        t.add_row("torch", f"{torch.__version__} [{accel}]")
    except ImportError:
        ok = False
        t.add_row("torch", "[red]MISSING[/] run scripts/install")
    import transformers

    t.add_row("transformers", transformers.__version__)
    for extra, mod in (("web", "gradio"), ("tui", "textual")):
        have = importlib.util.find_spec(mod) is not None
        t.add_row(escape(f"[{extra}] {mod}"),
                  "ok" if have else "[yellow]not installed[/] " + escape(f'pip install "ai-brainscan[{extra}]"'))
    hf = config.hf_home()
    sync = config.synced_folder(hf)
    t.add_row("HF cache", f"{hf}" + (f"  [red]inside {sync}! move it (HF_HOME)[/]" if sync else ""))
    t.add_row("artifacts", str(config.artifacts_dir()))
    t.add_row("default model", config.default_model())
    console.print(t)

    mt = Table(title="modules")
    for col in ("key", "category", "status"):
        mt.add_column(col)
    for m in REGISTRY.values():
        mt.add_row(m.info.key, m.info.category, "ok" if m.available() else f"[dim]{escape(m.install_hint())}[/]")
    console.print(mt)

    scannable = [m for m in local_models() if m.scannable]
    console.print(f"scannable local models: {len(scannable)}"
                  + ("" if scannable else "  -> brainscan download " + config.default_model()))
    if ollama_models():
        console.print("[yellow]Ollama models found but not scannable[/] (GGUF exposes no hidden states).")
    raise typer.Exit(0 if ok else 1)


@app.command()
def models(all: bool = typer.Option(False, "--all", help="include non-scannable entries")):
    """List models on this machine and whether they can be scanned."""
    from brainscan.models import local_models, ollama_models

    t = Table()
    for col in ("model", "layers", "d_model", "GB", "scannable", "note"):
        t.add_column(col)
    for m in local_models():
        if m.scannable or all:
            t.add_row(m.repo_id, str(m.n_layers or "?"), str(m.d_model or "?"), f"{m.size_gb}",
                      "[green]yes[/]" if m.scannable else "no", m.note)
    console.print(t)
    oll = ollama_models()
    if oll:
        console.print(f"[yellow]{len(oll)} Ollama model(s) skipped:[/] GGUF via llama.cpp returns tokens "
                      "only, no hidden states. Download the HF safetensors version instead.")


@app.command()
def download(model_id: str = typer.Argument(None, help="HF repo id (default: BRAINSCAN_MODEL)")):
    """Download a model's weights into the HF cache (never into the repo)."""
    from brainscan.models import download as dl

    model_id = model_id or config.default_model()
    sync = config.synced_folder(config.hf_home())
    if sync:
        console.print(f"[red]HF cache is inside {sync}[/]; set HF_HOME to a local folder first.")
        raise typer.Exit(1)
    console.print(f"downloading {model_id} ...")
    console.print(f"[green]done[/] {dl(model_id)}")


@app.command()
def capture(
    prompt: str = typer.Argument(..., help="prompt text, or @file.json with a list of chat messages"),
    model: str = typer.Option(None, "--model", "-m", help="HF repo id"),
    label: str = typer.Option("", "--label", "-l", help="e.g. refused / jailbreak:dan / control"),
    chat: bool = typer.Option(True, "--chat/--raw", help="apply the chat template (on for -it models)"),
    top_k: int = typer.Option(5, "--top-k"),
    attn: bool = typer.Option(False, "--attn", help="also save attention patterns"),
    generate: int = typer.Option(0, "--generate", "-g", help="generate N tokens and judge refused/complied"),
    keep_output: bool = typer.Option(False, "--keep-output", help="store generated text (gitignored)"),
    module: list[str] = typer.Option(None, "--module", help="run a module after capture (e.g. llama_guard, llm_guard). repeatable"),  # noqa: B008
):
    """Run one prompt through the model and save a scan to artifacts/."""
    from brainscan.capture import capture as run
    from brainscan.modules import REGISTRY, RunContext

    model = model or config.default_model()
    p: str | list = prompt
    if prompt.startswith("@"):
        p = json.loads(Path(prompt[1:]).read_text(encoding="utf-8"))
    with console.status(f"loading {model} and running the forward pass ..."):
        scan = run(model, p, label=label, chat=chat, top_k=top_k, attentions=attn,
                   generate_tokens=generate, keep_output=keep_output)
    out = scan.save(config.artifacts_dir())
    console.print(f"[green]scan saved[/] {out}")
    if scan.meta.verdict:
        console.print(f"verdict (heuristic): {scan.meta.verdict}")
    console.print(scan.meta.scan_id)

    if module:
        ctx = RunContext(
            prompt=scan.meta.prompt,
            model_id=model,
            output_text=scan.meta.output_text,
            scan=scan,
        )
        for key in module:
            try:
                mod = REGISTRY[key]
            except KeyError:
                console.print(f"[red]unknown module {key!r}; known: {', '.join(REGISTRY)}[/]")
                continue
            if not mod.available():
                console.print(f"[yellow]{key}[/] not installed: {escape(mod.install_hint())}")
                continue
            with console.status(f"running module {key} ..."):
                try:
                    result = mod.run(ctx)
                    scan.save_module_result(result)
                    console.print(
                        f"[green]module {key}[/]: verdict={result.verdict} "
                        f"score={result.score}"
                    )
                except NotImplementedError as e:
                    console.print(f"[yellow]{key}[/]: adapter not implemented ({e})")
                except Exception as e:  # noqa: BLE001 - report module errors, keep going
                    console.print(f"[red]{key}[/]: error: {e}")


@app.command("list")
def list_scans():
    """List saved scans."""
    from brainscan.contracts import list_scans as ls

    t = Table()
    for col in ("scan_id", "model", "label", "tokens", "verdict"):
        t.add_column(col)
    for m in ls(config.artifacts_dir()):
        t.add_row(m.scan_id, m.model_id, m.label, str(len(m.tokens)), m.verdict or "")
    console.print(t)


@app.command()
def show(
    scan: str = typer.Argument(..., help="scan id (prefix ok) or path"),
    position: int = typer.Option(-1, "--pos", "-p", help="token position to trace (default: last)"),
    every: int = typer.Option(1, "--every", help="show every Nth layer"),
):
    """Logit lens for one token position: what the model 'is about to say' at each layer."""
    from brainscan.contracts import Scan

    s = Scan.load(_scan_path(scan))
    toks = s.meta.tokens
    pos = position if position >= 0 else len(toks) + position
    norms = s.resid_norm()
    console.print(f"[bold]{s.meta.model_id}[/]  label={s.meta.label!r}  position {pos} = {toks[pos]!r}")
    t = Table()
    for col in ("layer", "|resid|", "top predictions"):
        t.add_column(col)
    for layer in range(0, len(s.lens), every):
        preds = "  ".join(f"{tok!r} {p:.2f}" for tok, p in s.lens[layer][pos])
        t.add_row(str(layer), f"{norms[layer, pos]:.1f}", preds)
    console.print(t)


@app.command("modules")
def list_modules():
    """List the mixture of modules and which are installed."""
    from brainscan.modules import CATEGORIES, by_category

    for cat, desc in CATEGORIES.items():
        t = Table(title=f"{cat}: {desc}", title_justify="left")
        for col in ("key", "name", "licence", "installed", "what it tells us"):
            t.add_column(col)
        for m in by_category(cat):
            t.add_row(m.info.key, m.info.name, m.info.license,
                      "[green]yes[/]" if m.available() else "no", m.info.what)
        console.print(t)


@app.command()
def setup(model: str | None = typer.Option(None, "--model", "-m"),
          skip_download: bool = typer.Option(False, "--skip-download")):
    """First-run setup: create .env, download the default model, run doctor."""
    env = Path.cwd() / ".env"
    example = Path(__file__).resolve().parents[2] / ".env.example"
    if not env.exists():
        env.write_text(example.read_text(encoding="utf-8") if example.exists()
                       else f"BRAINSCAN_MODEL={config.DEFAULT_MODEL}\n", encoding="utf-8")
        console.print(f"wrote {env} (add HF_TOKEN there only if you use gated google/* repos)")
    if not skip_download:
        download(model)
    doctor()


@app.command()
def web(port: int = typer.Option(7860, "--port"),
        share: bool = typer.Option(False, "--share", help="public gradio.live link for teammates"),
        host: str = typer.Option("127.0.0.1", "--host")):
    """Launch the web UI (Gradio)."""
    try:
        from brainscan.ui.web import launch
    except ImportError as e:
        console.print(f"[red]{escape(str(e))}[/]\n" + escape('pip install "ai-brainscan[web]"'))
        raise typer.Exit(1)
    launch(host=host, port=port, share=share)


@app.command()
def tui():
    """Launch the terminal UI (Textual)."""
    try:
        from brainscan.ui.tui import run
    except ImportError as e:
        console.print(f"[red]{escape(str(e))}[/]\n" + escape('pip install "ai-brainscan[tui]"'))
        raise typer.Exit(1)
    run()


if __name__ == "__main__":
    app()
