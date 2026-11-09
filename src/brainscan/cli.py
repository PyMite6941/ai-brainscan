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
    if not matches:
        from brainscan.contracts import list_scans as ls

        by_label = [m for m in ls(root) if m.label == scan]  # newest first
        if by_label:
            if len(by_label) > 1:
                console.print(f"[dim]{len(by_label)} scans labelled {scan!r}; using newest "
                              f"{by_label[0].scan_id}[/]")
            return root / by_label[0].scan_id
    raise typer.BadParameter(f"no unique scan matching {scan!r} (id prefix or label) in {root}")


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
def list_scans(sweep: str = typer.Option(None, "--sweep", help="only scans from this sweep run id")):
    """List saved scans."""
    from brainscan.contracts import list_scans as ls

    t = Table()
    t.add_column("scan_id", no_wrap=True, min_width=22)  # never truncate: it's what you copy
    for col in ("model", "label", "group", "tokens", "verdict"):
        t.add_column(col)
    for m in ls(config.artifacts_dir()):
        sw = m.extra.get("sweep") or {}
        if sweep and not str(sw.get("run_id", "")).startswith(sweep):
            continue
        t.add_row(m.scan_id, m.model_id, escape(m.label), escape(sw.get("group", "")),
                  str(len(m.tokens)), m.verdict or "")
    console.print(t)


@app.command()
def show(
    scan: str = typer.Argument(..., help="scan id (prefix ok), label, or path"),
    position: int = typer.Option(-1, "--pos", "-p", help="token position to trace (default: last)"),
    every: int = typer.Option(1, "--every", help="show every Nth layer"),
    find: str = typer.Option(None, "--find", "-f",
                             help="token to track, e.g. Texas: prints its probability per layer"),
):
    """Logit lens for one token position: what the model 'is about to say' at each layer."""
    from brainscan.contracts import Scan
    from brainscan.study import first_layer, settles_at, token_prob_curve

    s = Scan.load(_scan_path(scan))
    toks = s.meta.tokens
    pos = position if position >= 0 else len(toks) + position
    norms = s.resid_norm()
    console.print(f"[bold]{s.meta.model_id}[/]  label={escape(repr(s.meta.label))}  "
                  f"position {pos} = {escape(repr(toks[pos]))}")
    if s.meta.output_text:
        console.print(f"output: {escape(repr(s.meta.output_text))}")
    curve = token_prob_curve(s, find, pos) if find else None
    t = Table()
    cols = ["layer", "|resid|"] + ([f"p({find})"] if find else []) + ["top predictions"]
    for col in cols:
        t.add_column(escape(col))
    for layer in range(0, len(s.lens), every):
        preds = "  ".join(f"{tok!r} {p:.2f}" for tok, p in s.lens[layer][pos])
        row = [str(layer), f"{norms[layer, pos]:.1f}"]
        if curve is not None:
            row.append(f"{curve[layer]:.2f}" if curve[layer] else "")
        t.add_row(*row, escape(preds))
    console.print(t)
    if find:
        k = len(s.lens[0][pos])
        fk, sk, s1 = first_layer(s, find, pos, k), settles_at(s, find, pos, k), settles_at(s, find, pos, 1)
        if fk is None:
            console.print(f"{escape(repr(find))} never reaches the top-{k}")
        else:
            console.print(f"{escape(repr(find))}: first in top-{k} at layer {fk}; "
                          f"stays in top-{k} from layer {sk if sk is not None else '-'}; "
                          f"stays top-1 from layer {s1 if s1 is not None else '-'}")


@app.command()
def diff(
    a: str = typer.Argument(..., help="scan A (id prefix, label or path)"),
    b: str = typer.Argument(..., help="scan B"),
    pos_a: int = typer.Option(-1, "--pos-a"),
    pos_b: int = typer.Option(-1, "--pos-b"),
    every: int = typer.Option(1, "--every", help="show every Nth layer"),
):
    """Compare two scans layer by layer: where do the two inputs start being treated differently?"""
    from brainscan.contracts import Scan
    from brainscan.contracts import list_scans as ls
    from brainscan.study import baseline_mean
    from brainscan.study import diff as run_diff

    sa, sb = Scan.load(_scan_path(a)), Scan.load(_scan_path(b))
    root = config.artifacts_dir()
    same_model = [Scan.load(root / m.scan_id) for m in ls(root)
                  if m.model_id == sa.meta.model_id and m.d_model == sa.meta.d_model][:200]
    center = baseline_mean(same_model)
    try:
        rows = run_diff(sa, sb, pos_a, pos_b, center=center)
    except ValueError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    console.print(f"A = {sa.meta.scan_id} {escape(repr(sa.meta.label))}\n"
                  f"B = {sb.meta.scan_id} {escape(repr(sb.meta.label))}")
    key = (lambda r: r.cosine_centred) if center is not None else (lambda r: r.cosine)
    t = Table()
    cols = ["layer", "cos raw"] + (["cos centred"] if center is not None else []) + \
        ["", "|A|", "|B|", "A top-1", "B top-1"]
    for col in cols:
        t.add_column(col)
    for r in rows[::every]:
        v = key(r)
        bar = "█" * max(0, round(v * 8))
        same = "" if r.top_a.strip() != r.top_b.strip() else "[green]"
        t.add_row(str(r.layer), f"{r.cosine:.3f}",
                  *([f"{r.cosine_centred:.3f}"] if center is not None else []),
                  f"[cyan]{bar}[/]", f"{r.norm_a:.0f}", f"{r.norm_b:.0f}",
                  same + escape(repr(r.top_a)), same + escape(repr(r.top_b)))
    console.print(t)
    if center is not None:
        console.print(f"[dim]centred on the mean of {len(same_model)} saved scans of this model "
                      "(removes the outlier dimensions every input shares)[/]")
    else:
        console.print("[dim]raw cosine only: save 3+ scans of this model for a centred baseline[/]")
    low = min(rows[1:], key=key)
    console.print(f"most different at layer {low.layer} ({key(low):.3f}); final layer {key(rows[-1]):.3f}")


@app.command()
def prompts():
    """List the bundled prompt sets for `brainscan sweep`."""
    from brainscan.study import bundled_prompt_sets, load_prompt_set

    t = Table()
    for col in ("set", "prompts", "groups", "about"):
        t.add_column(col)
    for p in bundled_prompt_sets():
        items = load_prompt_set(p)
        about = next((ln.lstrip("# ").strip() for ln in p.read_text(encoding="utf-8").splitlines()
                      if ln.startswith("#")), "")
        t.add_row(p.stem, str(len(items)), str(len({i.group for i in items if i.group})), about)
    console.print(t)
    console.print("run one: [bold]brainscan sweep multihop[/]  (or pass your own .jsonl)")


@app.command()
def sweep(
    prompt_set: str = typer.Argument(..., help="bundled set name (see `brainscan prompts`) or a .jsonl path"),
    model: str = typer.Option(None, "--model", "-m", help="HF repo id"),
    chat: bool = typer.Option(True, "--chat/--raw", help="default for items without a 'chat' field"),
    generate: int = typer.Option(0, "--generate", "-g", help="generate N tokens per prompt and judge"),
    keep_output: bool = typer.Option(False, "--keep-output", help="store generated text (gitignored)"),
    top_k: int = typer.Option(5, "--top-k"),
    limit: int = typer.Option(0, "--limit", help="only the first N prompts (0 = all)"),
):
    """Run a whole prompt set with the model loaded once. Every scan is tagged with the sweep
    run id, group and label, then a summary compares variants within each group."""
    import datetime as dt

    from brainscan.capture import capture as run
    from brainscan.study import (
        load_prompt_set,
        mean_vector,
        resolve_prompt_set,
        settles_at,
        similarity_matrix,
    )

    try:
        path = resolve_prompt_set(prompt_set)
        items = load_prompt_set(path)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]{escape(str(e))}[/]")
        raise typer.Exit(1)
    if limit:
        items = items[:limit]
    model = model or config.default_model()
    run_id = f"{path.stem}-{dt.datetime.now():%Y%m%d-%H%M%S}"
    console.print(f"sweep [bold]{run_id}[/]: {len(items)} prompts on {model}")

    scans = []
    with console.status(f"loading {model} ...") as status:
        for i, it in enumerate(items, 1):
            status.update(f"[{i}/{len(items)}] {it.group or '-'} / {it.label or '-'}")
            s = run(model, it.prompt, label=it.label, chat=chat if it.chat is None else it.chat,
                    top_k=top_k, generate_tokens=generate, keep_output=keep_output)
            s.meta.extra["sweep"] = {"run_id": run_id, "set": path.stem, "index": i - 1,
                                     "group": it.group, "expect": it.expect}
            s.save(config.artifacts_dir())
            scans.append((it, s))

    t = Table(title=f"sweep {run_id}")
    for col in ("#", "group", "label", "final top-1", "expect", f"expect settles (top-{top_k} / top-1)",
                "verdict"):
        t.add_column(escape(col))
    for i, (it, s) in enumerate(scans):
        final = s.top_token()[-1][-1]
        seen = ""
        if it.expect:
            fk, f1 = settles_at(s, it.expect, -1, top_k), settles_at(s, it.expect, -1, 1)
            seen = "-" if fk is None else f"L{fk} / " + ("-" if f1 is None else f"L{f1}")
        t.add_row(str(i), escape(it.group), escape(it.label), escape(repr(final)),
                  escape(it.expect or ""), seen, s.meta.verdict or "")
    console.print(t)

    groups: dict[str, list] = {}
    for it, s in scans:
        if it.group:
            groups.setdefault(it.group, []).append(s)
    groups = {g: v for g, v in groups.items() if len(v) > 1}
    if groups:
        n_layers = scans[0][1].meta.n_layers
        layers = sorted({1, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers})
        all_scans = [s for _, s in scans]
        centers = {layer: mean_vector(all_scans, layer) for layer in layers}
        gt = Table(title="within-group similarity to the group's first prompt "
                         "(cosine at the last token, centred on the sweep mean)")
        for col in ("group", "variant", *[f"L{x}" for x in layers]):
            gt.add_column(col)
        for g, ss in groups.items():
            sims = [similarity_matrix(ss, layer, center=centers[layer])[0] for layer in layers]
            for j, s in enumerate(ss[1:], 1):
                gt.add_row(escape(g), escape(f"{s.meta.label} vs {ss[0].meta.label}"),
                           *[f"{sims[k][j]:.2f}" for k in range(len(layers))])
        console.print(gt)
    console.print(f"scans tagged [bold]{run_id}[/]: brainscan list --sweep {run_id}; "
                  "brainscan diff <a> <b>; brainscan show <scan> --find <token>")


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
