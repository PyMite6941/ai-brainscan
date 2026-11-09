"""Gradio web UI. `brainscan web` (add --share for a public link teammates can open).

Two halves, same seam as everywhere else:
  * Scan and Sweep run capture (slow, queued, one model kept in memory)
  * Explore and Compare only read artifacts/ and never touch the model

Explore is the layer-analysis view. Its "layer map" can be coloured four ways:
  * logit lens: the top-1 token each layer would output, coloured by confidence
  * layer update: how much each layer rewrites the residual (1 - cos with the layer below)
  * residual norm: the plainest "EEG"
  * p(token): one token's probability at every layer x position
"""

from __future__ import annotations

import datetime as dt

import gradio as gr
import numpy as np
import plotly.graph_objects as go

from brainscan import __version__, config
from brainscan.contracts import Scan, list_scans

LENS = "logit lens: top-1 token (colour = confidence)"
UPDATE = "layer update: how much each layer changes it"
NORM = "residual norm"
TOKEN = "p(token): probability of the tracked token"
BASE_MODES = [LENS, UPDATE, NORM, TOKEN]

PLOT_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"size": 12},
}

# Gradio picks light/dark from the OS; force dark so the plots always match.
FORCE_DARK = """() => {
  const u = new URL(window.location);
  if (u.searchParams.get('__theme') !== 'dark') { u.searchParams.set('__theme', 'dark'); window.location.replace(u); }
}"""

CSS = """
.bs-hint { opacity: .75; font-size: .9em; }
footer { display: none !important; }
"""


# ---------------------------------------------------------------- data helpers

def _directions() -> dict[str, np.ndarray]:
    try:
        from brainscan.analysis.directions import load_directions

        return load_directions()
    except Exception:  # noqa: BLE001 - Track A stub or no directions saved yet
        return {}


def _scan_choices() -> list[tuple[str, str]]:
    out = []
    for m in list_scans(config.artifacts_dir()):
        sw = m.extra.get("sweep") or {}
        tag = f"{sw.get('set')}/{sw.get('group') or '-'}/" if sw else ""
        out.append((f"{m.scan_id}  [{tag}{m.label or '-'}]  {m.model_id.split('/')[-1]}"
                    + (f"  ({m.verdict})" if m.verdict else ""), m.scan_id))
    return out


def _load(scan_id: str | None) -> Scan | None:
    if not scan_id:
        return None
    p = config.artifacts_dir() / scan_id
    return Scan.load(p) if p.exists() else None


def _tok(t: str) -> str:
    return t.replace("\n", "\\n").replace("<", "‹").replace(">", "›") or "∅"


def _q(t: str) -> str:
    """Token for display, escaped and quoted once (repr() would double the backslashes)."""
    return f"'{_tok(t)}'"


def _pos_choices(scan: Scan | None) -> list[tuple[str, int]]:
    if scan is None:
        return []
    return [(f"{i}: {_q(t)}", i) for i, t in enumerate(scan.meta.tokens)]


def _layer_update(scan: Scan) -> np.ndarray:
    """[L+1, T] 1 - cos(resid_l, resid_{l-1}); row 0 = 0. Where the model does its work."""
    r = scan.resid.astype(np.float32)
    a, b = r[1:], r[:-1]
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8)
    return np.vstack([np.zeros((1, r.shape[1]), np.float32), 1 - cos])


def _values(scan: Scan, mode: str, token: str, skip_first: bool) -> tuple[np.ndarray, str, list | None]:
    """[L+1, T] colour matrix, colourbar title, optional per-cell text."""
    from brainscan.study import token_prob_curve

    T = len(scan.meta.tokens)
    if mode == LENS:
        z = np.array([[cell[0][1] if cell else 0.0 for cell in layer] for layer in scan.lens])
        return z, "top-1 prob", scan.top_token()
    if mode == UPDATE:
        return _layer_update(scan), "1 − cos(prev layer)", None
    if mode == TOKEN:
        z = np.array([token_prob_curve(scan, token, p) for p in range(T)]).T if token.strip() \
            else np.zeros((scan.meta.n_layers + 1, T))
        return z, f"p({token!r})", None
    if mode in _directions():
        d = _directions()[mode]
        if d.shape[-1] == scan.meta.d_model:
            return scan.project(d), f"projection on {mode}", None
    v = scan.resid_norm()
    # scale each layer by its max over the visible tokens, so <bos> (an attention
    # sink with a huge norm) doesn't wash everything else out
    ref = v[:, 1:] if (skip_first and T > 1) else v
    return np.clip(v / (ref.max(axis=1, keepdims=True) + 1e-8), 0, 1), "norm / layer max", None


# ---------------------------------------------------------------- figures

def _empty(msg: str, height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**PLOT_LAYOUT, title=msg, height=height,
                      xaxis={"visible": False}, yaxis={"visible": False})
    return fig


def _zmax(mode: str, z: np.ndarray) -> float | None:
    if mode in (LENS, TOKEN):
        return 1.0
    if mode == UPDATE and z.shape[0] > 4:
        # layer 1 (leaving the embeddings) and the last layer (final norm) rewrite far
        # more than the middle; scale to the middle layers so their differences show
        return float(max(np.percentile(z[2:-1], 99), 1e-3))
    return None


def heatmap(scan: Scan | None, mode: str, token: str = "", skip_first: bool = True,
            highlight: int | None = None) -> go.Figure:
    if scan is None:
        return _empty("pick a scan")
    vals, ctitle, text = _values(scan, mode or LENS, token or "", skip_first)
    T = len(scan.meta.tokens)
    start = 1 if (skip_first and T > 1) else 0
    cols = list(range(start, T))
    x = [f"{i}:{_tok(scan.meta.tokens[i])}" for i in cols]
    z = vals[:, cols]
    top = scan.top_token()
    hover = [[f"layer {li} · {x[j]}<br>lens top-1: {_q(top[li][i])}<br>{ctitle}: {vals[li, i]:.3f}"
              for j, i in enumerate(cols)] for li in range(vals.shape[0])]
    tr = go.Heatmap(z=z, x=x, y=list(range(vals.shape[0])), colorscale="Viridis", hoverinfo="text",
                    hovertext=hover, colorbar={"title": {"text": ctitle, "side": "right"}},
                    zmin=0 if mode in (LENS, TOKEN, UPDATE) else None, zmax=_zmax(mode, z))
    if text is not None:
        tr.update(text=[[_tok(text[li][i])[:8] for i in cols] for li in range(vals.shape[0])],
                  texttemplate="%{text}", textfont={"size": 9})
    fig = go.Figure(tr)
    if highlight is not None and highlight in cols:
        j = cols.index(highlight)
        fig.add_vrect(x0=j - 0.5, x1=j + 0.5, line={"color": "#ff7f0e", "width": 2})
    fig.update_layout(
        **PLOT_LAYOUT,
        title=f"{scan.meta.label or scan.meta.scan_id} · {scan.meta.model_id.split('/')[-1]} · layer × token",
        yaxis={"title": "layer (0 = embeddings, top = output)", "autorange": True, "dtick": 2},
        xaxis={"title": None, "tickangle": -45, "automargin": True},
        height=max(460, 22 * vals.shape[0] + 180), margin={"l": 60, "r": 20, "t": 50, "b": 40},
    )
    return fig


def lens_table(scan: Scan | None, position: int | None) -> list[list]:
    if scan is None:
        return []
    T = len(scan.meta.tokens)
    pos = T - 1 if position is None else max(0, min(T - 1, int(position)))
    norms = scan.resid_norm()[:, pos]
    upd = _layer_update(scan)[:, pos]
    rows = []
    for layer in range(len(scan.lens) - 1, -1, -1):          # output at the top, like the map
        preds = [f"{_q(t)} {p:.2f}" for t, p in scan.lens[layer][pos]]
        preds += [""] * (5 - len(preds))
        rows.append([layer, round(float(norms[layer]), 1), round(float(upd[layer]), 3), *preds[:5]])
    return rows


def trace_plot(scan: Scan | None, position: int | None, token: str) -> go.Figure:
    """Through depth at one position: top-1 confidence, layer update, and p(tracked token)."""
    from brainscan.study import token_prob_curve

    if scan is None:
        return _empty("pick a scan")
    T = len(scan.meta.tokens)
    pos = T - 1 if position is None else int(position)
    layers = list(range(scan.meta.n_layers + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=layers, y=[c[pos][0][1] for c in scan.lens], name="top-1 confidence",
                             mode="lines+markers"))
    fig.add_trace(go.Scatter(x=layers, y=_layer_update(scan)[:, pos], name="layer update",
                             mode="lines+markers", line={"dash": "dot"}))
    if token.strip():
        fig.add_trace(go.Scatter(x=layers, y=token_prob_curve(scan, token, pos), name=f"p({token!r})",
                                 mode="lines+markers", line={"width": 3}))
    fig.update_layout(**PLOT_LAYOUT, height=320, xaxis_title="layer", yaxis={"range": [0, 1.02]},
                      title=f"through depth at token {pos}: {_q(scan.meta.tokens[pos])}",
                      legend={"orientation": "h", "y": -0.25})
    return fig


def depth_curve(scans: list[Scan | None], mode: str, token: str, position: int) -> go.Figure:
    fig = go.Figure()
    for s in scans:
        if s is None:
            continue
        vals, ctitle, _ = _values(s, mode, token, True)
        T = vals.shape[1]
        pos = max(0, min(T - 1, position if position >= 0 else T + position))
        fig.add_trace(go.Scatter(y=vals[:, pos], mode="lines+markers",
                                 name=f"{s.meta.label or s.meta.scan_id} ({_q(s.meta.tokens[pos])})"))
        fig.update_layout(yaxis_title=ctitle)
    fig.update_layout(**PLOT_LAYOUT, title="this colour mode through depth, one token per scan",
                      xaxis_title="layer", height=340)
    return fig


def similarity_curve(a: Scan | None, b: Scan | None, pos_a: int, pos_b: int) -> go.Figure:
    """cos(A, B) per layer, raw and centred on all saved scans of the same model."""
    from brainscan.study import baseline_mean, diff

    if a is None or b is None:
        return _empty("pick two scans of the same model")
    same = [Scan.load(config.artifacts_dir() / m.scan_id) for m in list_scans(config.artifacts_dir())
            if m.model_id == a.meta.model_id and m.d_model == a.meta.d_model][:200]
    try:
        rows = diff(a, b, pos_a, pos_b, center=baseline_mean(same))
    except ValueError as e:
        return _empty(str(e))
    x = [r.layer for r in rows]
    fig = go.Figure()
    if rows[0].cosine_centred is not None:
        fig.add_trace(go.Scatter(x=x, y=[r.cosine_centred for r in rows], mode="lines+markers",
                                 name=f"centred ({len(same)} scans)", line={"width": 3}))
    fig.add_trace(go.Scatter(x=x, y=[r.cosine for r in rows], name="raw", mode="lines",
                             line={"dash": "dot"}))
    fig.update_layout(**PLOT_LAYOUT, title="how alike are A and B at each depth?", xaxis_title="layer",
                      yaxis_title="cosine", height=340, legend={"orientation": "h", "y": -0.25})
    return fig


def token_curves(scans: list[Scan | None], token: str, position: int) -> go.Figure:
    from brainscan.study import token_prob_curve

    if not token.strip():
        return _empty("type a token to track (e.g. Austin, 4, 5)")
    fig = go.Figure()
    for s in scans:
        if s is not None:
            fig.add_trace(go.Scatter(y=token_prob_curve(s, token, position), mode="lines+markers",
                                     name=s.meta.label or s.meta.scan_id))
    fig.update_layout(**PLOT_LAYOUT, title=f"p({token!r}) through depth (logit lens)",
                      xaxis_title="layer", yaxis={"title": "probability", "range": [0, 1.02]},
                      height=340, legend={"orientation": "h", "y": -0.25})
    return fig


def _info(scan: Scan | None) -> dict:
    if scan is None:
        return {}
    m = scan.meta
    return {
        "scan_id": m.scan_id, "model": m.model_id, "label": m.label, "prompt": m.prompt,
        "tokens": len(m.tokens), "layers": m.n_layers, "d_model": m.d_model,
        "chat_template": m.chat_template, "verdict": m.verdict, "output": m.output_text,
        "sweep": m.extra.get("sweep"), "created": m.created,
        "modules": [r.__dict__ for r in scan.module_results()],
    }


def _model_choices() -> list[str]:
    try:
        from brainscan.models import local_models

        found = [m.repo_id for m in local_models() if m.scannable]
    except Exception:  # noqa: BLE001 - HF cache may be unreadable
        found = []
    d = config.default_model()
    return found if d in found else [d, *found]


def _modules_rows() -> list[list[str]]:
    from brainscan.modules import REGISTRY

    return [[m.info.category, m.info.key, m.info.name, m.info.license,
             "yes" if m.available() else m.install_hint(), m.info.what] for m in REGISTRY.values()]


def _prompt_sets() -> list[str]:
    from brainscan.study import bundled_prompt_sets

    return [p.stem for p in bundled_prompt_sets()]


def _model_loaded(model_id: str) -> bool:
    from brainscan.capture import _CACHE

    return any(k[0] == model_id for k in _CACHE)


# ---------------------------------------------------------------- app

def build() -> gr.Blocks:
    modes = [*BASE_MODES, *_directions().keys()]
    with gr.Blocks(title="ai-brainscan") as demo:
        gr.Markdown(f"## ai-brainscan `{__version__}` — watch the model think, layer by layer")

        with gr.Tabs() as tabs:
            # ---------------- Scan
            with gr.Tab("Scan", id="scan"):
                with gr.Row():
                    with gr.Column(scale=3):
                        prompt = gr.Textbox(label="prompt", lines=5,
                                            placeholder="What is the capital of the state containing Dallas?")
                        gr.Markdown("Runs one forward pass and saves every layer's activations. "
                                    "The first scan loads the model (about a minute on CPU); after "
                                    "that each scan takes a few seconds. You land in **Explore** when "
                                    "it is done.", elem_classes="bs-hint")
                    with gr.Column(scale=1):
                        model = gr.Dropdown(_model_choices(), value=config.default_model(), label="model",
                                            allow_custom_value=True)
                        label = gr.Textbox(label="label", placeholder="e.g. control / false-premise")
                        chat = gr.Checkbox(True, label="apply chat template (for -it models)")
                        gen = gr.Slider(0, 128, value=16, step=8,
                                        label="generate tokens (0 = forward pass only)")
                        keep = gr.Checkbox(False, label="save generated text (gitignored)")
                        go_btn = gr.Button("Run scan", variant="primary")
                status = gr.Markdown()

            # ---------------- Explore
            with gr.Tab("Explore", id="explore"):
                with gr.Row():
                    scan_dd = gr.Dropdown(_scan_choices(), label="scan", scale=4)
                    refresh = gr.Button("↻ refresh list", scale=0, min_width=120)
                with gr.Row():
                    mode = gr.Dropdown(modes, value=LENS, label="colour the layer map by", scale=3)
                    token = gr.Textbox(label="track a token (for p(token) + trace)", scale=2,
                                       placeholder="e.g. Texas")
                    skip_bos = gr.Checkbox(True, label="hide first token (‹bos›)", scale=1)
                plot = gr.Plot(label="layer map")
                with gr.Row():
                    with gr.Column(scale=1):
                        pos = gr.Dropdown([], label="token position to inspect", interactive=True)
                        trace = gr.Plot(label="trace")
                    with gr.Column(scale=2):
                        lens = gr.Dataframe(
                            headers=["layer", "|resid|", "update", "top-1", "top-2", "top-3", "top-4",
                                     "top-5"],
                            label="logit lens at this token: what each layer would say next",
                            wrap=True, max_height=520)
                with gr.Accordion("scan details", open=False):
                    info = gr.JSON()

            # ---------------- Compare
            with gr.Tab("Compare", id="compare"):
                with gr.Row():
                    a_dd = gr.Dropdown(_scan_choices(), label="scan A (e.g. neutral)")
                    b_dd = gr.Dropdown(_scan_choices(), label="scan B (e.g. false premise)")
                with gr.Row():
                    cpos = gr.Number(-1, label="token position (-1 = last)", precision=0, scale=1)
                    track = gr.Textbox(label="track a token (e.g. Austin, 4, 5)", scale=2)
                    cmode = gr.Dropdown(modes, value=UPDATE, label="colour the maps by", scale=2)
                with gr.Row():
                    sim = gr.Plot()
                    tcurve = gr.Plot()
                curve = gr.Plot()
                with gr.Row():
                    pa, pb = gr.Plot(), gr.Plot()

            # ---------------- Sweep
            with gr.Tab("Sweep", id="sweep"):
                gr.Markdown("Run a whole prompt set with the model loaded once. Variants in the same "
                            "**group** ask the same question in different ways; compare them in "
                            "**Compare**. Bundled sets: " + ", ".join(_prompt_sets())
                            + "; or a path to your own `.jsonl`.", elem_classes="bs-hint")
                with gr.Row():
                    sets = _prompt_sets()
                    set_dd = gr.Dropdown(sets, value=sets[0] if sets else None, label="prompt set",
                                         allow_custom_value=True, scale=2)
                    smodel = gr.Dropdown(_model_choices(), value=config.default_model(), label="model",
                                         allow_custom_value=True, scale=2)
                    sgen = gr.Slider(0, 64, value=0, step=8, label="generate tokens", scale=1)
                    sweep_btn = gr.Button("Run sweep", variant="primary", scale=0, min_width=140)
                sweep_status = gr.Markdown()
                sweep_df = gr.Dataframe(headers=["scan_id", "group", "label", "final top-1", "expect",
                                                 "expect settles (top-1)", "verdict"], wrap=True)

            # ---------------- Modules
            with gr.Tab("Modules", id="modules"):
                gr.Markdown("Mixture of modules. White-box reads activations; the rest judge "
                            "inputs/outputs.", elem_classes="bs-hint")
                gr.Dataframe(_modules_rows(), headers=["category", "key", "name", "licence", "installed",
                                                       "what"], wrap=True)

        # ---------------- wiring: Explore
        def on_scan(sid, md, tk, sb):
            s = _load(sid)
            last = len(s.meta.tokens) - 1 if s else None
            return (gr.update(choices=_pos_choices(s), value=last), heatmap(s, md, tk, sb, last),
                    lens_table(s, last), trace_plot(s, last, tk or ""), _info(s))

        scan_dd.change(on_scan, [scan_dd, mode, token, skip_bos], [pos, plot, lens, trace, info])

        def on_view(sid, md, tk, sb, p):
            s = _load(sid)
            return heatmap(s, md, tk, sb, p), lens_table(s, p), trace_plot(s, p, tk or "")

        for ev in (mode.change, skip_bos.change, pos.change, token.submit):
            ev(on_view, [scan_dd, mode, token, skip_bos, pos], [plot, lens, trace])

        def do_refresh():
            c = _scan_choices()
            return gr.update(choices=c), gr.update(choices=c), gr.update(choices=c)

        refresh.click(do_refresh, None, [scan_dd, a_dd, b_dd])

        # ---------------- wiring: Scan
        def run_scan(p, m, lab, ch, g, kp, progress=gr.Progress()):  # noqa: B008 - gradio idiom
            from brainscan.capture import capture

            if not (p or "").strip():
                return "enter a prompt", gr.update(), gr.update(), gr.update(), gr.update()
            progress(0.1, desc="running the forward pass" if _model_loaded(m)
                     else f"loading {m} (first scan only, ~1 min on CPU)")
            s = capture(m, p, label=lab, chat=ch, generate_tokens=int(g), keep_output=True)
            answer = s.meta.output_text
            if not kp:
                s.meta.output_text = None
            progress(0.9, desc="saving")
            s.save(config.artifacts_dir())
            c = _scan_choices()
            msg = f"saved **{s.meta.scan_id}** · {len(s.meta.tokens)} tokens × {s.meta.n_layers} layers"
            if answer is not None:
                msg += f"\n\n**model answered:** {answer.strip()!r} · verdict (heuristic): {s.meta.verdict}"
            return (msg, gr.update(choices=c, value=s.meta.scan_id), gr.update(choices=c),
                    gr.update(choices=c), gr.Tabs(selected="explore"))

        go_btn.click(run_scan, [prompt, model, label, chat, gen, keep],
                     [status, scan_dd, a_dd, b_dd, tabs], concurrency_limit=1)

        # ---------------- wiring: Compare
        def compare(a, b, md, p, tk):
            sa, sb = _load(a), _load(b)
            p = int(p if p is not None else -1)
            return (similarity_curve(sa, sb, p, p), token_curves([sa, sb], tk or "", p),
                    depth_curve([sa, sb], md, tk or "", p),
                    heatmap(sa, md, tk or ""), heatmap(sb, md, tk or ""))

        for ev in (a_dd.change, b_dd.change, cmode.change, cpos.submit, track.submit):
            ev(compare, [a_dd, b_dd, cmode, cpos, track], [sim, tcurve, curve, pa, pb])

        # ---------------- wiring: Sweep
        def run_sweep(name, m, g, progress=gr.Progress()):  # noqa: B008 - gradio idiom
            from brainscan.capture import capture
            from brainscan.study import load_prompt_set, resolve_prompt_set, settles_at

            try:
                path = resolve_prompt_set(name or "")
                items = load_prompt_set(path)
            except (FileNotFoundError, ValueError) as e:
                return f"**error:** {e}", [], gr.update(), gr.update(), gr.update()
            run_id = f"{path.stem}-{dt.datetime.now():%Y%m%d-%H%M%S}"
            if not _model_loaded(m):
                progress(0, desc=f"loading {m} (~1 min on CPU)")
            rows = []
            for i, it in enumerate(progress.tqdm(items, desc="scanning")):
                s = capture(m, it.prompt, label=it.label, chat=True if it.chat is None else it.chat,
                            generate_tokens=int(g))
                s.meta.extra["sweep"] = {"run_id": run_id, "set": path.stem, "index": i,
                                         "group": it.group, "expect": it.expect}
                s.save(config.artifacts_dir())
                onset = settles_at(s, it.expect, -1, 1) if it.expect else None
                rows.append([s.meta.scan_id, it.group, it.label, s.top_token()[-1][-1], it.expect or "",
                             "" if not it.expect else ("-" if onset is None else f"L{onset}"),
                             s.meta.verdict or ""])
            c = _scan_choices()
            msg = (f"sweep **{run_id}**: {len(rows)} scans saved. Open any of them in **Explore**, or "
                   "two from the same group in **Compare**.")
            return msg, rows, gr.update(choices=c), gr.update(choices=c), gr.update(choices=c)

        sweep_btn.click(run_sweep, [set_dd, smodel, sgen],
                        [sweep_status, sweep_df, scan_dd, a_dd, b_dd], concurrency_limit=1)
    return demo


def launch(host: str = "127.0.0.1", port: int = 7860, share: bool = False) -> None:
    build().queue(default_concurrency_limit=1).launch(
        server_name=host, server_port=port, share=share,
        theme=gr.themes.Default(primary_hue="orange", neutral_hue="slate"), css=CSS, js=FORCE_DARK,
    )
