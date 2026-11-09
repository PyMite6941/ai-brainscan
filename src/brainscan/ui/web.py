"""Gradio web UI. `brainscan web` (add --share for a public link teammates can open).

Two halves, same seam as everywhere else:
  * the Scan tab runs capture (slow, queued, one model kept in memory)
  * every other tab only reads artifacts/ and never touches the model
"""

from __future__ import annotations

import gradio as gr
import numpy as np
import plotly.graph_objects as go

from brainscan import __version__, config
from brainscan.contracts import Scan, list_scans

NORM = "residual norm (per-layer scaled)"


def _directions() -> dict[str, np.ndarray]:
    try:
        from brainscan.analysis.directions import load_directions

        return load_directions()
    except Exception:  # noqa: BLE001 - directions dir may not exist yet
        return {}


def _scan_choices() -> list[tuple[str, str]]:
    return [
        (f"{m.scan_id}  [{m.label or '-'}]  {m.model_id.split('/')[-1]}"
         + (f"  ({m.verdict})" if m.verdict else ""), m.scan_id)
        for m in list_scans(config.artifacts_dir())
    ]


def _load(scan_id: str | None) -> Scan | None:
    if not scan_id:
        return None
    p = config.artifacts_dir() / scan_id
    return Scan.load(p) if p.exists() else None


def _values(scan: Scan, mode: str) -> tuple[np.ndarray, str]:
    """[L+1, T] matrix for the heatmap + colourbar title."""
    if mode != NORM:
        d = _directions().get(mode)
        if d is not None and d.shape[-1] == scan.meta.d_model:
            return scan.project(d), f"projection on {mode}"
    v = scan.resid_norm()
    return v / (v.max(axis=1, keepdims=True) + 1e-8), "norm / layer max"


def _tok(t: str) -> str:
    return t.replace("\n", "\\n") or "∅"


def heatmap(scan: Scan | None, mode: str) -> go.Figure:
    fig = go.Figure()
    if scan is None:
        fig.update_layout(title="pick a scan")
        return fig
    vals, ctitle = _values(scan, mode)
    toks = [f"{i}:{_tok(t)}" for i, t in enumerate(scan.meta.tokens)]
    top = scan.top_token()
    hover = [[f"layer {li}<br>{toks[ti]}<br>lens top-1: {_tok(top[li][ti])!r}<br>value {vals[li, ti]:.3f}"
              for ti in range(len(toks))] for li in range(vals.shape[0])]
    fig.add_trace(go.Heatmap(z=vals, x=toks, y=list(range(vals.shape[0])), colorscale="Viridis",
                             hoverinfo="text", text=hover, colorbar={"title": ctitle}))
    fig.update_layout(
        title=f"{scan.meta.model_id} · {scan.meta.label or 'unlabelled'} · layer × token",
        yaxis={"title": "layer (0 = embeddings)", "autorange": "reversed"},
        xaxis={"title": "token", "tickangle": -60},
        height=max(420, 16 * vals.shape[0] + 160), margin={"l": 60, "r": 20, "t": 50, "b": 120},
    )
    return fig


def lens_table(scan: Scan | None, position: int) -> list[list]:
    if scan is None:
        return []
    T = len(scan.meta.tokens)
    pos = max(0, min(T - 1, position if position >= 0 else T + position))
    norms = scan.resid_norm()[:, pos]
    rows = []
    for layer, cell in enumerate(scan.lens):
        preds = [f"{_tok(t)!r} {p:.2f}" for t, p in cell[pos]]
        rows.append([layer, round(float(norms[layer]), 1), *preds])
    return rows


def depth_curve(scans: list[Scan], mode: str, position: int) -> go.Figure:
    """The 'what we expect to see' chart: one value per layer at one token, per scan."""
    fig = go.Figure()
    for s in scans:
        if s is None:
            continue
        vals, ctitle = _values(s, mode)
        pos = position if position >= 0 else vals.shape[1] + position
        pos = max(0, min(vals.shape[1] - 1, pos))
        fig.add_trace(go.Scatter(y=vals[:, pos], mode="lines+markers",
                                 name=f"{s.meta.label or s.meta.scan_id} ({_tok(s.meta.tokens[pos])!r})"))
        fig.update_layout(yaxis_title=ctitle)
    fig.update_layout(title="signal through depth at one token position", xaxis_title="layer", height=380)
    return fig


def similarity_curve(a: Scan | None, b: Scan | None, pos_a: int, pos_b: int) -> go.Figure:
    """cos(A, B) per layer, raw and centred on all saved scans of the same model."""
    from brainscan.study import baseline_mean, diff

    fig = go.Figure()
    if a is None or b is None:
        fig.update_layout(title="pick two scans of the same model", height=320)
        return fig
    same = [Scan.load(config.artifacts_dir() / m.scan_id) for m in list_scans(config.artifacts_dir())
            if m.model_id == a.meta.model_id and m.d_model == a.meta.d_model][:200]
    try:
        rows = diff(a, b, pos_a, pos_b, center=baseline_mean(same))
    except ValueError as e:
        fig.update_layout(title=str(e), height=320)
        return fig
    x = [r.layer for r in rows]
    fig.add_trace(go.Scatter(x=x, y=[r.cosine for r in rows], name="raw", mode="lines+markers"))
    if rows[0].cosine_centred is not None:
        fig.add_trace(go.Scatter(x=x, y=[r.cosine_centred for r in rows], mode="lines+markers",
                                 name=f"centred on {len(same)} scans"))
    fig.update_layout(title="how alike are A and B at each depth? (cosine, chosen positions)",
                      xaxis_title="layer", yaxis_title="cosine", height=340)
    return fig


def token_curves(scans: list[Scan | None], token: str, position: int) -> go.Figure:
    """Logit-lens probability of one token through depth, per scan."""
    from brainscan.study import token_prob_curve

    fig = go.Figure()
    if not token.strip():
        fig.update_layout(title="type a token to track (e.g. Austin)", height=320)
        return fig
    for s in scans:
        if s is None:
            continue
        fig.add_trace(go.Scatter(y=token_prob_curve(s, token, position), mode="lines+markers",
                                 name=s.meta.label or s.meta.scan_id))
    fig.update_layout(title=f"p({token!r}) through depth (logit lens; 0 = outside saved top-k)",
                      xaxis_title="layer", yaxis_title="probability", height=340)
    return fig


def _info(scan: Scan | None) -> dict:
    if scan is None:
        return {}
    m = scan.meta
    return {
        "model": m.model_id, "label": m.label, "tokens": len(m.tokens), "layers": m.n_layers,
        "d_model": m.d_model, "chat_template": m.chat_template, "verdict": m.verdict,
        "verdict_method": m.extra.get("verdict_method"), "prompt_sha256": m.prompt_sha256,
        "output_sha256": m.output_sha256, "created": m.created,
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


def build() -> gr.Blocks:
    mode_choices = [NORM, *_directions().keys()]
    with gr.Blocks(title="ai-brainscan") as demo:
        gr.Markdown(f"## ai-brainscan `{__version__}` — watch the model think, layer by layer")

        with gr.Tab("Scan"):
            with gr.Row():
                prompt = gr.Textbox(label="prompt", lines=4, scale=3,
                                    placeholder="The capital of the state containing Dallas is")
                with gr.Column(scale=1):
                    model = gr.Dropdown(_model_choices(), value=config.default_model(), label="model",
                                        allow_custom_value=True)
                    label = gr.Textbox(label="label", placeholder="refused / jailbreak:roleplay / control")
                    chat = gr.Checkbox(True, label="apply chat template")
                    gen = gr.Slider(0, 128, value=0, step=8, label="generate tokens (0 = forward pass only)")
                    go_btn = gr.Button("Run scan", variant="primary")
            status = gr.Markdown()

        with gr.Tab("Explore"):
            with gr.Row():
                scan_dd = gr.Dropdown(_scan_choices(), label="scan", scale=3)
                mode = gr.Dropdown(mode_choices, value=NORM, label="colour by", scale=2)
                refresh = gr.Button("↻", scale=0)
            plot = gr.Plot()
            pos = gr.Slider(-64, 63, value=-1, step=1, label="token position for the logit lens (-1 = last)")
            lens = gr.Dataframe(headers=["layer", "|resid|", "top-1", "top-2", "top-3", "top-4", "top-5"],
                                label="logit lens: what it is about to say, at each depth")
            info = gr.JSON(label="scan")

        with gr.Tab("Compare"):
            with gr.Row():
                a_dd = gr.Dropdown(_scan_choices(), label="scan A (e.g. refused)")
                b_dd = gr.Dropdown(_scan_choices(), label="scan B (e.g. jailbroken)")
                cmode = gr.Dropdown(mode_choices, value=NORM, label="colour by")
            with gr.Row():
                cpos = gr.Slider(-64, 63, value=-1, step=1, label="token position (-1 = last)", scale=3)
                track = gr.Textbox(label="track a token (e.g. Austin, 4, 5)", scale=1)
            with gr.Row():
                sim = gr.Plot()
                tcurve = gr.Plot()
            curve = gr.Plot()
            with gr.Row():
                pa, pb = gr.Plot(), gr.Plot()

        with gr.Tab("Sweep"):
            gr.Markdown("Run a whole prompt set with the model loaded once. Variants in the same "
                        "**group** are compared. Bundled sets: multihop, framing, language, "
                        "refusal_probe; or give a path to your own `.jsonl`.")
            with gr.Row():
                set_dd = gr.Dropdown(_prompt_sets(), value=(_prompt_sets() or [None])[0],
                                     label="prompt set", allow_custom_value=True, scale=2)
                smodel = gr.Dropdown(_model_choices(), value=config.default_model(), label="model",
                                     allow_custom_value=True, scale=2)
                sgen = gr.Slider(0, 64, value=0, step=8, label="generate tokens", scale=1)
                sweep_btn = gr.Button("Run sweep", variant="primary", scale=0)
            sweep_status = gr.Markdown()
            sweep_df = gr.Dataframe(headers=["scan_id", "group", "label", "final top-1", "expect",
                                             "expect settles (top-1)", "verdict"],
                                    wrap=True)

        with gr.Tab("Modules"):
            gr.Markdown("Mixture of modules. White-box reads activations; the rest judge inputs/outputs.")
            gr.Dataframe(_modules_rows(), headers=["category", "key", "name", "licence", "installed", "what"],
                         wrap=True)

        # ---- wiring ----
        def run_scan(p, m, lab, ch, g):
            from brainscan.capture import capture

            if not p.strip():
                return "enter a prompt", gr.update(), gr.update(), gr.update()
            s = capture(m, p, label=lab, chat=ch, generate_tokens=int(g))
            s.save(config.artifacts_dir())
            choices = _scan_choices()
            msg = f"saved **{s.meta.scan_id}** · {len(s.meta.tokens)} tokens × {s.meta.n_layers} layers"
            if s.meta.verdict:
                msg += f" · verdict (heuristic): **{s.meta.verdict}**"
            upd = gr.update(choices=choices)
            return msg + " · open the Explore tab", gr.update(choices=choices, value=s.meta.scan_id), upd, upd

        go_btn.click(run_scan, [prompt, model, label, chat, gen], [status, scan_dd, a_dd, b_dd],
                     concurrency_limit=1)

        def explore(sid, md, p):
            s = _load(sid)
            return heatmap(s, md), lens_table(s, int(p)), _info(s)

        for ev in (scan_dd.change, mode.change, pos.release):
            ev(explore, [scan_dd, mode, pos], [plot, lens, info])

        def do_refresh():
            c = _scan_choices()
            return gr.update(choices=c), gr.update(choices=c), gr.update(choices=c)

        refresh.click(do_refresh, None, [scan_dd, a_dd, b_dd])

        def compare(a, b, md, p, tk):
            sa, sb = _load(a), _load(b)
            p = int(p)
            return (similarity_curve(sa, sb, p, p), token_curves([sa, sb], tk or "", p),
                    depth_curve([sa, sb], md, p), heatmap(sa, md), heatmap(sb, md))

        for ev in (a_dd.change, b_dd.change, cmode.change, cpos.release, track.submit):
            ev(compare, [a_dd, b_dd, cmode, cpos, track], [sim, tcurve, curve, pa, pb])

        def run_sweep(name, m, g, progress=gr.Progress()):  # noqa: B008 - gradio idiom
            from brainscan.capture import capture
            from brainscan.study import load_prompt_set, resolve_prompt_set, settles_at

            try:
                path = resolve_prompt_set(name or "")
                items = load_prompt_set(path)
            except (FileNotFoundError, ValueError) as e:
                return f"**error:** {e}", [], gr.update(), gr.update(), gr.update()
            import datetime as dt

            run_id = f"{path.stem}-{dt.datetime.now():%Y%m%d-%H%M%S}"
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
            msg = (f"sweep **{run_id}**: {len(rows)} scans saved. Pick any two in **Compare** "
                   "(same group = same question, different input).")
            return msg, rows, gr.update(choices=c), gr.update(choices=c), gr.update(choices=c)

        sweep_btn.click(run_sweep, [set_dd, smodel, sgen],
                        [sweep_status, sweep_df, scan_dd, a_dd, b_dd], concurrency_limit=1)
    return demo


def _prompt_sets() -> list[str]:
    from brainscan.study import bundled_prompt_sets

    return [p.stem for p in bundled_prompt_sets()]


def launch(host: str = "127.0.0.1", port: int = 7860, share: bool = False) -> None:
    build().queue(default_concurrency_limit=1).launch(server_name=host, server_port=port, share=share)
