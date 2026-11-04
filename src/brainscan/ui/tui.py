"""Read-only terminal viewer of saved scans (Textual).

Never loads a model or imports torch/transformers: it only reads artifacts/
and should start instantly. Think "EEG in a terminal".
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Label,
    ListItem,
    ListView,
    Static,
    TabbedContent,
    TabPane,
)

from brainscan import config
from brainscan.analysis.directions import load_directions
from brainscan.contracts import Scan, list_scans

# 8-step sequential colour ramp, dark -> bright (ColorBrewer Blues subset).
RAMP = (
    "#08306b",
    "#08519c",
    "#2171b5",
    "#4292c6",
    "#6baed6",
    "#9ecae1",
    "#c6dbef",
    "#deebf7",
)
CELL_WIDTH = 6  # header cell width and cell field width, in characters


class BrainscanTUI(App):
    """App: a scan list on the left, three read-only tabs on the right."""

    TITLE = "ai-brainscan — scan viewer"

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("q", "quit", "quit"),
        Binding("r", "reload", "reload scan list"),
        Binding("d", "cycle_direction", "cycle direction"),
        Binding("left", "prev_token", "previous token"),
        Binding("right", "next_token", "next token"),
        Binding("h", "prev_token", "previous token", show=False),
        Binding("l", "next_token", "next token", show=False),
    ]

    CSS = """
#left { width: 58; }
#right { width: 1fr; }
#heatmap { height: 1fr; }
#lens-table { height: 1fr; }
"""

    def __init__(self) -> None:
        super().__init__()
        self.directions: dict[str, np.ndarray] = {}
        self._direction: np.ndarray | None = None
        self._mode_index = 0  # 0 == "none" (residual norm mode)
        self._position = 0
        self._selected_id: str | None = None
        self._ids: list[str] = []  # list row -> scan_id (scan ids start with digits: invalid widget ids)
        self._cache: tuple[str, Scan] | None = None

    # ---------------------------------------------------------------- setup

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            with VerticalScroll(id="left"):
                yield Label("saved scans", id="list-label")
                yield ListView(id="scan-list")
                yield Label(
                    'No scans found.\nRun: brainscan capture "your prompt"',
                    id="empty",
                )
            with VerticalScroll(id="right"), TabbedContent(id="tabs"):
                    with TabPane("Heatmap", id="tab-heatmap"):
                        yield VerticalScroll(Static(id="heatmap"))
                    with TabPane("Logit lens", id="tab-lens"):
                        yield VerticalScroll(
                            Static(id="pos-label"), DataTable(id="lens-table")
                        )
                    with TabPane("Info", id="tab-info"):
                        yield VerticalScroll(Static(id="info"))
        yield Footer()

    def on_mount(self) -> None:
        try:
            self.directions = load_directions()
        except Exception:  # noqa: BLE001 - directory may not exist; treat as empty
            self.directions = {}
        self._reload_list()
        self.query_one("#scan-list", ListView).focus()
        self._refresh()

    # ---------------------------------------------------------------- scan list

    def _reload_list(self) -> None:
        lst = self.query_one("#scan-list", ListView)
        empty = self.query_one("#empty", Label)
        lst.clear()
        keep = self._selected_id
        metas = list_scans(config.artifacts_dir())
        self._ids = [m.scan_id for m in metas]
        self._cache = None
        for m in metas:
            short = m.model_id.split("/")[-1]
            text = f"{m.scan_id[:15]}  {m.label or '-'}  {short}  {m.verdict or '-'}"
            lst.append(ListItem(Label(text)))
        if self._ids:
            empty.display = False
            lst.display = True
            if keep in self._ids:
                lst.index = self._ids.index(keep)
            else:
                lst.index = 0
                self._select(0)
        else:
            empty.display = True
            lst.display = False
            self._selected_id = None

    def action_reload(self) -> None:
        self._reload_list()
        try:
            self.directions = load_directions()
        except Exception:  # noqa: BLE001 - directory may not exist; treat as empty
            self.directions = {}
        self._refresh()

    @on(ListView.Highlighted, "#scan-list")
    @on(ListView.Selected, "#scan-list")
    def on_list_view_selected(self, event: ListView.Highlighted | ListView.Selected) -> None:
        idx = event.list_view.index
        if idx is None or idx >= len(self._ids) or self._ids[idx] == self._selected_id:
            return
        self._select(idx)
        self._refresh()

    def _select(self, idx: int) -> None:
        self._selected_id = self._ids[idx]
        scan = self._load_selected()
        self._position = max(0, len(scan.meta.tokens) - 1) if scan else 0

    # ---------------------------------------------------------------- directions / token position

    def action_cycle_direction(self) -> None:
        modes = ["none", *sorted(self.directions)]
        self._mode_index = (self._mode_index + 1) % len(modes)
        self._apply_direction(modes)

    def _apply_direction(self, modes: list[str]) -> None:
        mode = modes[self._mode_index]
        scan = self._load_selected()
        self._direction = None
        if mode != "none":
            direction = self.directions.get(mode)
            if scan is None or direction is None or direction.shape[-1] != scan.meta.d_model:
                self.notify(
                    f"direction {mode} shape mismatch; using residual norm",
                    severity="error",
                    timeout=3,
                )
                self._mode_index = 0
                self._refresh()
                return
            self._direction = direction
        self._refresh()

    def action_prev_token(self) -> None:
        scan = self._load_selected()
        if scan is None:
            return
        self._position = max(0, self._position - 1)
        self._refresh()

    def action_next_token(self) -> None:
        scan = self._load_selected()
        if scan is None:
            return
        self._position = min(len(scan.meta.tokens) - 1, self._position + 1)
        self._refresh()

    # ---------------------------------------------------------------- data access

    def _load_selected(self) -> Scan | None:
        if not self._selected_id:
            return None
        if self._cache and self._cache[0] == self._selected_id:
            return self._cache[1]
        p = config.artifacts_dir() / self._selected_id
        if not p.exists():
            return None
        scan = Scan.load(p)
        self._cache = (self._selected_id, scan)
        return scan

    def _modes(self) -> list[str]:
        return ["none", *sorted(self.directions)]

    # ---------------------------------------------------------------- heatmap

    def _refresh(self) -> None:
        scan = self._load_selected()
        self._refresh_heatmap(scan)
        self._refresh_lens()
        self._refresh_info(scan)

    def _refresh_heatmap(self, scan: Scan | None) -> None:
        static = self.query_one("#heatmap", Static)
        if scan is None:
            static.update("select a scan from the list")
            return
        width = max(8, static.size.width or 80)
        n = max(1, (width - 8) // CELL_WIDTH)
        tokens = scan.meta.tokens
        total = len(tokens)
        start = max(0, total - n)  # show the LAST N tokens that fit
        shown = tokens[start:]

        values, title = self._heatmap_values(scan)

        out = Text()
        out.append("    ")
        for tok in shown:
            head = tok.replace("\n", "\\n")[:CELL_WIDTH].ljust(CELL_WIDTH)
            out.append(head, style="bold")
        out.append("\n")
        out.append("    ")
        for ti in range(len(shown)):
            mark = "  ▼  " if start + ti == self._position else "      "
            out.append(mark, style="dim")
        out.append("\n")
        for li in range(values.shape[0]):
            out.append(f"{li:3d} ")
            row = values[li]
            norm = row / (row.max() + 1e-8)
            for ti in range(len(shown)):
                v = float(norm[start + ti])
                idx = min(int(v * (len(RAMP) - 1 + 1e-9)), len(RAMP) - 1)
                out.append("██████", style=f"{RAMP[idx]} on {RAMP[idx]}")
            out.append("\n")
        window = values[:, start:]
        out.append("legend: ")
        for colour in RAMP:
            out.append("██", style=f"{colour} on {colour}")
        out.append(
            f"  low -> high  ({float(window.min()):.2f} .. {float(window.max()):.2f})  {title}\n"
        )
        static.update(out)

    def _heatmap_values(self, scan: Scan) -> tuple[np.ndarray, str]:
        if self._direction is not None:
            return scan.project(self._direction), "projection"
        return scan.resid_norm(), "residual norm"

    # ---------------------------------------------------------------- logit lens

    def _refresh_lens(self) -> None:
        scan = self._load_selected()
        table = self.query_one("#lens-table", DataTable)
        label = self.query_one("#pos-label", Static)
        if scan is None:
            label.update("no scan selected")
            table.clear(columns=True)
            return
        tokens = scan.meta.tokens
        pos = max(0, min(self._position, len(tokens) - 1))
        label.update(f"position {pos} = {tokens[pos]!r}")
        norms = scan.resid_norm()
        table.clear(columns=True)
        table.add_columns("layer", "|resid|", "top-1", "p", "top-2..k")
        for layer in range(len(scan.lens)):
            cands = scan.lens[layer][pos]
            top = cands[0] if cands else ("", 0.0)
            rest = " ".join(f"{t} {p:.2f}" for t, p in cands[1:])
            table.add_row(
                str(layer),
                f"{float(norms[layer, pos]):.2f}",
                top[0].replace("\n", "\\n"),
                f"{top[1]:.3f}",
                rest,
            )

    # ---------------------------------------------------------------- info

    def _refresh_info(self, scan: Scan | None) -> None:
        static = self.query_one("#info", Static)
        if scan is None:
            static.update("no scan selected")
            return
        m = scan.meta
        lines = [
            f"model: {m.model_id}",
            f"label: {m.label or '-'}",
            f"n_layers: {m.n_layers}",
            f"d_model: {m.d_model}",
            f"chat_template: {m.chat_template}",
            f"created: {m.created}",
            f"prompt_sha256: {m.prompt_sha256}",
            f"verdict: {m.verdict or '-'}",
            f"output_sha256: {m.output_sha256 or '-'}",
            "",
            "prompt (first 500 chars):",
            m.prompt[:500] + ("…" if len(m.prompt) > 500 else ""),
        ]
        results = scan.module_results()
        if results:
            lines.append("")
            lines.append(f"module results ({len(results)}):")
            for r in results:
                lines.append(
                    f"  {r.module} [{r.category}]: verdict={r.verdict} score={r.score}"
                )
                for key, value in r.details.items():
                    lines.append(f"    {key}: {value}")
        if m.output_text is not None:
            lines.append("")
            lines.append("output_text:")
            lines.append(m.output_text[:500])
        static.update("\n".join(lines))

    def action_quit(self) -> None:
        self.exit()


def run() -> None:
    BrainscanTUI().run()
