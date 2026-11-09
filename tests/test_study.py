"""study.py: prompt sets, lens lookups, scan comparison (no model needed)."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from typer.testing import CliRunner

from brainscan import study
from brainscan.cli import app

runner = CliRunner()


def test_bundled_prompt_sets_load():
    sets = study.bundled_prompt_sets()
    names = {p.stem for p in sets}
    assert {"multihop", "framing", "language", "refusal_probe"} <= names
    for p in sets:
        items = study.load_prompt_set(p)
        assert items, p
        assert all(it.prompt for it in items)


def test_load_prompt_set_parses_fields(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text('# comment\n\n{"prompt": "hi", "label": "a", "group": "g", "chat": false, "expect": "x"}\n'
                 '{"messages": [{"role": "user", "content": "yo"}]}\n', encoding="utf-8")
    a, b = study.load_prompt_set(p)
    assert (a.prompt, a.label, a.group, a.chat, a.expect) == ("hi", "a", "g", False, "x")
    assert isinstance(b.prompt, list) and b.chat is None


def test_load_prompt_set_errors(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"label": "no prompt"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="needs 'prompt'"):
        study.load_prompt_set(p)
    with pytest.raises(FileNotFoundError, match="bundled sets"):
        study.resolve_prompt_set("does-not-exist")


def test_first_layer_and_settles(fake_scan):
    s = copy.deepcopy(fake_scan)
    # last position: put 'Austin' top-1 at layer 1 (noise), gone at 2, back from 3 to the end
    for layer, top in enumerate(["x", " Austin", "y", " austin", "Austin"]):
        s.lens[layer][-1] = [[top, 0.9], ["z", 0.1]]
    assert study.first_layer(s, "Austin") == 1
    assert study.settles_at(s, "Austin") == 3
    assert study.settles_at(s, "nope") is None
    assert study.token_prob_curve(s, "austin") == [0.0, 0.9, 0.0, 0.9, 0.9]


def test_diff_identical_and_centred(fake_scan):
    rows = study.diff(fake_scan, fake_scan)
    assert len(rows) == fake_scan.meta.n_layers + 1
    assert all(r.cosine == pytest.approx(1.0, abs=1e-5) for r in rows)
    assert rows[0].cosine_centred is None

    other = copy.deepcopy(fake_scan)
    other.resid = -other.resid
    center = np.zeros((5, 8), dtype=np.float32)
    rows = study.diff(fake_scan, other, center=center)
    assert all(r.cosine_centred == pytest.approx(-1.0, abs=1e-5) for r in rows)


def test_diff_rejects_other_models(fake_scan):
    other = copy.deepcopy(fake_scan)
    other.meta.d_model = 16
    with pytest.raises(ValueError, match="different models"):
        study.diff(fake_scan, other)


def test_similarity_matrix_centring(fake_scan):
    a, b = copy.deepcopy(fake_scan), copy.deepcopy(fake_scan)
    shared = np.full(8, 100.0, dtype=np.float32)          # a huge shared "outlier" component
    a.resid[2, -1] = shared + np.eye(8)[0]
    b.resid[2, -1] = shared + np.eye(8)[1]
    raw = study.similarity_matrix([a, b], layer=2)
    centred = study.similarity_matrix([a, b], layer=2, center=study.mean_vector([a, b], 2))
    assert raw[0, 1] > 0.99
    assert centred[0, 1] < 0


def test_cli_prompts_diff_and_find(fake_scan):
    r = runner.invoke(app, ["prompts"])
    assert r.exit_code == 0 and "multihop" in r.output

    r = runner.invoke(app, ["diff", fake_scan.meta.scan_id, "control"])  # label lookup works too
    assert r.exit_code == 0, r.output
    assert "cos raw" in r.output

    r = runner.invoke(app, ["show", "control", "--find", "tok-4-4-0"])
    assert r.exit_code == 0, r.output
    assert "first in top-3 at layer 4" in r.output


def test_cli_sweep_unknown_set():
    r = runner.invoke(app, ["sweep", "nope-not-a-set"])
    assert r.exit_code == 1
    assert "no prompt set" in r.output
