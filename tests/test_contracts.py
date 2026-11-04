"""Tests for the on-disk contract (brainscan/contracts.py)."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np

from brainscan.contracts import (
    ModuleResult,
    Scan,
    ScanMeta,
    list_scans,
    sha256,
)

SHA256_ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def _meta(scan_id: str, created: str) -> ScanMeta:
    return ScanMeta(
        scan_id=scan_id,
        model_id="test-model",
        label="control",
        prompt="hello world",
        prompt_sha256=sha256("hello world"),
        chat_template=False,
        tokens=[f"token-{t}" for t in range(5)],
        n_layers=4,
        d_model=8,
        created=created,
        brainscan_version="0.1.0",
    )


def test_save_load_round_trip(fake_scan, tmp_path):
    out = fake_scan.save(tmp_path)
    loaded = Scan.load(out)
    assert loaded.meta == fake_scan.meta
    assert loaded.attn is None
    assert loaded.lens == fake_scan.lens
    assert np.allclose(loaded.resid, fake_scan.resid, rtol=1e-2, atol=1e-2)


def test_attn_round_trip(fake_scan, tmp_path):
    attn = np.random.default_rng(3).standard_normal((4, 2, 5, 5)).astype(np.float32)
    scan = Scan(
        meta=fake_scan.meta,
        resid=fake_scan.resid,
        lens=fake_scan.lens,
        attn=attn,
    )
    out = scan.save(tmp_path)
    loaded = Scan.load(out)
    assert loaded.meta.has_attn is True
    assert loaded.attn.shape == (4, 2, 5, 5)
    assert np.allclose(loaded.attn, attn, rtol=1e-2, atol=1e-2)


def test_resid_norm(fake_scan):
    norms = fake_scan.resid_norm()
    assert norms.shape == (5, 5)
    expected = np.linalg.norm(fake_scan.resid.astype(np.float32), axis=-1)
    np.testing.assert_allclose(norms, expected)


def test_project(fake_scan):
    e0 = np.zeros(8, dtype=np.float32)
    e0[0] = 1.0
    proj = fake_scan.project(e0)
    assert proj.shape == (5, 5)
    np.testing.assert_allclose(proj, fake_scan.resid[..., 0].astype(np.float32))

    per_layer = np.zeros((5, 8), dtype=np.float32)
    per_layer[:, 0] = 1.0
    proj = fake_scan.project(per_layer)
    assert proj.shape == (5, 5)
    np.testing.assert_allclose(proj, fake_scan.resid[..., 0].astype(np.float32))


def test_top_token(fake_scan):
    top = fake_scan.top_token()
    expected = [[cell[0][0] for cell in layer] for layer in fake_scan.lens]
    assert top == expected


def test_module_results_round_trip(fake_scan):
    result = ModuleResult(
        module="llm_guard",
        category="guardrail",
        verdict="pass",
        score=0.93,
        details={"scanner": "toxicity"},
    )
    fake_scan.save_module_result(result)
    assert fake_scan.module_results() == [result]


def test_module_results_unsaved_returns_empty():
    scan = Scan(
        meta=_meta("unsaved", "2026-01-01T00:00:00"),
        resid=np.zeros((1, 1, 1), dtype=np.float32),
        lens=[[]],
    )
    assert scan.module_results() == []


def test_list_scans_newest_first_skips_corrupt(tmp_path):
    (tmp_path / "older").mkdir()
    (tmp_path / "older" / "meta.json").write_text(
        json.dumps(asdict(_meta("older", "2026-01-01T00:00:00"))), encoding="utf-8"
    )
    (tmp_path / "newer").mkdir()
    (tmp_path / "newer" / "meta.json").write_text(
        json.dumps(asdict(_meta("newer", "2026-06-01T00:00:00"))), encoding="utf-8"
    )
    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    (corrupt / "meta.json").write_text("{not valid json", encoding="utf-8")

    metas = list_scans(tmp_path)
    assert [m.scan_id for m in metas] == ["newer", "older"]


def test_sha256_known_value():
    assert sha256("abc") == SHA256_ABC
