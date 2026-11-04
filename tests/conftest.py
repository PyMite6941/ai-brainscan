"""Shared fixtures for the ai-brainscan test suite."""

from __future__ import annotations

import numpy as np
import pytest

from brainscan.contracts import Scan, ScanMeta, sha256


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    """Point BRAINSCAN_ARTIFACTS at a temp dir and return it."""
    monkeypatch.setenv("BRAINSCAN_ARTIFACTS", str(tmp_path))
    return tmp_path


@pytest.fixture
def fake_scan(artifacts):
    """Build a small Scan (n_layers=4, d_model=8, 5 tokens), save it, return it."""
    rng = np.random.default_rng(7)
    resid = rng.standard_normal((5, 5, 8)).astype(np.float32)
    lens = [
        [  # one layer
            [  # one token
                [f"tok-{l}-{t}-{k}", round(1.0 / (k + 1), 5)] for k in range(3)
            ]
            for t in range(5)
        ]
        for l in range(5)
    ]
    meta = ScanMeta(
        scan_id="scan-abcd1234",
        model_id="test-model",
        label="control",
        prompt="hello world",
        prompt_sha256=sha256("hello world"),
        chat_template=False,
        tokens=[f"token-{t}" for t in range(5)],
        n_layers=4,
        d_model=8,
        created="2026-01-01T00:00:00",
        brainscan_version="0.1.0",
    )
    scan = Scan(meta=meta, resid=resid, lens=lens, attn=None)
    scan.save(artifacts)
    return scan
