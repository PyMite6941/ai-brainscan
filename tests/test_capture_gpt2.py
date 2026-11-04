"""Real-model capture tests for gpt2 (marked slow)."""

from __future__ import annotations

import pytest

from brainscan.capture import capture

GPT2 = "gpt2"


def _gpt2_in_cache() -> bool:
    from huggingface_hub import try_to_load_from_cache

    return bool(try_to_load_from_cache(GPT2, "model.safetensors"))


@pytest.mark.slow
@pytest.mark.skipif(
    not _gpt2_in_cache(),
    reason="gpt2 weights are not in the Hugging Face cache",
)
def test_capture_gpt2_shapes_and_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    pytest.importorskip("torch")

    scan = capture(GPT2, "The capital of France is", chat=False, top_k=3)

    T = len(scan.meta.tokens)
    assert scan.meta.n_layers == 12
    assert scan.resid.shape == (13, T, 768)
    assert len(scan.lens) == 13
    assert all(len(cell) == 3 for layer in scan.lens for cell in layer)

    from brainscan.contracts import Scan

    out = scan.save(tmp_path)
    loaded = Scan.load(out)
    assert loaded.meta == scan.meta
    assert loaded.attn is None


@pytest.mark.slow
@pytest.mark.skipif(
    not _gpt2_in_cache(),
    reason="gpt2 weights are not in the Hugging Face cache",
)
def test_capture_gpt2_last_layer_matches_direct_model(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    pytest.importorskip("torch")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    scan = capture(GPT2, "The capital of France is", chat=False, top_k=3)

    model = AutoModelForCausalLM.from_pretrained(GPT2)
    tok = AutoTokenizer.from_pretrained(GPT2)
    enc = tok("The capital of France is", return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
    direct_top = tok.decode([int(logits[0, -1].argmax())])

    # Proves the logit lens applies the final norm exactly once at the last
    # layer (no double final-norm).
    assert scan.lens[-1][-1][0][0] == direct_top


@pytest.mark.slow
@pytest.mark.skipif(
    not _gpt2_in_cache(),
    reason="gpt2 weights are not in the Hugging Face cache",
)
def test_capture_gpt2_generate(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    pytest.importorskip("torch")

    scan = capture(
        GPT2,
        "The capital of France is",
        chat=False,
        top_k=3,
        generate_tokens=5,
    )

    assert scan.meta.output_sha256 is not None
    assert scan.meta.verdict in {"refused", "complied"}
    assert scan.meta.output_text is None

    scan2 = capture(
        GPT2,
        "The capital of France is",
        chat=False,
        top_k=3,
        generate_tokens=5,
        keep_output=True,
    )
    assert scan2.meta.output_text is not None
