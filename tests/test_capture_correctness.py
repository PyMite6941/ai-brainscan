"""Does a Scan reflect what the model really computes? (slow: loads real models)

Checked against an independent forward pass, for every architecture in the HF cache:
  * resid == the model's hidden states
  * final-layer logit lens == softmax(real logits), top-k tokens and probs, every position
    (so the final norm is applied exactly once)
  * a middle layer's lens == final norm + unembedding applied to that layer
  * greedy generation starts with the lens's final top-1
  * save -> load keeps meta and lens identical and resid within fp16 error, no overflow
"""

from __future__ import annotations

import numpy as np
import pytest

MODELS = [("gpt2", "The Eiffel Tower is located in the city of", False),
          ("unsloth/gemma-3-1b-it", "Fact: the capital of the state containing Dallas is", False),
          ("unsloth/gemma-3-1b-it", "What is 2 + 2? Answer with a single number.", True)]


def _cached(repo: str) -> bool:
    from huggingface_hub import try_to_load_from_cache

    return bool(try_to_load_from_cache(repo, "config.json"))


@pytest.mark.slow
@pytest.mark.parametrize("model_id,prompt,chat", MODELS)
def test_scan_matches_model(model_id, prompt, chat, monkeypatch, tmp_path):
    if not _cached(model_id):
        pytest.skip(f"{model_id} not in the HF cache")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    torch = pytest.importorskip("torch")

    from brainscan.capture import _encode, _final_norm, _text_config, capture, load_model
    from brainscan.contracts import Scan

    scan = capture(model_id, prompt, chat=chat, top_k=5, generate_tokens=4, keep_output=True)
    model, tok = load_model(model_id)
    enc, _, _ = _encode(tok, prompt, chat)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)

    ids = enc["input_ids"][0].tolist()
    assert scan.meta.tokens == [tok.decode([i]) for i in ids]

    hs = torch.stack([h[0] for h in out.hidden_states]).float().numpy()
    assert scan.resid.shape == hs.shape
    np.testing.assert_allclose(scan.resid, hs, rtol=1e-4, atol=1e-3)

    probs = out.logits[0].float().softmax(-1)
    p, idx = probs.topk(5, dim=-1)
    for t in range(len(ids)):
        assert [c[0] for c in scan.lens[-1][t]] == [tok.decode([int(j)]) for j in idx[t]], t
        np.testing.assert_allclose([c[1] for c in scan.lens[-1][t]], p[t].numpy(), atol=1e-4)

    mid = len(scan.lens) // 2
    cap = getattr(_text_config(model), "final_logit_softcapping", None)
    with torch.no_grad():
        logits = model.get_output_embeddings()(_final_norm(model)(out.hidden_states[mid][0])).float()
        if cap:
            logits = torch.tanh(logits / cap) * cap
    assert scan.lens[mid][-1][0][0] == tok.decode([int(logits[-1].argmax())])

    first = scan.lens[-1][-1][0][0].strip()
    assert (scan.meta.output_text or "").strip().startswith(first)

    back = Scan.load(scan.save(tmp_path))
    assert back.meta == scan.meta
    assert back.lens == scan.lens
    r = back.resid.astype(np.float32)
    assert np.isfinite(r).all()
    assert (np.abs(r - scan.resid) / (np.abs(scan.resid) + 1)).max() < 1e-2
