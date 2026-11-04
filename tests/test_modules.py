"""Tests for the module registry (brainscan/modules/__init__.py, base.py)."""

from __future__ import annotations

import pytest

from brainscan.modules import CATEGORIES, REGISTRY, get
from brainscan.modules.base import RunContext


def test_registry_covers_all_categories():
    categories = {m.info.category for m in REGISTRY.values()}
    assert categories == set(CATEGORIES)
    assert all(m.info.category in CATEGORIES for m in REGISTRY.values())


def test_registry_keys_unique_and_match_info_key():
    keys = [m.info.key for m in REGISTRY.values()]
    assert len(keys) == len(set(keys))
    assert all(key == m.info.key for key, m in REGISTRY.items())


def test_available_returns_bool_and_never_raises():
    for module in REGISTRY.values():
        assert isinstance(module.available(), bool)


def test_native_available():
    # transformers is installed, so the native white-box module is available.
    assert REGISTRY["native"].available() is True


def test_module_without_adapter_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        REGISTRY["garak"].run(RunContext(prompt="x"))


def test_get_unknown_mentions_known_keys():
    with pytest.raises(KeyError) as exc_info:
        get("nope")
    message = str(exc_info.value)
    assert "nope" in message
    for key in REGISTRY:
        assert key in message
