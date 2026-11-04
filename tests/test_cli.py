"""CLI tests via typer.testing.CliRunner against brainscan.cli.app."""

from __future__ import annotations

from typer.testing import CliRunner

from brainscan.cli import app

runner = CliRunner()


def test_version_prints_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_modules_output_contains_whitebox_and_garak():
    result = runner.invoke(app, ["modules"])
    assert result.exit_code == 0
    assert "whitebox" in result.output
    assert "garak" in result.output


def test_list_shows_fake_scan(fake_scan):
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert fake_scan.meta.scan_id in result.output


def test_show_prefix_and_nonexistent(fake_scan):
    result = runner.invoke(app, ["show", fake_scan.meta.scan_id[:8]])
    assert result.exit_code == 0
    assert "layer" in result.output

    result = runner.invoke(app, ["show", "nonexistent"])
    assert result.exit_code != 0


def test_models_mentions_ollama(monkeypatch):
    from brainscan import models

    monkeypatch.setattr(models, "local_models", list)
    monkeypatch.setattr(models, "ollama_models", lambda: ["llama3:8b"])
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "Ollama" in result.output
