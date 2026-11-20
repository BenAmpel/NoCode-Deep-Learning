"""run_healthcheck: a normal launch checks packages are installed without importing them."""
from __future__ import annotations

import importlib
import importlib.util

import pytest

from runtime import healthcheck


def _free_port() -> int:
    from runtime.runtime_setup import find_free_port

    return find_free_port(7950)


def test_normal_launch_does_not_import_required_packages(monkeypatch):
    imported: list[str] = []
    real_import = importlib.import_module

    def spy(name, *args, **kwargs):
        imported.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(healthcheck.importlib, "import_module", spy)
    messages = healthcheck.run_healthcheck(server_port=_free_port())
    assert "Core dependencies are installed" in messages
    assert not set(imported) & set(healthcheck.REQUIRED_IMPORTS)


def test_missing_package_is_named(monkeypatch):
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "torchaudio":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(healthcheck.importlib.util, "find_spec", fake_find_spec)
    with pytest.raises(RuntimeError, match="torchaudio"):
        healthcheck.run_healthcheck(server_port=_free_port())


def test_import_all_imports_every_package(monkeypatch):
    imported: list[str] = []
    real_import = importlib.import_module

    def spy(name, *args, **kwargs):
        imported.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(healthcheck.importlib, "import_module", spy)
    messages = healthcheck.run_healthcheck(server_port=_free_port(), import_all=True)
    assert "Core dependencies import successfully" in messages
    assert set(healthcheck.REQUIRED_IMPORTS) <= set(imported)
