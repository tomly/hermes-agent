"""Tests for ``gateway.run._resolve_hermes_bin`` — the /update and /restart
re-exec argv. Regression coverage for the 2026-10-01 fleet incident: a gateway
that imports ``hermes_cli`` only through a runtime ``sys.path`` injection (bare
interpreter started via ``python -c``) must not re-exec the module form, whose
child starts with clean interpreter state and dies with ``ModuleNotFoundError:
No module named 'hermes_cli'``."""

from __future__ import annotations

import sys

from gateway import run as gateway_run
from hermes_cli import install_shape


def test_resolve_hermes_bin_injected_runtime_uses_path_wrapper(monkeypatch):
    """Not child-viable (runtime sys.path injection, nothing served by the
    interpreter) -> the ``hermes`` wrapper on PATH, which carries its own
    injection; the module argv would crash-loop every /restart."""
    import shutil

    monkeypatch.setattr(install_shape, "module_argv_child_viable", lambda: False)
    monkeypatch.setattr(shutil, "which", lambda name: "/Users/op/.local/bin/hermes")
    assert gateway_run._resolve_hermes_bin() == ["/Users/op/.local/bin/hermes"]


def test_resolve_hermes_bin_viable_module_form_beats_path(monkeypatch):
    """Child-viable -> the module form wins even when PATH offers a ``hermes``
    (#111569: a PATH-first lookup would re-exec an attacker-planted
    executable)."""
    import shutil

    monkeypatch.setattr(install_shape, "module_argv_child_viable", lambda: True)
    monkeypatch.setattr(shutil, "which", lambda name: "/tmp/attacker/hermes")
    assert gateway_run._resolve_hermes_bin() == [sys.executable, "-m", "hermes_cli.main"]
