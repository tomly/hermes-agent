"""Tests for ``hermes_cli.install_shape`` — whether a fresh child interpreter of
the running one can import ``hermes_cli`` (the module-argv viability question
behind the 2026-10-01 kanban worker crash fleet incident)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from hermes_cli import install_shape


def _fake_spec(package_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        origin=str(package_dir / "__init__.py"),
        submodule_search_locations=None,
    )


def _inject_source_tree(tmp_path: Path) -> Path:
    package_dir = tmp_path / "hermes-agent" / "hermes_cli"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    return package_dir


def test_module_argv_viable_for_editable_install(tmp_path, monkeypatch):
    """PEP 660 editable install: ``find_spec`` resolves into the source tree,
    but the ``hermes_agent`` dist-info in the interpreter's own site-packages
    serves the package to children (``__editable__`` finder / ``.pth``). The
    module form must stay viable there — dropping to PATH would let whatever
    ``hermes`` names shadow the running install (#111569)."""
    package_dir = _inject_source_tree(tmp_path)
    site_dir = tmp_path / "site-packages"
    dist_info = site_dir / "hermes_agent-1.0.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: hermes-agent\nVersion: 1.0\n", encoding="utf-8")

    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: _fake_spec(package_dir))
    monkeypatch.setattr(install_shape, "_interpreter_package_roots", lambda: [site_dir])
    # cwd is not the package root: only the installed dist can carry viability.
    monkeypatch.chdir(tmp_path)
    assert install_shape.module_argv_child_viable() is True


def test_module_argv_not_viable_for_bare_injected_runtime(tmp_path, monkeypatch):
    """The fleet-incident shape: the package reaches the running process only
    through a runtime ``sys.path`` insert — no site dir contains it, no
    ``hermes_agent`` distribution is installed, cwd is elsewhere. Children
    spawned as ``sys.executable -m hermes_cli.main`` would die with
    ``ModuleNotFoundError``, so the resolver must see NOT viable and fall back
    to the ``hermes`` wrapper on PATH."""
    package_dir = _inject_source_tree(tmp_path)
    empty_site = tmp_path / "site-packages"
    empty_site.mkdir()

    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: _fake_spec(package_dir))
    monkeypatch.setattr(install_shape, "_interpreter_package_roots", lambda: [empty_site])
    monkeypatch.chdir(tmp_path)
    assert install_shape.module_argv_child_viable() is False


def test_module_argv_viable_when_cwd_is_package_root(tmp_path, monkeypatch):
    """Source-tree launch (``python -m hermes_cli.main`` / pytest from the
    checkout root): the process imported the package through cwd, the
    historical module-form preference holds."""
    package_dir = _inject_source_tree(tmp_path)
    empty_site = tmp_path / "site-packages"
    empty_site.mkdir()

    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: _fake_spec(package_dir))
    monkeypatch.setattr(install_shape, "_interpreter_package_roots", lambda: [empty_site])
    monkeypatch.chdir(package_dir.parent)
    assert install_shape.module_argv_child_viable() is True


def test_module_argv_viable_for_site_packages_install(tmp_path, monkeypatch):
    """Real install: the package root lives inside the interpreter's
    site-packages, which every child serves again."""
    site_dir = tmp_path / "site-packages"
    package_dir = site_dir / "hermes_cli"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")

    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: _fake_spec(package_dir))
    monkeypatch.setattr(install_shape, "_interpreter_package_roots", lambda: [site_dir])
    monkeypatch.chdir(tmp_path)
    assert install_shape.module_argv_child_viable() is True
