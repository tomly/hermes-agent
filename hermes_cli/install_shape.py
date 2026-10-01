"""Can a fresh child interpreter import the running ``hermes_cli`` install?

``find_spec("hermes_cli")`` inside a long-running process speaks only for THAT
process. A gateway launched by a bare interpreter via ``python -c`` plus an
in-script ``sys.path.insert`` imports the package fine, but a child spawned as
``sys.executable -m hermes_cli.main`` starts with clean interpreter state — the
source root is neither in its site-packages nor re-injected, its cwd is the task
workspace, and Hermes-owned PYTHONPATH entries are stripped from spawn envs
(:mod:`tools.environments.local_pythonpath`) — so it dies with
``ModuleNotFoundError: No module named 'hermes_cli'`` (fleet incident
2026-10-01: 7 kanban tasks circuit-broke to ``blocked`` on consecutive worker
crashes; the corrected root cause behind #1243's symptom). Resolvers must treat
such a runtime as NOT module-viable and fall back to the ``hermes`` wrapper on
PATH, which carries its own path injection.

Module viability: the package root sits in THIS interpreter's own package
directories (a real install — the child's site processing serves it again), the
process cwd IS the package root (a source-tree launch — ``python -m
hermes_cli.main``, pytest from the checkout root — that imported the package
through that cwd and keeps the historical module-form preference), or a
``hermes_agent`` distribution is installed in those directories (PEP 660
editable installs: ``find_spec`` resolves into the source tree, but the child
reaches it through the site hook). Module argv stays the FIRST choice whenever
it is viable; PATH is consulted only when it is not — a PATH-first lookup lets
an attacker-planted ``hermes`` shadow the running install (#111569).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _interpreter_package_roots() -> "list[Path]":
    """Package directories this interpreter serves to every child it spawns."""
    roots: "list[Path]" = []
    try:
        import sysconfig
        for key in ("purelib", "platlib"):
            value = sysconfig.get_paths().get(key)
            if value:
                roots.append(Path(value).resolve())
    except Exception:
        pass
    try:
        import site
        for value in list(site.getsitepackages()):
            if value:
                roots.append(Path(value).resolve())
        usersite = site.getusersitepackages()
        if usersite:
            roots.append(Path(usersite).resolve())
    except Exception:
        pass
    deduped: "list[Path]" = []
    for root in roots:
        if root not in deduped:
            deduped.append(root)
    return deduped


def _package_root_from_spec(spec) -> Optional[Path]:
    """Directory CONTAINING the package, from its import spec. ``None`` when the
    spec carries no usable filesystem location (frozen/zip importers, mock
    specs): the caller fails open to the historical module-form answer — only a
    resolvable source-tree root outside the served roots proves non-viability."""
    origin = getattr(spec, "origin", None)
    if isinstance(origin, str) and origin:
        # .../<root>/hermes_cli/__init__.py -> <root>
        return Path(origin).resolve().parent.parent
    locations = getattr(spec, "submodule_search_locations", None)
    if locations:
        first = next(iter(locations), None)
        if isinstance(first, str) and first:
            return Path(first).resolve().parent
    return None


def _hermes_dist_installed_in(site_roots: "list[Path]") -> bool:
    """True when a ``hermes_agent`` distribution lives in the interpreter's own
    site directories — including PEP 660 editable installs, whose dist-info is
    a real site-packages entry even though ``find_spec`` resolves into the
    source tree. A distribution there means the child's site processing (direct
    install, ``__editable__`` finder, or legacy ``.pth``) will serve the
    package again."""
    import importlib.metadata

    existing = [str(root) for root in site_roots if root.is_dir()]
    if not existing:
        return False
    try:
        for dist in importlib.metadata.distributions(path=existing):
            name = (dist.metadata.get("Name") or "").strip().lower().replace("_", "-")
            if name == "hermes-agent":
                return True
    except Exception:
        return False
    return False


def module_argv_child_viable() -> bool:
    """True when ``[sys.executable, "-m", "hermes_cli.main"]`` will import in a
    spawned child. Cheap and static — import-spec path provenance compared
    against the interpreter's package directories and the process cwd, never a
    per-spawn subprocess probe.

    A ``find_spec`` failure answers NOT importable, exactly as the pre-check
    resolvers' ``try/except: pass`` did. But a spec whose origin cannot be
    resolved to a filesystem path (frozen/zip importers) fails OPEN to the
    historical module-form answer: rerouting to PATH happens only on PROOF of
    non-viability (a resolvable package root that no served root covers, cwd
    does not match, and no ``hermes_agent`` distribution is installed)."""
    import importlib.util

    try:
        spec = importlib.util.find_spec("hermes_cli")
    except Exception:
        return False
    if spec is None:
        return False
    try:
        package_root = _package_root_from_spec(spec)
    except Exception:
        return True
    if package_root is None:
        return True
    served = _interpreter_package_roots()
    if any(package_root.is_relative_to(root) for root in served):
        return True
    try:
        if package_root == Path.cwd().resolve():
            return True
    except OSError:
        pass
    return _hermes_dist_installed_in(served)
