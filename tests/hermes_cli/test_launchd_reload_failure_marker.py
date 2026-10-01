"""launchd plist reload must survive an unknown gateway pid and leave a durable failure marker.

Regression for the 2026-09-30 default-gateway outage: a service reinstall rewrote
``ai.hermes.gateway.plist`` but never completed the bootstrap, leaving the launchd job
unregistered. When the gateway then exited ("exit 1 so the service manager can revive the
gateway"), KeepAlive had nothing to revive — the default gateway stayed down ~23h until a manual
``launchctl bootstrap``. Two defects locked here:

1. ``refresh_launchd_plist_if_needed`` silently degraded to the in-process bootout/bootstrap
   fallback whenever ``get_running_pid()`` returned None — but an unknown pid does NOT mean this
   process is outside the gateway's launchd coalition, and the in-process bootout kills the
   retrying process itself (coalition membership survives reparenting), so bootstrap never ran
   and the freshly written plist stayed unregistered.
2. Neither reload path left any durable failure signal beyond logs/launchd-reload.log, so
   ``hermes gateway status`` and daily inspection had nothing to alert on.
"""
import json
import shlex
from types import SimpleNamespace

import pytest

import hermes_cli.gateway as gateway_cli


@pytest.fixture
def stale_plist(tmp_path, monkeypatch):
    """A stale installed plist with every process side effect recorded or neutralized."""
    plist_path = tmp_path / "ai.hermes.gateway.plist"
    plist_path.write_text("<old/>", encoding="utf-8")
    monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)
    monkeypatch.setattr(gateway_cli, "launchd_plist_is_current", lambda: False)
    monkeypatch.setattr(gateway_cli, "generate_launchd_plist", lambda: "<new/>")
    monkeypatch.setattr(gateway_cli, "_refuse_temp_home_service_write", lambda *a: False)
    monkeypatch.setattr(gateway_cli, "_prepare_service_launcher", lambda: None)
    monkeypatch.setattr(gateway_cli, "get_launchd_label", lambda: "ai.hermes.gateway")
    monkeypatch.setattr(gateway_cli, "_launchd_domain", lambda: "gui/501")
    monkeypatch.setattr(gateway_cli, "_append_launchd_reload_log", lambda msg: None)

    service_calls = []

    def fake_run(cmd, check=False, **kwargs):
        service_calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)

    def no_real_popen(cmd, **kwargs):
        raise AssertionError(
            "tests must never spawn a real launchctl helper; stub subprocess.Popen"
        )

    monkeypatch.setattr(gateway_cli.subprocess, "Popen", no_real_popen)
    return plist_path


def test_refresh_with_unknown_gateway_pid_defers_to_helper_instead_of_in_process_bootout(
    stale_plist, monkeypatch,
):
    """A failed pid lookup must still route the reload to the coalition-safe deferred helper, not
    the in-process bootout that can kill the retrying process mid-reload and strand the service
    unregistered (defect 1 — the ~23h outage's mechanism)."""
    monkeypatch.setattr("gateway.status.get_running_pid", lambda *a, **k: None)

    helpers, in_process = [], []
    monkeypatch.setattr(
        gateway_cli, "_spawn_deferred_launchd_reload", lambda **kw: helpers.append(kw) or True
    )
    monkeypatch.setattr(
        gateway_cli, "_retry_launchctl_bootstrap_until_registered",
        lambda *a, **k: in_process.append(a) or True,
    )

    assert gateway_cli.refresh_launchd_plist_if_needed() is True
    assert helpers, "unknown gateway pid must still use the deferred (launchctl submit) helper"
    assert helpers[0]["gateway_pid"] is None
    assert not in_process, "in-process bootout/bootstrap must not run when the helper spawned"


def test_in_process_reload_failure_leaves_durable_marker_and_success_clears_it(
    stale_plist, monkeypatch,
):
    """When the helper can't spawn and the in-process retry still fails to register the job, the
    reload-failed marker must be written for status/inspection — and cleared by a later reload
    that ends with a supervised PID (defect 2)."""
    monkeypatch.setattr("gateway.status.get_running_pid", lambda *a, **k: None)
    monkeypatch.setattr(gateway_cli, "_spawn_deferred_launchd_reload", lambda **kw: False)

    registered = iter([False, True])
    monkeypatch.setattr(
        gateway_cli, "_retry_launchctl_bootstrap_until_registered",
        lambda *a, **k: next(registered),
    )

    assert gateway_cli.refresh_launchd_plist_if_needed() is False
    marker = gateway_cli._launchd_reload_failed_marker_path()
    assert marker.exists(), "a failed reload must leave the durable failure marker"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert "supervised PID" in payload["reason"]

    assert gateway_cli.refresh_launchd_plist_if_needed() is True
    assert not marker.exists(), "a reload that ends with a supervised PID must clear the marker"


def test_deferred_helper_script_skips_pid_wait_and_marks_failure_without_a_known_gateway_pid(
    stale_plist, monkeypatch,
):
    """The helper's final verdict is a positive-PID ``launchctl list`` — success clears the failure
    marker, failure writes it — and with no known gateway pid there is no ``kill -0`` wait on a
    garbage pid."""
    monkeypatch.setattr("gateway.status.get_running_pid", lambda *a, **k: None)

    popen_calls = []
    monkeypatch.setattr(
        gateway_cli.subprocess, "Popen",
        lambda cmd, **kw: popen_calls.append(cmd) or SimpleNamespace(pid=1),
    )

    assert gateway_cli._spawn_deferred_launchd_reload(
        domain="gui/501", label="ai.hermes.gateway",
        target="gui/501/ai.hermes.gateway", plist_path=stale_plist, gateway_pid=None,
    ) is True

    script = popen_calls[0][popen_calls[0].index("--") + 3]
    assert "kill -0" not in script, "no pid wait is possible when the gateway pid is unknown"
    marker_q = shlex.quote(str(gateway_cli._launchd_reload_failed_marker_path()))
    # The verdict block is the LAST `if launchctl list` — after the bootstrap retries.
    verdict = script[script.rindex("if launchctl list"):]
    assert script.index("launchctl bootstrap") < script.rindex("if launchctl list")
    assert f"rm -f {marker_q}" in verdict, "success must clear the failure marker"
    assert "NOT registered after" in verdict
    assert f"> {marker_q}" in verdict, "failure must write the durable failure marker"
    assert script.rindex("launchctl remove") > script.rindex("fi;")


def test_launchd_status_alerts_on_reload_failed_marker(stale_plist, monkeypatch, capsys):
    """``hermes gateway status`` must surface the unregistered-service state loudly — the generic
    'not loaded' line is what let the 2026-09-30 outage sit unnoticed for ~23h."""
    monkeypatch.setattr(gateway_cli, "launchd_plist_is_current", lambda: True)
    monkeypatch.setattr("gateway.status.get_running_pid", lambda *a, **k: None)

    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["launchctl", "list"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)
    gateway_cli._write_launchd_reload_failed_marker(
        "launchd reload of gui/501/ai.hermes.gateway ended without a supervised PID"
    )

    gateway_cli.launchd_status()

    out = capsys.readouterr().out
    assert "not loaded" in out
    assert "reload failed" in out.lower()
    assert "ended without a supervised PID" in out
    assert "launchd CANNOT revive" in out

    # Service recovered since: the marker note downgrades to stale, not alarming.
    def fake_run_supervised(cmd, **kwargs):
        if cmd[:2] == ["launchctl", "list"]:
            return SimpleNamespace(
                returncode=0,
                stdout='{\n\t"PID" = 5150;\n\t"Label" = "ai.hermes.gateway";\n};',
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run_supervised)
    gateway_cli.launchd_status()

    out = capsys.readouterr().out
    assert "supervised by launchd (PID 5150)" in out
    assert "supervised again" in out
    assert "CANNOT revive" not in out
