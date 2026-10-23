"""Bug 1468 regression: the cron run's task-level wall-clock cap.

A cron run wedged inside a single tool/provider call never looks idle — the wait paths
heartbeat activity while they block (``sequential tool running (Ns)``, ``waiting for
provider response``) — so the inactivity watchdog alone cannot end it (Bug 1468: the
ibkr-paper post-market run hung 2h13m on a dead broker until an external restart marked
it failed(interrupted)). ``cron.max_run_seconds`` (else the ledger's derived stale bound)
is the last-resort breaker that interrupts the run and fails it with the limit named.
"""

import threading
import time

import pytest


class TestRuntimeLimitLoopEdge:
    """The watchdog loop's wall-clock edge: fire even while the idle clock stays at zero."""

    def test_fires_when_wall_deadline_passes_while_idle_stays_zero(self):
        """Locks the Bug 1468 behavior: heartbeats kept ``seconds_since_activity`` at ~30s
        forever, so a limit on idle time alone never fired; the wall deadline must."""
        from cron.scheduler_liveness import _inactivity_watchdog_loop

        stop = threading.Event()
        verdict: dict = {}
        try:
            fired = _inactivity_watchdog_loop(
                get_idle_seconds=lambda: 0.0,  # the wedged path keeps the agent "active"
                limit_s=600.0,  # inactivity never trips
                poll_s=0.02,
                stop=stop,
                future_done=lambda: False,
                wall_limit_s=0.1,
                verdict=verdict,
            )
        finally:
            stop.set()
        assert fired is True
        assert verdict["reason"] == "runtime_limit"

    def test_does_not_fire_before_the_wall_deadline(self):
        """A healthy run inside its budget is untouched: the loop only leaves via stop."""
        from cron.scheduler_liveness import _inactivity_watchdog_loop

        stop = threading.Event()
        settler = threading.Timer(0.3, stop.set)
        settler.start()
        try:
            fired = _inactivity_watchdog_loop(
                get_idle_seconds=lambda: 0.0,
                limit_s=600.0,
                poll_s=0.02,
                stop=stop,
                future_done=lambda: False,
                wall_limit_s=30.0,
            )
        finally:
            settler.cancel()
            stop.set()
        assert fired is False


class TestRuntimeLimitResolution:
    """``_cron_wall_clock_seconds``: explicit config wins, else the ledger's derived bound."""

    def test_explicit_config_wins(self, monkeypatch):
        from cron import scheduler

        monkeypatch.setattr(scheduler, "load_config", lambda: {"cron": {"max_run_seconds": 1234}})
        assert scheduler._cron_wall_clock_seconds() == 1234.0

    def test_zero_disables(self, monkeypatch):
        from cron import scheduler

        monkeypatch.setattr(scheduler, "load_config", lambda: {"cron": {"max_run_seconds": 0}})
        assert scheduler._cron_wall_clock_seconds() is None

    def test_derives_from_the_stale_claim_bound_when_unset(self, monkeypatch):
        from cron import executions, scheduler

        monkeypatch.setattr(scheduler, "load_config", dict)
        monkeypatch.setattr(executions, "_live_owner_stale_after_seconds", lambda: 7200.0)
        assert scheduler._cron_wall_clock_seconds() == 7200.0
        # HERMES_CRON_TIMEOUT=0 opts out of every bound: no derived cap either.
        monkeypatch.setattr(executions, "_live_owner_stale_after_seconds", lambda: None)
        assert scheduler._cron_wall_clock_seconds() is None


def test_runtime_limit_interrupts_a_never_idle_run(monkeypatch):
    """The Bug 1468 integration lock: a run whose agent NEVER goes idle (the wedged tool
    call keeps stamping activity) is still hard-interrupted at the wall-clock cap and fails
    with a TimeoutError naming the limit. Before the cap existed this run was uninterruptible:
    the idle clock never crossed 600s and nothing else interrupted it in-process."""
    from cron import scheduler

    started, finish = threading.Event(), threading.Event()
    interrupts: list[str] = []

    class AlwaysBusyAgent:
        """Stuck in ``execute_code`` against a dead broker; heartbeats never stop."""

        def run_conversation(self, *args, **kwargs):
            started.set()
            finish.wait(60)
            return {"final_response": "finished"}

        def get_activity_summary(self):
            return {
                "seconds_since_activity": 5.0,  # heartbeat cadence, never near 600s
                "last_activity_desc": "sequential tool running (5s): execute_code",
                "api_call_count": 1,
                "max_iterations": 0,
                "current_tool": "execute_code",
            }

        def interrupt(self, message, **kwargs):
            interrupts.append(message)
            finish.set()

    monkeypatch.setattr(scheduler, "_cron_inactivity_seconds", lambda: 600.0)
    monkeypatch.setattr(scheduler, "_cron_wall_clock_seconds", lambda: 0.5)
    try:
        with pytest.raises(TimeoutError, match="wall-clock runtime limit"):
            scheduler._run_agent_with_watchdog(
                AlwaysBusyAgent(), "probe", {"schedule": {"kind": "every"}},
                "probe-job", "probe", "probe", None)
    finally:
        finish.set()
    assert interrupts, "the wall-clock cap must hard-interrupt the wedged run"
