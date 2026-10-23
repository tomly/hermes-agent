"""Bug 1468 regression: compensation re-runs for occurrences killed mid-flight.

Contract (cron/unreachable_retry.py, ``plan_retry(job, interrupted=True)``): a recurring
job whose run reaches the model and is then killed — by the inactivity/runtime-limit
watchdog or by ``mark_running_jobs_interrupted`` at gateway shutdown — gets its already-
consumed occurrence re-run on the same bounded 5/15/30-minute ladder, instead of leaving
the schedule advanced past a day whose output was never produced (Bug 1468: the 2h13m
hang was recorded failed(interrupted), the ET 2026-10-21 daily report simply never
existed). Re-runs never double-count ``repeat.completed``, one-shots are excluded, the
interrupted class has its own ``cron.retry_interrupted`` gate, and a run that completes
normally resets the ladder.
"""

from datetime import datetime, timedelta, UTC
from threading import Event

import pytest

import cron.scheduler as sched
from cron import unreachable_retry as ur
from cron.jobs import (
    claim_job_for_fire,
    create_job,
    get_due_jobs,
    get_job,
    mark_job_run,
)
from cron.scheduler_bookkeeping import finish_completed_run


@pytest.fixture
def tmp_cron_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture(autouse=True)
def _reset_scheduler_state():
    """The in-flight registries are module-level globals shared across the process."""
    sched._running_job_ids.clear()
    sched._running_fire_owners.clear()
    sched._interrupted_job_ids.clear()
    yield
    sched._running_job_ids.clear()
    sched._running_fire_owners.clear()
    sched._interrupted_job_ids.clear()


_KILL_ERROR = (
    "TimeoutError: Cron job 'ibkr-paper post-market' exceeded its 7200s wall-clock "
    "runtime limit (cron.max_run_seconds) after 8132s — hard-interrupted")


def test_killed_run_parks_the_compensation_ladder_then_falls_back(tmp_cron_home):
    """Locks the Bug 1468 defect: a mid-flight kill used to leave the occurrence consumed
    (next_run_at already advanced a full period, no retry state) — the missing daily
    report could never heal itself. Now the kill parks the 5/15/30-minute ladder, the
    ladder gives up after its last rung, and a run that completes normally resets it."""
    job_id = create_job("nightly report", "every 24h")["id"]
    now = datetime.now(UTC)
    for i, delay in enumerate(ur.RETRY_DELAYS_SECONDS):
        assert mark_job_run(job_id, False, _KILL_ERROR, interrupted=True)
        j = get_job(job_id)
        nxt = datetime.fromisoformat(j["next_run_at"])
        assert timedelta(0) < nxt - now <= timedelta(seconds=delay + 120), (
            f"attempt {i}: expected compensation ~{delay}s out, got {nxt - now}")
        assert j[ur.STATE_KEY]["attempt"] == i + 1

    # Ladder exhausted: the next kill keeps the natural schedule instead of re-parking.
    assert mark_job_run(job_id, False, _KILL_ERROR, interrupted=True)
    j = get_job(job_id)
    assert j.get(ur.STATE_KEY) is None
    assert datetime.fromisoformat(j["next_run_at"]) - now > timedelta(hours=1)

    # A completed (non-killed) run resets a pending ladder: the compensation either
    # healed the job or the failure is intrinsic — either way the streak is over.
    assert mark_job_run(job_id, False, _KILL_ERROR, interrupted=True)
    assert get_job(job_id)[ur.STATE_KEY]["attempt"] == 1
    assert mark_job_run(job_id, True, None)
    assert get_job(job_id).get(ur.STATE_KEY) is None


def test_watchdog_kill_flags_the_run_and_the_tail_compensates(tmp_cron_home, monkeypatch):
    """The integration lock: a run hard-interrupted at the wall-clock cap must reach the
    bookkeeping tail flagged, and the tail must park the compensation ladder in the same
    fenced store write. Before the fix the flag did not exist: the TimeoutError was
    recorded as an ordinary failure and next_run_at stayed a full day out."""
    started, finish = Event(), Event()
    job = create_job("post-market report", "every 24h")

    class AlwaysBusyAgent:
        """Heartbeating the whole time — the wedged-broker shape the idle clock cannot see."""

        def run_conversation(self, *args, **kwargs):
            started.set()
            finish.wait(60)
            return {"final_response": "finished"}

        def get_activity_summary(self):
            return {
                "seconds_since_activity": 5.0,
                "last_activity_desc": "sequential tool running (5s): execute_code",
                "api_call_count": 1,
                "max_iterations": 0,
                "current_tool": "execute_code",
            }

        def interrupt(self, message, **kwargs):
            finish.set()

    monkeypatch.setattr(sched, "_cron_inactivity_seconds", lambda: 600.0)
    monkeypatch.setattr(sched, "_cron_wall_clock_seconds", lambda: 0.5)
    monkeypatch.setattr(sched, "finish_execution", lambda *_a, **_kw: None)
    try:
        with pytest.raises(TimeoutError, match="wall-clock runtime limit"):
            sched._run_agent_with_watchdog(
                AlwaysBusyAgent(), "probe", job, job["id"], "probe", "probe", None)
    finally:
        finish.set()
    assert job["_interrupted_retry"] is True, (
        "the watchdog kill must flag the run for compensation")

    assert finish_completed_run(
        sched._RunDelivery(job=job, success=False, error=_KILL_ERROR), None, "exec-kill")
    j = get_job(job["id"])
    now = datetime.now(UTC)
    nxt = datetime.fromisoformat(j["next_run_at"])
    assert timedelta(0) < nxt - now <= timedelta(seconds=ur.RETRY_DELAYS_SECONDS[0] + 120), (
        "the killed occurrence must be re-run in minutes, not wait a full period")
    assert j[ur.STATE_KEY]["attempt"] == 1


def test_gateway_shutdown_kill_compensates(tmp_cron_home):
    """The original incident's kill path: the gateway restart marked the in-flight run
    interrupted. That shutdown mark must plan the compensation re-run, not just record
    the failure — before the fix the mark left next_run_at on the natural schedule."""
    job = create_job("nightly report", "every 24h")
    job_id = job["id"]
    claimed = claim_job_for_fire(job_id, force=True, return_job=True)
    owner = claimed["fire_claim"]["by"]
    profile_home = sched._get_hermes_home().resolve()
    sched._running_job_ids.add(sched._inflight_key(job_id))
    sched._running_fire_owners[sched._inflight_key(job_id)] = {
        object(): (owner, profile_home)}

    assert sched.mark_running_jobs_interrupted("gateway restart (hermes update)") == [job_id]

    j = get_job(job_id)
    now = datetime.now(UTC)
    nxt = datetime.fromisoformat(j["next_run_at"])
    assert timedelta(0) < nxt - now <= timedelta(seconds=ur.RETRY_DELAYS_SECONDS[0] + 120), (
        "a shutdown-killed occurrence must be compensated, not consumed")
    assert j[ur.STATE_KEY]["attempt"] == 1
    assert j["last_status"] != "ok", "an interrupted run is never ok"


def test_compensation_reruns_do_not_spend_extra_repeat_budget(tmp_cron_home, monkeypatch):
    """幂等防重: a compensation re-run repeats an occurrence that already counted toward
    ``repeat``, so the whole ladder must leave ``repeat.completed`` untouched — otherwise
    a kill storm retires a finite job without ever producing its output."""
    clock = [datetime.now(UTC)]
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: clock[0])
    monkeypatch.setattr(ur, "_hermes_now", lambda: clock[0])
    monkeypatch.setattr(sched, "finish_execution", lambda *_a, **_kw: None)
    runs = []
    monkeypatch.setattr(sched, "run_one_job", lambda job, **_kw: runs.append(job) or True)
    monkeypatch.setattr(sched, "mark_execution_running", lambda *_a, **_kw: {})
    monkeypatch.setattr(sched, "_deliver_crash_failure", lambda *_a, **_kw: (None, "suppressed"))
    job_id = create_job("digest", "every 24h", repeat=2)["id"]

    for rung in range(1 + len(ur.RETRY_DELAYS_SECONDS)):  # the occurrence, then every rung
        execution_id = f"exec-{rung}"
        clock[0] = datetime.fromisoformat(get_job(job_id)["next_run_at"]) + timedelta(seconds=1)
        due = next(d for d in get_due_jobs() if d["id"] == job_id)
        assert sched._process_due_job(dict(due, execution_id=execution_id), None, None, False)
        run = runs[-1]
        run["_interrupted_retry"] = True  # the watchdog's stamp on the dispatch snapshot
        assert finish_completed_run(
            sched._RunDelivery(job=run, success=False, error=_KILL_ERROR),
            run["fire_claim"]["by"], execution_id)
        assert get_job(job_id)["repeat"]["completed"] == 1, (
            f"rung {rung}: only the occurrence itself counts")

    j = get_job(job_id)
    assert j["state"] == "scheduled" and j.get(ur.STATE_KEY) is None
    assert datetime.fromisoformat(j["next_run_at"]) - clock[0] > timedelta(hours=23)


def test_retry_interrupted_gate_and_one_shot_exclusion(tmp_cron_home, monkeypatch):
    """``cron.retry_interrupted: false`` disables the compensation (its own gate: unlike
    the unreachable class this re-run is not spend-neutral), and one-shot jobs are never
    resurrected — their dispatch accounting is at-most-times."""
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"cron": {"retry_interrupted": False}})
    job_id = create_job("nightly report", "every 24h")["id"]
    now = datetime.now(UTC)
    assert mark_job_run(job_id, False, _KILL_ERROR, interrupted=True)
    j = get_job(job_id)
    assert j.get(ur.STATE_KEY) is None, "gate off: no compensation state"
    assert datetime.fromisoformat(j["next_run_at"]) - now > timedelta(hours=23)
    assert ur.will_retry(j, interrupted=True) is False

    once = create_job(
        "one shot", (datetime.now(UTC) + timedelta(minutes=1)).isoformat())
    assert mark_job_run(once["id"], False, _KILL_ERROR, interrupted=True)
    remaining = get_job(once["id"])
    assert remaining is None or remaining.get(ur.STATE_KEY) is None


def test_interrupted_gate_is_independent_of_retry_unreachable(tmp_cron_home, monkeypatch):
    """The two classes have separate switches: turning the spend-neutral unreachable
    ladder off must not silently disable compensation for killed runs."""
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"cron": {"retry_unreachable": False}})
    job_id = create_job("nightly report", "every 24h")["id"]
    now = datetime.now(UTC)
    assert mark_job_run(job_id, False, _KILL_ERROR, interrupted=True)
    j = get_job(job_id)
    nxt = datetime.fromisoformat(j["next_run_at"])
    assert timedelta(0) < nxt - now <= timedelta(seconds=ur.RETRY_DELAYS_SECONDS[0] + 120)
    assert j[ur.STATE_KEY]["attempt"] == 1
    assert ur.will_retry(j, interrupted=True) is True, (
        "a pending compensation holds the failure notice")
