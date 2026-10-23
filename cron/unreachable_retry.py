"""Automatic bounded re-runs for scheduled fires that never reached the model.

Inspired by Claude Cowork (desktop changelog v1.46388.1, 2026-09-04): "automatic re-runs
(after 5, 15, and 30 minutes) for a scheduled task that could not reach the model at all,
for example right after the computer wakes behind a VPN."

The unreachable class is deliberately narrow: the run must have FAILED with a transient
network / DNS error (``cron.scheduler_preflight._is_transient_provider_resolve_error``)
AND the agent must have completed zero API calls. Nothing was executed and nothing was
spent, so re-running cannot double a side effect — unlike a generic failure retry (see
PR #16512), which has to answer for one-shot dispatch accounting and mid-run side
effects. Recurring jobs only: finite one-shots are pre-claimed by ``claim_dispatch``
(at-most-times, #38758) and must not regain a consumed dispatch here.

The interrupted class (``plan_retry(job, interrupted=True)``, Bug 1468) covers the other
half-finished shape: the run reached the model and worked, then a timeout breaker (the
inactivity watchdog or the ``cron.max_run_seconds`` wall-clock cap) or a gateway shutdown
killed it mid-flight. The occurrence is already consumed — the tick advanced
``next_run_at`` before dispatch — so without compensation a daily-report job leaves a
permanently missing day. Not spend-neutral (the model already ran), hence the separate
``cron.retry_interrupted`` gate; the shared ladder, its cap, the recurring-only scope and
the no-double-count accounting keep the exposure bounded.

Both classes share one ladder state: the cap counts compensations per failure streak,
whatever killed the run. While a retry is pending the failure notice is suppressed
(Cowork re-runs silently); a run that reaches the model — success or not — resets the
ladder. Disable with ``cron.retry_unreachable: false`` / ``cron.retry_interrupted:
false`` in config.yaml.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from cron.constants import is_recurring
from hermes_time import now as _hermes_now

logger = logging.getLogger("cron.scheduler")

# Cowork's ladder: re-run after 5, 15, then 30 minutes; then give up until the
# schedule's own next occurrence.
RETRY_DELAYS_SECONDS: tuple[int, ...] = (300, 900, 1800)

# Persisted on the job while a retry cycle is active: {"attempt": <1-based count of
# retries already scheduled>, "at": <ISO instant of the pending retry>, "expr": <the cron
# expression it was planned under>}. Cleared by any run
# that reached the model.
STATE_KEY = "unreachable_retry"


def retry_enabled(cfg: Optional[dict] = None, *, interrupted: bool = False) -> bool:
    """``cron.retry_unreachable`` / ``cron.retry_interrupted`` — default ON (the
    unreachable class is spend-neutral; the interrupted class is bounded to the ladder)."""
    key = "retry_interrupted" if interrupted else "retry_unreachable"
    if cfg is None:
        try:
            from hermes_cli.config import load_config

            cfg = load_config() or {}
        except Exception:  # config unreadable — keep the reliability default
            return True
    cron_cfg = (cfg or {}).get("cron")
    if not isinstance(cron_cfg, dict):
        return True
    return cron_cfg.get(key) is not False


def is_model_unreachable_failure(exc: BaseException, agent: Any = None) -> bool:
    """True when *exc* is a transient network/DNS failure and *agent* (may be ``None``)
    never completed a model call — the run consumed nothing and executed nothing."""
    if int(getattr(agent, "session_api_calls", 0) or 0) > 0:
        return False
    from cron.scheduler_preflight import _is_transient_provider_resolve_error

    return _is_transient_provider_resolve_error(exc)


def _ladder_applies(job: dict[str, Any], *, interrupted: bool = False) -> bool:
    """The ladder's applicability gate: recurring, not paused, and enabled in config."""
    return is_recurring(job) and job.get("state") != "paused" and retry_enabled(interrupted=interrupted)


def _ladder_instant(job: dict[str, Any], natural_next: Optional[datetime],
                    now: datetime, *, interrupted: bool = False) -> Optional[datetime]:
    """The ladder instant ``plan_retry`` parks for this flagged failure, or None when it
    parks nothing: not recurring, paused, disabled, ladder exhausted, or the schedule's
    own *natural_next* fires at or before the rung. The single source of that decision
    for ``plan_retry`` and ``will_retry``."""
    attempt = int((job.get(STATE_KEY) or {}).get("attempt") or 0)
    if attempt >= len(RETRY_DELAYS_SECONDS) or not _ladder_applies(job, interrupted=interrupted):
        return None
    # late: jobs imports this module's helpers
    from cron.jobs import _instant_at_or_before, _seconds_after

    retry_dt = _seconds_after(now, RETRY_DELAYS_SECONDS[attempt])
    if natural_next is not None and _instant_at_or_before(natural_next, retry_dt):
        return None  # the natural occurrence IS the retry
    return retry_dt


def will_retry(job: dict[str, Any], *, interrupted: bool = False) -> bool:
    """True iff ``plan_retry`` will park a re-run for this flagged failure, so the scheduler
    may hold the interim notice. Called before ``mark_job_run``."""
    repeat = job.get("repeat") or {}
    times = repeat.get("times")
    counted = 0 if is_retry_run(job) else 1  # a ladder re-run does not count (``ladder_rung``)
    if times is not None and times > 0 and int(repeat.get("completed") or 0) + counted >= times:
        return False  # _advance_after_run completes the job; plan_retry never runs
    from cron.jobs import _parse_aware, compute_next_run

    now = _hermes_now()
    natural_next = _parse_aware(compute_next_run(job.get("schedule") or {}, now.isoformat()))
    if natural_next is None:
        # The natural occurrence is uncomputable (e.g. croniter missing): _advance_after_run
        # leaves the record state=error — terminal — so plan_retry never runs either.
        return False
    return _ladder_instant(job, natural_next, now, interrupted=interrupted) is not None


def clear_state(job: dict[str, Any]) -> None:
    """A run reached the model (any outcome): the ladder resets."""
    job.pop(STATE_KEY, None)


def is_retry_fire(job: dict[str, Any], next_run: str) -> bool:
    """True for the exact ladder instant parked by ``plan_retry`` (off the cron lattice).

    The expression fingerprint keeps a direct ``jobs.json`` schedule edit from inheriting the
    exception, as in ``cron.quota_hold.is_recovery_fire``.
    """
    state = job.get(STATE_KEY) or {}
    return state.get("at") == next_run and state.get("expr") == (job.get("schedule") or {}).get("expr")


def is_retry_run(job: dict[str, Any]) -> bool:
    """True when this dispatch snapshot fired the ladder instant parked by ``plan_retry``: a
    re-run of an occurrence that already counted toward ``repeat``, so it must not count again.
    Manual runs carry no ``_scheduled_instant`` and always count."""
    from cron.occurrences import scheduled_instant

    instant = job.get("_scheduled_instant")
    return instant is not None and scheduled_instant((job.get(STATE_KEY) or {}).get("at")) == instant


def plan_retry(job: dict[str, Any], *, interrupted: bool = False) -> bool:
    """Called under the jobs lock AFTER ``_advance_after_run`` computed the schedule's
    natural ``next_run_at`` for a failed, flagged run. Pulls ``next_run_at`` earlier to
    the ladder instant when that is sooner than the natural occurrence; exhausted or
    inapplicable cycles clear state and leave the schedule untouched. Returns True when
    a retry was scheduled.

    ``interrupted=True``: the run was killed mid-flight by a timeout breaker or a gateway
    shutdown (Bug 1468) instead of never reaching the model — same ladder, separate
    config gate, different copy."""
    from cron.jobs import _parse_aware  # late: jobs imports this module's helpers

    retry_dt = _ladder_instant(job, _parse_aware(job.get("next_run_at")), _hermes_now(),
                               interrupted=interrupted)
    attempt = int((job.get(STATE_KEY) or {}).get("attempt") or 0)
    if retry_dt is None:
        if attempt >= len(RETRY_DELAYS_SECONDS) and _ladder_applies(job, interrupted=interrupted):
            # Ladder exhausted: fall back to the natural schedule and reset so the NEXT
            # occurrence gets a fresh ladder if the condition is still there.
            logger.warning(
                "Job '%s': %s after %d automatic re-runs — waiting for the "
                "scheduled occurrence at %s",
                job.get("name", job.get("id", "?")),
                "runs keep getting killed mid-flight" if interrupted
                else "model unreachable", attempt, job.get("next_run_at"))
        clear_state(job)
        return False
    delay = RETRY_DELAYS_SECONDS[attempt]
    retry_at = retry_dt.isoformat()
    job[STATE_KEY] = {"attempt": attempt + 1, "at": retry_at, "expr": job["schedule"].get("expr")}
    job["next_run_at"] = retry_at
    if job.get("state") != "paused":
        job["state"] = "scheduled"
    logger.info(
        "Job '%s': %s — automatic re-run %d/%d in %ds (at %s)",
        job.get("name", job.get("id", "?")),
        "run killed mid-flight by a timeout breaker or gateway shutdown" if interrupted
        else "model unreachable with zero API calls",
        attempt + 1, len(RETRY_DELAYS_SECONDS), delay, retry_at)
    return True
