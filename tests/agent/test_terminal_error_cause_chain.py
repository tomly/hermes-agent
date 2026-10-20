"""The turn terminal paths must keep the typed API error reachable from the agent.

Contract (agent/turn_recovery.py): ``max_retries_exhausted_result`` and
``nonretryable_client_error_result`` stamp ``agent._terminal_api_error`` with the original
exception. The cron caller re-raises only ``result["error"]``'s text — a bare
``RuntimeError`` — and chains that stamp as ``__cause__``, so the scheduler's
unreachable-model classifier (``cron.unreachable_retry.is_model_unreachable_failure``) can
walk the cause chain to the typed SDK exception. Before the stamp, a gateway blip that the
OpenAI SDK reports as ``APIConnectionError("Connection error.")`` reached the scheduler as
``RuntimeError("Connection error.")`` with no chain, classified as a permanent failure, and
the daily cron report was lost for the day (2026-10-20 gap).

The loop modules are imported (and patched via ``patch.object``) at collection time: their
first import does bootstrap file I/O that the suite's home I/O guard forbids mid-test.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import openai
import pytest

from agent import conversation_loop, turn_recovery_autorecover
from agent.error_classifier import classify_api_error
from agent.turn_api_error import settle_unrecovered_error


class _Agent:
    log_prefix = ""
    verbose = False
    verbose_logging = False
    provider = "openai"
    model = "gpt-5"
    base_url = "https://api.openai.com/v1"
    _current_streamed_assistant_text = ""
    _fallback_index = 0

    def _summarize_api_error(self, error):
        return str(error)

    def _has_pending_fallback(self):
        return False

    def _try_activate_fallback(self, **_kw):
        return False

    def _try_recover_primary_transport(self, *_a, **_kw):
        return False

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class _Http(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


@pytest.mark.parametrize("status", [None, 401])  # None -> max-retries path; 401 -> nonretryable path
def test_terminal_paths_keep_the_typed_api_error_on_the_agent(status):
    error = (
        openai.APIConnectionError(request=None) if status is None
        else _Http(401, "HTTP 401: invalid api key"))
    classified = classify_api_error(error, provider="openai", model="gpt-5")
    agent = _Agent()
    retry = SimpleNamespace(copilot_stale_cred_retry_attempted=False, primary_recovery_attempted=True)
    with patch.object(conversation_loop, "_is_copilot_provider", lambda a: False), \
            patch.object(turn_recovery_autorecover, "auto_recover_after_exhaustion",
                         lambda *a, **k: None):
        verdict = settle_unrecovered_error(
            agent, api_error=error, classified=classified, _retry=retry, status_code=status,
            error_msg=str(error).lower(), is_context_length_error=False,
            is_rate_limited=False, _is_zai_coding_overload=False, _provider="openai",
            _base=agent.base_url, _model="gpt-5", messages=[], api_messages=[], api_kwargs=None,
            active_system_prompt="", conversation_history=None, approx_tokens=10,
            retry_count=3, max_retries=3, compression_attempts=0, api_call_count=1,
        )
    assert verdict.action == "return" and verdict.result["failed"] is True
    assert getattr(agent, "_terminal_api_error", None) is error, (
        "terminal path dropped the typed exception; the cron raise cannot chain it")
