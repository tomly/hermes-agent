"""Regression test for Bug 1345: unresolved ${VAR} placeholders in MCP headers.

When a profile's .env is updated to add a new secret (e.g. MCP_POSTIZ_API_KEY)
and the corresponding MCP server config references it as
``Authorization: Bearer ${MCP_POSTIZ_API_KEY}``, the interpolation must resolve
the placeholder. If the env var is missing from both the profile scope
(multiplex) and os.environ (single-profile), the placeholder is left as a
literal string and sent to the remote server, which rejects it with 401.

This test locks in the warning that fires when an unresolved ${VAR} placeholder
survives interpolation — the diagnostic that would have surfaced this bug early.
"""

import logging

import pytest

from tools import mcp_tool_config as _mcp_config
from tools.mcp_tool_config import _warn_unresolved_env_refs


@pytest.fixture(autouse=True)
def _reset_dedupe():
    _mcp_config._unresolved_warned.clear()
    yield
    _mcp_config._unresolved_warned.clear()


# ---------------------------------------------------------------------------
# Unit tests for _warn_unresolved_env_refs
# ---------------------------------------------------------------------------

def test_fully_resolved_config_no_warnings(caplog):
    """When all ${VAR} refs are resolved (no ${...} left), no warning fires."""
    config = {
        "url": "https://example.com/mcp",
        "headers": {"Authorization": "Bearer actual-secret-123"},
        "enabled": True,
    }
    with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
        flagged = _warn_unresolved_env_refs("srv", config)
    assert flagged == []
    assert not any("unresolved" in r.getMessage().lower() for r in caplog.records)


def test_unresolved_placeholder_in_header_flagged(caplog):
    """Bug 1345 scenario: ${MCP_POSTIZ_API_KEY} left as-is in Authorization header."""
    config = {
        "url": "http://localhost:3000/api/mcp",
        "headers": {"Authorization": "Bearer ${MCP_POSTIZ_API_KEY}"},
        "enabled": True,
    }
    with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
        flagged = _warn_unresolved_env_refs("postiz", config)
    assert flagged == ["headers.Authorization"]
    messages = [r.getMessage() for r in caplog.records]
    assert any("postiz" in m and "headers.Authorization" in m for m in messages)
    # The placeholder itself should NOT be logged (secret-redaction hygiene).
    assert not any("MCP_POSTIZ_API_KEY" in m for m in messages)


def test_unresolved_placeholder_in_url_flagged(caplog):
    """An unresolved ${VAR} in the URL is also flagged."""
    config = {
        "url": "https://${MCP_HOST}/api/mcp",
        "enabled": True,
    }
    with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
        flagged = _warn_unresolved_env_refs("srv", config)
    assert flagged == ["url"]


def test_unresolved_placeholder_in_env_dict_flagged():
    """An unresolved ${VAR} in a stdio env value is flagged."""
    config = {
        "command": "npx",
        "env": {"API_TOKEN": "${MISSING_TOKEN}"},
    }
    flagged = _warn_unresolved_env_refs("srv", config)
    assert flagged == ["env.API_TOKEN"]


def test_multiple_unresolved_refs_all_flagged(caplog):
    """Multiple unresolved ${VAR} refs produce separate flagged paths."""
    config = {
        "url": "https://${MCP_HOST}/api",
        "headers": {
            "Authorization": "Bearer ${MCP_TOKEN}",
            "X-Api-Key": "${MCP_KEY}",
        },
    }
    with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
        flagged = _warn_unresolved_env_refs("srv", config)
    assert set(flagged) == {"url", "headers.Authorization", "headers.X-Api-Key"}


def test_non_string_values_ignored():
    """Non-string values (int, bool, None) are skipped."""
    config = {"timeout": 30, "enabled": True, "retries": None}
    flagged = _warn_unresolved_env_refs("srv", config)
    assert flagged == []


def test_warning_deduped_per_process(caplog):
    """The warning is emitted only once per (server, key_path) pair."""
    config = {"url": "https://${MISSING}/api"}
    with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
        first = _warn_unresolved_env_refs("srv", config)
        second = _warn_unresolved_env_refs("srv", config)
    # Both calls still report the flagged path.
    assert first == second == ["url"]
    # But only one warning record is emitted.
    warn_records = [r for r in caplog.records if "unresolved" in r.getMessage().lower()]
    assert len(warn_records) == 1


def test_warning_not_deduped_across_servers(caplog):
    """Different servers with the same unresolved ref each get a warning."""
    config = {"url": "https://${MISSING}/api"}
    with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
        _warn_unresolved_env_refs("srv_a", config)
        _warn_unresolved_env_refs("srv_b", config)
    warn_records = [r for r in caplog.records if "unresolved" in r.getMessage().lower()]
    assert len(warn_records) == 2


# ---------------------------------------------------------------------------
# Integration test through _load_mcp_config
# ---------------------------------------------------------------------------

def test_load_mcp_config_warns_on_unresolved_placeholder(tmp_path, monkeypatch, caplog):
    """E2E: _load_mcp_config emits the unresolved-placeholder warning when a
    referenced env var is absent from both the profile .env and os.environ.

    This is the exact scenario of Bug 1345: Postiz MCP tools returned 401
    because ${MCP_POSTIZ_API_KEY} was sent as a literal string.
    """
    from unittest.mock import patch as mock_patch

    home = tmp_path / "profile"
    home.mkdir()

    # .env does NOT contain MCP_POSTIZ_API_KEY (simulates pre-fix state)
    (home / ".env").write_text("OTHER_KEY=other-value\n", encoding="utf-8")

    servers = {
        "postiz": {
            "url": "http://localhost:3000/api/mcp",
            "headers": {"Authorization": "Bearer ${MCP_POSTIZ_API_KEY}"},
            "enabled": True,
        },
    }

    # Ensure the var is not in os.environ
    monkeypatch.delenv("MCP_POSTIZ_API_KEY", raising=False)

    with mock_patch("hermes_cli.config.load_config",
                    return_value={"mcp_servers": servers}), \
         mock_patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
        result = _mcp_config._load_mcp_config()

    assert "postiz" in result
    # The placeholder is left as-is (unresolved) — that is the bug behavior.
    # The fix adds a warning to make it visible.
    auth = result["postiz"]["headers"]["Authorization"]
    # The placeholder is NOT interpolated because the env var is missing
    assert "${MCP_POSTIZ_API_KEY}" in auth

    # The warning should have been emitted
    messages = [r.getMessage() for r in caplog.records]
    assert any("unresolved" in m.lower() and "headers.Authorization" in m for m in messages)


def test_load_mcp_config_no_warning_when_resolved(tmp_path, monkeypatch, caplog):
    """E2E: When the env var IS available, no unresolved-placeholder warning fires."""
    from unittest.mock import patch as mock_patch

    servers = {
        "postiz": {
            "url": "http://localhost:3000/api/mcp",
            "headers": {"Authorization": "Bearer ${MCP_POSTIZ_API_KEY}"},
            "enabled": True,
        },
    }

    # The env var IS in os.environ
    monkeypatch.setenv("MCP_POSTIZ_API_KEY", "real-api-key-12345")

    with mock_patch("hermes_cli.config.load_config",
                    return_value={"mcp_servers": servers}), \
         mock_patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
        result = _mcp_config._load_mcp_config()

    assert "postiz" in result
    auth = result["postiz"]["headers"]["Authorization"]
    # Placeholder is resolved correctly
    assert auth == "Bearer real-api-key-12345"
    assert "${MCP_POSTIZ_API_KEY}" not in auth

    # No unresolved warning
    messages = [r.getMessage() for r in caplog.records]
    assert not any("unresolved" in m.lower() for m in messages)
