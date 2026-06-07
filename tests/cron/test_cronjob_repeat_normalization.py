"""Tests for _normalize_job_record repeat field handling."""

from __future__ import annotations

import pytest


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


class TestNormalizeJobRecordRepeatField:
    def test_repeat_string_forever_normalized_to_dict(self, tmp_cron_dir):
        """Legacy 'forever' repeat value should be normalized to dict."""
        from cron.jobs import _normalize_job_record

        job = {"id": "test-job", "repeat": "forever"}
        normalized = _normalize_job_record(job)

        assert isinstance(normalized["repeat"], dict)
        assert normalized["repeat"].get("times") is None
        assert normalized["repeat"].get("completed") == 0

    def test_repeat_string_infinite_normalized_to_dict(self, tmp_cron_dir):
        """Legacy 'infinite' repeat value should be normalized to dict."""
        from cron.jobs import _normalize_job_record

        job = {"id": "test-job", "repeat": "infinite"}
        normalized = _normalize_job_record(job)

        assert isinstance(normalized["repeat"], dict)
        assert normalized["repeat"].get("times") is None
        assert normalized["repeat"].get("completed") == 0

    def test_repeat_dict_unchanged(self, tmp_cron_dir):
        """Valid dict repeat should remain unchanged."""
        from cron.jobs import _normalize_job_record

        job = {"id": "test-job", "repeat": {"times": 10, "completed": 3}}
        normalized = _normalize_job_record(job)

        assert normalized["repeat"] == {"times": 10, "completed": 3}

    def test_repeat_none_normalized_to_empty_dict(self, tmp_cron_dir):
        """None repeat should be normalized to empty dict."""
        from cron.jobs import _normalize_job_record

        job = {"id": "test-job", "repeat": None}
        normalized = _normalize_job_record(job)

        assert normalized["repeat"] == {}

    def test_repeat_missing_normalized_to_empty_dict(self, tmp_cron_dir):
        """Missing repeat should default to empty dict."""
        from cron.jobs import _normalize_job_record

        job = {"id": "test-job"}
        normalized = _normalize_job_record(job)

        assert normalized["repeat"] == {}

    def test_repeat_empty_dict_unchanged(self, tmp_cron_dir):
        """Empty dict repeat should remain unchanged."""
        from cron.jobs import _normalize_job_record

        job = {"id": "test-job", "repeat": {}}
        normalized = _normalize_job_record(job)

        assert normalized["repeat"] == {}

    def test_list_jobs_with_string_repeat_does_not_crash(self, tmp_cron_dir):
        """list_jobs should not crash when job has string repeat."""
        from cron.jobs import create_job, list_jobs, update_job

        job = create_job(prompt="test", schedule="every 1h")
        update_job(job["id"], {"repeat": "forever"})

        jobs = list_jobs(include_disabled=True)
        assert len(jobs) == 1
        assert isinstance(jobs[0]["repeat"], dict)

    def test_hermes_cli_cron_list_with_string_repeat(self, tmp_cron_dir, capsys):
        """hermes_cli.cron.cron_list should handle string repeat gracefully."""
        from cron.jobs import create_job, update_job
        from hermes_cli.cron import cron_list

        job = create_job(prompt="test", schedule="every 1h", name="Test Job")
        update_job(job["id"], {"repeat": "forever"})

        cron_list(show_all=True)

        out = capsys.readouterr().out
        assert "Test Job" in out
        assert "∞" in out
        assert "AttributeError" not in out