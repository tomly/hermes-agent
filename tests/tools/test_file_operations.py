"""Tests for tools/file_operations.py — deny list, result dataclasses, helpers."""

import os
import pytest
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

from tools.file_operations import (
    _is_write_denied,
    WRITE_DENIED_PATHS,
    WRITE_DENIED_PREFIXES,
    ReadResult,
    WriteResult,
    PatchResult,
    SearchResult,
    SearchMatch,
    LintResult,
    ShellFileOperations,
    BINARY_EXTENSIONS,
    IMAGE_EXTENSIONS,
    MAX_LINE_LENGTH,
    normalize_read_pagination,
    normalize_search_pagination,
)


# =========================================================================
# Write deny list
# =========================================================================

class TestIsWriteDenied:
    def test_ssh_authorized_keys_denied(self):
        path = os.path.join(str(Path.home()), ".ssh", "authorized_keys")
        assert _is_write_denied(path) is True

    def test_ssh_id_rsa_denied(self):
        path = os.path.join(str(Path.home()), ".ssh", "id_rsa")
        assert _is_write_denied(path) is True

    def test_netrc_denied(self):
        path = os.path.join(str(Path.home()), ".netrc")
        assert _is_write_denied(path) is True

    def test_aws_prefix_denied(self):
        path = os.path.join(str(Path.home()), ".aws", "credentials")
        assert _is_write_denied(path) is True

    def test_kube_prefix_denied(self):
        path = os.path.join(str(Path.home()), ".kube", "config")
        assert _is_write_denied(path) is True

    def test_normal_file_allowed(self, tmp_path):
        path = str(tmp_path / "safe_file.txt")
        assert _is_write_denied(path) is False

    def test_project_file_allowed(self):
        assert _is_write_denied("/tmp/project/main.py") is False

    def test_tilde_expansion(self):
        assert _is_write_denied("~/.ssh/authorized_keys") is True



# =========================================================================
# Result dataclasses
# =========================================================================

class TestReadResult:
    def test_to_dict_omits_defaults(self):
        r = ReadResult()
        d = r.to_dict()
        assert "error" not in d    # None omitted
        assert "similar_files" not in d  # empty list omitted

    def test_to_dict_preserves_empty_content(self):
        """Empty file should still have content key in the dict."""
        r = ReadResult(content="", total_lines=0, file_size=0)
        d = r.to_dict()
        assert "content" in d
        assert d["content"] == ""
        assert d["total_lines"] == 0
        assert d["file_size"] == 0

    def test_to_dict_includes_values(self):
        r = ReadResult(content="hello", total_lines=10, file_size=50, truncated=True)
        d = r.to_dict()
        assert d["content"] == "hello"
        assert d["total_lines"] == 10
        assert d["truncated"] is True

    def test_binary_fields(self):
        r = ReadResult(is_binary=True, is_image=True, mime_type="image/png")
        d = r.to_dict()
        assert d["is_binary"] is True
        assert d["is_image"] is True
        assert d["mime_type"] == "image/png"


class TestWriteResult:
    def test_to_dict_omits_none(self):
        r = WriteResult(bytes_written=100)
        d = r.to_dict()
        assert d["bytes_written"] == 100
        assert "error" not in d
        assert "warning" not in d

    def test_to_dict_includes_error(self):
        r = WriteResult(error="Permission denied")
        d = r.to_dict()
        assert d["error"] == "Permission denied"


class TestPatchResult:
    def test_to_dict_success(self):
        r = PatchResult(success=True, diff="--- a\n+++ b", files_modified=["a.py"])
        d = r.to_dict()
        assert d["success"] is True
        assert d["diff"] == "--- a\n+++ b"
        assert d["files_modified"] == ["a.py"]

    def test_to_dict_error(self):
        r = PatchResult(error="File not found")
        d = r.to_dict()
        assert d["success"] is False
        assert d["error"] == "File not found"


class TestSearchResult:
    def test_to_dict_with_matches(self):
        m = SearchMatch(path="a.py", line_number=10, content="hello")
        r = SearchResult(matches=[m], total_count=1)
        d = r.to_dict()
        assert d["total_count"] == 1
        assert len(d["matches"]) == 1
        assert d["matches"][0]["path"] == "a.py"

    def test_to_dict_empty(self):
        r = SearchResult()
        d = r.to_dict()
        assert d["total_count"] == 0
        assert "matches" not in d

    def test_to_dict_files_mode(self):
        r = SearchResult(files=["a.py", "b.py"], total_count=2)
        d = r.to_dict()
        assert d["files"] == ["a.py", "b.py"]

    def test_to_dict_count_mode(self):
        r = SearchResult(counts={"a.py": 3, "b.py": 1}, total_count=4)
        d = r.to_dict()
        assert d["counts"]["a.py"] == 3

    def test_truncated_flag(self):
        r = SearchResult(total_count=100, truncated=True)
        d = r.to_dict()
        assert d["truncated"] is True


class TestLintResult:
    def test_skipped(self):
        r = LintResult(skipped=True, message="No linter for .md files")
        d = r.to_dict()
        assert d["status"] == "skipped"
        assert d["message"] == "No linter for .md files"

    def test_success(self):
        r = LintResult(success=True, output="")
        d = r.to_dict()
        assert d["status"] == "ok"

    def test_error(self):
        r = LintResult(success=False, output="SyntaxError line 5")
        d = r.to_dict()
        assert d["status"] == "error"
        assert "SyntaxError" in d["output"]


# =========================================================================
# ShellFileOperations helpers
# =========================================================================

@pytest.fixture()
def mock_env():
    """Create a mock terminal environment."""
    env = MagicMock()
    env.cwd = "/tmp/test"
    env.execute.return_value = {"output": "", "returncode": 0}
    return env


@pytest.fixture()
def file_ops(mock_env):
    return ShellFileOperations(mock_env)


def make_real_subprocess_env(cwd: str, include_stderr: bool = False) -> MagicMock:
    """Mock env whose execute() runs the command in a real subprocess.

    For tests that need the generated shell scripts to actually run
    (search fallback, atomic-write permissions) instead of being
    intercepted by a bare MagicMock.  ``include_stderr`` folds stderr
    into ``output`` for tests that surface shell error text; leave it
    off for tests that parse structured stdout (e.g. find results).
    """
    env = MagicMock()
    env.cwd = cwd

    def execute(command, **kwargs):
        completed = subprocess.run(
            command,
            shell=True,
            text=True,
            capture_output=True,
            input=kwargs.get("stdin_data"),
        )
        output = completed.stdout
        if include_stderr:
            output += completed.stderr
        return {
            "output": output,
            "returncode": completed.returncode,
        }

    env.execute = execute
    return env


def make_real_subprocess_env_replace(cwd: str) -> MagicMock:
    """Real-subprocess mock env that decodes stdout with ``errors="replace"``.

    Mirrors the real terminal backend (which decodes tool output with
    errors="replace"). Needed to exercise the truncated-mid-char sample path
    that ``head -c 1000`` produces on a multi-byte UTF-8 file — a strict
    decode would raise instead of yielding the phantom U+FFFD the real
    backend would have produced.
    """
    env = MagicMock()
    env.cwd = cwd

    def execute(command, **kwargs):
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            input=kwargs.get("stdin_data"),
        )
        output = completed.stdout.decode("utf-8", errors="replace")
        return {
            "output": output,
            "returncode": completed.returncode,
        }

    env.execute = execute
    return env


class TestShellFileOpsHelpers:
    def test_normalize_read_pagination_clamps_invalid_values(self):
        assert normalize_read_pagination(offset=0, limit=0) == (1, 1)
        assert normalize_read_pagination(offset=-10, limit=-5) == (1, 1)
        assert normalize_read_pagination(offset="bad", limit="bad") == (1, 500)
        assert normalize_read_pagination(offset=2, limit=999999) == (2, 2000)

    def test_normalize_search_pagination_clamps_invalid_values(self):
        assert normalize_search_pagination(offset=-10, limit=-5) == (0, 1)
        assert normalize_search_pagination(offset="bad", limit="bad") == (0, 50)
        assert normalize_search_pagination(offset=3, limit=0) == (3, 1)

    def test_escape_shell_arg_simple(self, file_ops):
        assert file_ops._escape_shell_arg("hello") == "'hello'"

    def test_escape_shell_arg_with_quotes(self, file_ops):
        result = file_ops._escape_shell_arg("it's")
        assert "'" in result
        # Should be safely escaped
        assert result.count("'") >= 4  # wrapping + escaping

    def test_escape_shell_arg_rewrites_forward_slash_native_paths(self, monkeypatch, file_ops):
        import tools.environments.local as local_mod

        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        assert file_ops._escape_shell_arg(
            "C:/Users/alice/notes.txt"
        ) == "'/c/Users/alice/notes.txt'"

    def test_read_file_uses_bash_safe_windows_paths(self, mock_env, monkeypatch):
        import tools.environments.local as local_mod

        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        commands = []

        def side_effect(command, **kwargs):
            commands.append(command)
            if command.startswith("wc -c"):
                return {"output": "5\n", "returncode": 0}
            if command.startswith("head -c"):
                return {"output": "hello", "returncode": 0}
            if command.startswith("sed -n"):
                return {"output": "hello\n", "returncode": 0}
            if command.startswith("wc -l"):
                return {"output": "1\n", "returncode": 0}
            return {"output": "", "returncode": 0}

        mock_env.execute.side_effect = side_effect
        ops = ShellFileOperations(mock_env)
        result = ops.read_file(r"C:\Users\alice\notes.txt")

        assert result.error is None
        assert commands[0] == "wc -c < '/c/Users/alice/notes.txt' 2>/dev/null"
        # Byte-domain binary probe runs between the size check and the
        # decoded-text sample (its mock output is empty → probe degrades to
        # None and the decoded heuristics take over).
        assert commands[1].startswith("python3 -c ")
        assert commands[2] == "head -c 1000 '/c/Users/alice/notes.txt' 2>/dev/null"
        assert commands[3] == "sed -n '1,2000p' '/c/Users/alice/notes.txt'"
        assert commands[4] == "wc -l < '/c/Users/alice/notes.txt'"

    def test_is_likely_binary_by_extension(self, file_ops):
        assert file_ops._is_likely_binary("photo.png") is True
        assert file_ops._is_likely_binary("data.db") is True
        assert file_ops._is_likely_binary("code.py") is False
        assert file_ops._is_likely_binary("readme.md") is False

    def test_is_likely_binary_by_content(self, file_ops):
        # High ratio of non-printable chars -> binary
        binary_content = "\x00\x01\x02\x03" * 250
        assert file_ops._is_likely_binary("unknown", binary_content) is True

        # Normal text -> not binary
        assert file_ops._is_likely_binary("unknown", "Hello world\nLine 2\n") is False

    def test_is_image(self, file_ops):
        assert file_ops._is_image("photo.png") is True
        assert file_ops._is_image("pic.jpg") is True
        assert file_ops._is_image("icon.ico") is True
        assert file_ops._is_image("data.pdf") is False
        assert file_ops._is_image("code.py") is False

    def test_add_line_numbers(self, file_ops):
        content = "line one\nline two\nline three"
        result = file_ops._add_line_numbers(content)
        assert "     1|line one" in result
        assert "     2|line two" in result
        assert "     3|line three" in result

    def test_add_line_numbers_with_offset(self, file_ops):
        content = "continued\nmore"
        result = file_ops._add_line_numbers(content, start_line=50)
        assert "    50|continued" in result
        assert "    51|more" in result

    def test_add_line_numbers_truncates_long_lines(self, file_ops):
        long_line = "x" * (MAX_LINE_LENGTH + 100)
        result = file_ops._add_line_numbers(long_line)
        assert "[truncated]" in result

    def test_unified_diff(self, file_ops):
        old = "line1\nline2\nline3\n"
        new = "line1\nchanged\nline3\n"
        diff = file_ops._unified_diff(old, new, "test.py")
        assert "-line2" in diff
        assert "+changed" in diff
        assert "test.py" in diff

    def test_cwd_from_env(self, mock_env):
        mock_env.cwd = "/custom/path"
        ops = ShellFileOperations(mock_env)
        assert ops.cwd == "/custom/path"

    def test_cwd_fallback_to_slash(self):
        env = MagicMock(spec=[])  # no cwd attribute
        ops = ShellFileOperations(env)
        assert ops.cwd == "/"

    def test_read_file_strips_leaked_terminal_fence_markers(self, mock_env):
        leaked = (
            "'\x07__HERMES_FENCE_a9f7b3__\x1b]0;cat "
            "'/tmp/test/a.py' 2> /dev/null\x07\n"
            "print('ok')\n"
            "__HERMES_FENCE_a9f7b3__\x07'\n"
        )

        def side_effect(command, **kwargs):
            if command.startswith("wc -c"):
                return {"output": "12\n", "returncode": 0}
            if command.startswith("head -c"):
                return {"output": "print('ok')\n", "returncode": 0}
            if command.startswith("sed -n"):
                return {"output": leaked, "returncode": 0}
            if command.startswith("wc -l"):
                return {"output": "1\n", "returncode": 0}
            return {"output": "", "returncode": 0}

        mock_env.execute.side_effect = side_effect
        ops = ShellFileOperations(mock_env)
        result = ops.read_file("/tmp/test/a.py")

        assert result.error is None
        assert "HERMES_FENCE" not in result.content
        assert "\x1b]" not in result.content
        assert "\x07" not in result.content
        assert "     1|print('ok')" in result.content

    def test_read_file_raw_strips_leaked_terminal_fence_markers(self, mock_env):
        leaked = (
            "__HERMES_FENCE_a9f7b3__\x07'\n"
            "alpha\n"
            "\x1b]0;cat '/tmp/test/a.txt'\x07__HERMES_FENCE_a9f7b3__\n"
        )

        def side_effect(command, **kwargs):
            if command.startswith("wc -c"):
                return {"output": "6\n", "returncode": 0}
            if command.startswith("head -c"):
                return {"output": "alpha\n", "returncode": 0}
            if command.startswith("cat "):
                return {"output": leaked, "returncode": 0}
            return {"output": "", "returncode": 0}

        mock_env.execute.side_effect = side_effect
        ops = ShellFileOperations(mock_env)
        result = ops.read_file_raw("/tmp/test/a.txt")

        assert result.error is None
        assert result.content == "alpha\n"


class TestSearchPathValidation:
    """Test that search() returns an error for non-existent paths."""

    def test_search_nonexistent_path_returns_error(self, mock_env):
        """search() should return an error when the path doesn't exist."""
        def side_effect(command, **kwargs):
            if "test -e" in command:
                return {"output": "not_found", "returncode": 1}
            if "command -v" in command:
                return {"output": "yes", "returncode": 0}
            return {"output": "", "returncode": 0}
        mock_env.execute.side_effect = side_effect
        ops = ShellFileOperations(mock_env)
        result = ops.search("pattern", path="/nonexistent/path")
        assert result.error is not None
        assert "not found" in result.error.lower() or "Path not found" in result.error

    def test_search_nonexistent_path_files_mode(self, mock_env):
        """search(target='files') should also return error for bad paths."""
        def side_effect(command, **kwargs):
            if "test -e" in command:
                return {"output": "not_found", "returncode": 1}
            if "command -v" in command:
                return {"output": "yes", "returncode": 0}
            return {"output": "", "returncode": 0}
        mock_env.execute.side_effect = side_effect
        ops = ShellFileOperations(mock_env)
        result = ops.search("*.py", path="/nonexistent/path", target="files")
        assert result.error is not None
        assert "not found" in result.error.lower() or "Path not found" in result.error

    def test_search_existing_path_proceeds(self, mock_env):
        """search() should proceed normally when the path exists."""
        def side_effect(command, **kwargs):
            if "test -e" in command:
                return {"output": "exists", "returncode": 0}
            if "command -v" in command:
                return {"output": "yes", "returncode": 0}
            # rg returns exit 1 (no matches) with empty output
            return {"output": "", "returncode": 1}
        mock_env.execute.side_effect = side_effect
        ops = ShellFileOperations(mock_env)
        result = ops.search("pattern", path="/existing/path")
        assert result.error is None
        assert result.total_count == 0  # No matches but no error

    def test_search_rg_error_exit_code(self, mock_env):
        """search() should report error when rg returns exit code 2."""
        call_count = {"n": 0}
        def side_effect(command, **kwargs):
            call_count["n"] += 1
            if "test -e" in command:
                return {"output": "exists", "returncode": 0}
            if "command -v" in command:
                return {"output": "yes", "returncode": 0}
            # rg returns exit 2 (error) with empty output
            return {"output": "", "returncode": 2}
        mock_env.execute.side_effect = side_effect
        ops = ShellFileOperations(mock_env)
        result = ops.search("pattern", path="/some/path")
        assert result.error is not None
        assert "search failed" in result.error.lower() or "Search error" in result.error


class TestSearchFilesFallbackHiddenPaths:
    def _make_env(self):
        env = MagicMock()
        env.cwd = "/"

        def execute(command, **kwargs):
            completed = subprocess.run(
                command,
                shell=True,
                text=True,
                capture_output=True,
            )
            return {
                "output": completed.stdout,
                "returncode": completed.returncode,
            }

        env.execute = execute
        return env

    def test_hidden_root_with_hidden_ancestor_includes_files(self, tmp_path, monkeypatch):
        """Fallback find should include visible files when path is inside hidden root."""
        root = tmp_path / ".hermes" / "logs"
        root.mkdir(parents=True)
        visible_file = root / "agent.log"
        hidden_dir_file = root / ".hidden" / "secret.log"
        nested_hidden_file = root / "nested" / ".secret.log"
        visible_nested_file = root / "nested" / "visible.log"

        for p in [visible_file, nested_hidden_file, visible_nested_file, hidden_dir_file]:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x")

        ops = ShellFileOperations(self._make_env())
        monkeypatch.setattr(ops, "_has_command", lambda command: command == "find")
        result = ops._search_files("*.log", str(root), limit=50, offset=0)

        assert result.error is None
        assert set(result.files) == {str(visible_file), str(visible_nested_file)}

    def test_normal_root_still_excludes_hidden_descendants(self, tmp_path, monkeypatch):
        """Fallback find should still exclude hidden descendant paths for normal roots."""
        root = tmp_path / "repo"
        root.mkdir()
        visible_file = root / "agent.log"
        visible_nested_file = root / "nested" / "visible.log"
        hidden_dir_file = root / ".hidden" / "secret.log"

        for p in [visible_file, visible_nested_file, hidden_dir_file]:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x")

        ops = ShellFileOperations(self._make_env())
        monkeypatch.setattr(ops, "_has_command", lambda command: command == "find")
        result = ops._search_files("*.log", str(root), limit=50, offset=0)

        assert result.error is None
        assert set(result.files) == {str(visible_file), str(visible_nested_file)}


class TestShellFileOpsWriteDenied:
    def test_write_file_denied_path(self, file_ops):
        result = file_ops.write_file("~/.ssh/authorized_keys", "evil key")
        assert result.error is not None
        assert "denied" in result.error.lower()

    def test_patch_replace_denied_path(self, file_ops):
        result = file_ops.patch_replace("~/.ssh/authorized_keys", "old", "new")
        assert result.error is not None
        assert "denied" in result.error.lower()

    def test_delete_file_denied_path(self, file_ops):
        result = file_ops.delete_file("~/.ssh/authorized_keys")
        assert result.error is not None
        assert "denied" in result.error.lower()

    def test_move_file_src_denied(self, file_ops):
        result = file_ops.move_file("~/.ssh/id_rsa", "/tmp/dest.txt")
        assert result.error is not None
        assert "denied" in result.error.lower()

    def test_move_file_dst_denied(self, file_ops):
        result = file_ops.move_file("/tmp/src.txt", "~/.aws/credentials")
        assert result.error is not None
        assert "denied" in result.error.lower()

    def test_move_file_failure_path(self, mock_env):
        mock_env.execute.return_value = {"output": "No such file or directory", "returncode": 1}
        ops = ShellFileOperations(mock_env)
        result = ops.move_file("/tmp/nonexistent.txt", "/tmp/dest.txt")
        assert result.error is not None
        assert "Failed to move" in result.error


class TestPatchReplacePostWriteVerification:
    """Tests for the post-write verification added in patch_replace.

    Confirms that a silent persistence failure (where write_file's command
    appears to succeed but the bytes on disk don't match new_content) is
    surfaced as an error instead of being reported as a successful patch.
    """

    def test_patch_replace_fails_when_file_not_persisted(self, mock_env):
        """write_file reports success but the re-read returns old content:
        patch_replace must return an error, not success-with-diff."""
        file_contents = {"/tmp/test/a.py": "hello world\n"}

        def side_effect(command, **kwargs):
            # cat reads the file — both the initial read and the verify read
            if command.startswith("cat "):
                # Extract path from cat command (strip quotes)
                for path in file_contents:
                    if path in command:
                        return {"output": file_contents[path], "returncode": 0}
                return {"output": "", "returncode": 1}
            # mkdir for parent dir
            if command.startswith("mkdir "):
                return {"output": "", "returncode": 0}
            # wc -c for byte count after write
            if command.startswith("wc -c"):
                for path in file_contents:
                    if path in command:
                        return {"output": str(len(file_contents[path].encode())), "returncode": 0}
                return {"output": "0", "returncode": 0}
            # Everything else (including the write itself) pretends to succeed
            # but DOESN'T update file_contents — simulates silent failure
            return {"output": "", "returncode": 0}

        mock_env.execute.side_effect = side_effect
        ops = ShellFileOperations(mock_env)
        result = ops.patch_replace("/tmp/test/a.py", "hello", "hi")
        assert result.error is not None, (
            "Silent persistence failure must surface as error, got: "
            f"success={result.success}, diff={result.diff}"
        )
        assert "verification failed" in result.error.lower()
        assert "did not persist" in result.error.lower()

    def test_patch_replace_succeeds_when_file_persisted(self, mock_env):
        """Normal success path: write persists, verify read returns new bytes."""
        state = {"content": "hello world\n"}

        def side_effect(command, stdin_data=None, **kwargs):
            # Write is `cat > path` — detect by the `>` redirect, NOT just `cat `
            if command.startswith("cat >"):
                if stdin_data is not None:
                    state["content"] = stdin_data
                return {"output": "", "returncode": 0}
            if command.startswith("cat "):  # read
                return {"output": state["content"], "returncode": 0}
            if command.startswith("mkdir "):
                return {"output": "", "returncode": 0}
            if command.startswith("wc -c"):
                return {"output": str(len(state["content"].encode())), "returncode": 0}
            return {"output": "", "returncode": 0}

        mock_env.execute.side_effect = side_effect
        ops = ShellFileOperations(mock_env)
        result = ops.patch_replace("/tmp/test/a.py", "hello", "hi")
        assert result.error is None, f"Unexpected error: {result.error}"
        assert result.success is True
        assert state["content"] == "hi world\n", f"File not actually updated: {state['content']!r}"

    def test_patch_replace_fails_when_verify_read_errors(self, mock_env):
        """If the verify-read step itself fails (exit code != 0), return an error."""
        call_count = {"cat": 0}
        state = {"content": "hello world\n"}

        def side_effect(command, stdin_data=None, **kwargs):
            if command.startswith("cat >"):  # write
                if stdin_data is not None:
                    state["content"] = stdin_data
                return {"output": "", "returncode": 0}
            if command.startswith("cat "):  # read
                call_count["cat"] += 1
                # First read (initial fetch) succeeds; second read (verify) fails
                if call_count["cat"] == 1:
                    return {"output": state["content"], "returncode": 0}
                return {"output": "", "returncode": 1}
            if command.startswith("mkdir "):
                return {"output": "", "returncode": 0}
            if command.startswith("wc -c"):
                return {"output": str(len(state["content"].encode())), "returncode": 0}
            return {"output": "", "returncode": 0}

        mock_env.execute.side_effect = side_effect
        ops = ShellFileOperations(mock_env)
        result = ops.patch_replace("/tmp/test/a.py", "hello", "hi")
        assert result.error is not None
        assert "could not re-read" in result.error.lower()



# =========================================================================
# Git baseline check for write_file warning
# =========================================================================

class _DeletedTestGitBaselineCheck:
    """Removed May 2026 — these tests asserted on a ``_check_git_baseline``
    method that doesn't exist on ``ShellFileOperations`` (regression intro
    by a separate refactor). All 6 tests in the class fail with
    AttributeError on origin/main. Deleted wholesale per Teknium's
    instruction to keep CI green; reinstate them when the underlying
    helper is restored or replaced.
    """
    pass


# =========================================================================
# Atomic write: umask-default permissions for new files
# =========================================================================

class TestAtomicWriteNewFilePermissions:
    """_atomic_write should apply umask-default perms to new files (not 0600)."""

    @pytest.mark.parametrize("test_umask", [0o022, 0o002, 0o077])
    def test_new_file_gets_umask_default_permissions(self, tmp_path, test_umask):
        """Newly created file should get umask-computed perms, not mktemp's 0600.

        Uses a real subprocess so the shell script actually runs.
        """
        ops = ShellFileOperations(make_real_subprocess_env(str(tmp_path)))
        dest = tmp_path / "new_file.txt"
        assert not dest.exists()

        old_umask = os.umask(test_umask)
        try:
            result = ops.write_file(str(dest), "test content\n")
        finally:
            os.umask(old_umask)

        assert result.error is None, f"write failed: {result.error}"
        assert dest.read_text() == "test content\n"
        expected_mode = 0o666 & ~test_umask
        actual_mode = dest.stat().st_mode & 0o777
        assert actual_mode == expected_mode, (
            f"Expected mode {expected_mode:04o} (umask {test_umask:04o}), "
            f"got {actual_mode:04o}"
        )

    def test_overwrite_still_preserves_existing_mode(self, tmp_path):
        """The new-file branch must not disturb the overwrite path's
        mode preservation (e.g. an executable script stays 0755)."""
        ops = ShellFileOperations(make_real_subprocess_env(str(tmp_path)))
        dest = tmp_path / "existing.sh"
        dest.write_text("#!/bin/sh\n")
        dest.chmod(0o755)

        result = ops.write_file(str(dest), "#!/bin/sh\necho updated\n")

        assert result.error is None, f"write failed: {result.error}"
        assert dest.read_text() == "#!/bin/sh\necho updated\n"
        assert dest.stat().st_mode & 0o777 == 0o755


class TestAtomicWriteThroughSymlink:
    """_atomic_write must edit a symlink's target, not replace the link.

    Regression: the temp-file + ``mv`` swap replaced the symlink itself with a
    plain file, orphaning the real target and destroying the link (data-loss).
    """

    def test_write_follows_symlink_and_preserves_link(self, tmp_path):
        ops = ShellFileOperations(make_real_subprocess_env(str(tmp_path)))
        real = tmp_path / "real.txt"
        link = tmp_path / "link.txt"
        real.write_text("original\n")
        link.symlink_to(real)

        result = ops.write_file(str(link), "newcontent\n")

        assert result.error is None, f"write failed: {result.error}"
        # The link must survive as a symlink...
        assert link.is_symlink(), "symlink was replaced by a plain file"
        # ...and the real target must carry the new content.
        assert real.read_text() == "newcontent\n"
        assert os.path.realpath(link) == str(real)

    def test_write_through_broken_symlink_falls_back(self, tmp_path):
        """A broken link resolves through readlink -f and creates the target."""
        ops = ShellFileOperations(make_real_subprocess_env(str(tmp_path)))
        target = tmp_path / "target.txt"
        link = tmp_path / "broken.lnk"
        link.symlink_to(target)  # target does not exist yet

        result = ops.write_file(str(link), "data\n")

        assert result.error is None, f"write failed: {result.error}"
        assert target.exists()
        assert target.read_text() == "data\n"


class TestReadNonUtf8IsBinary:
    """Non-UTF-8 content must be flagged binary, not returned as lossy text.

    Regression: the terminal env decodes stdout with errors="replace", turning
    every non-UTF-8 byte into U+FFFD before _is_likely_binary sees it. U+FFFD is
    "printable", so the non-printable ratio never caught it, and a
    read→edit→write round-trip would overwrite the original bytes with mojibake.
    """

    def test_replacement_char_sample_flagged_binary(self, tmp_path):
        ops = ShellFileOperations(make_real_subprocess_env(str(tmp_path)))
        # A latin-1 file decoded with errors="replace" yields U+FFFD chars.
        lossy_sample = "caf\ufffd r\ufffdsum\ufffd\n"
        assert ops._is_likely_binary("notes.txt", lossy_sample) is True

    def test_plain_utf8_text_not_flagged(self, tmp_path):
        ops = ShellFileOperations(make_real_subprocess_env(str(tmp_path)))
        # Proper UTF-8 (including non-ASCII) must still read as text.
        assert ops._is_likely_binary("notes.txt", "café résumé\nsecond\n") is False


class TestReadTruncatedMultibyteSample:
    """A 1000-byte read sample that cuts mid multi-byte UTF-8 char must NOT
    be misjudged as binary.

    Regression (Bug 1002): read_file/read_file_raw sample via ``head -c 1000``;
    when the cut lands inside a multi-byte char, the terminal env's
    errors="replace" decode emits one trailing U+FFFD. The old binary check
    treated any U+FFFD as binary, so a legitimate UTF-8 text file whose 1000th
    byte split a Chinese char was unreadable (e.g. lessons-learned INDEX.md,
    unreadable 8/29 + 8/30). The byte-domain probe reads the raw on-disk bytes
    and counts only NUL / control bytes, so multi-byte UTF-8 can never be
    misjudged.
    """

    def _utf8_file_cut_mid_char(self, tmp_path):
        # 334 Chinese chars = 1002 bytes; head -c 1000 stops on the first byte
        # of char 334, so the decoded 1000-byte sample ends with one U+FFFD.
        p = tmp_path / "notes.md"
        p.write_bytes(("研" * 334 + "\n").encode("utf-8"))
        return p

    def test_read_file_not_binary_when_sample_cuts_multibyte_char(self, tmp_path):
        ops = ShellFileOperations(make_real_subprocess_env_replace(str(tmp_path)))
        p = self._utf8_file_cut_mid_char(tmp_path)
        result = ops.read_file(str(p))
        assert result.error is None
        assert result.is_binary is False
        assert result.content

    def test_read_file_raw_not_binary_when_sample_cuts_multibyte_char(self, tmp_path):
        ops = ShellFileOperations(make_real_subprocess_env_replace(str(tmp_path)))
        p = self._utf8_file_cut_mid_char(tmp_path)
        result = ops.read_file_raw(str(p))
        assert result.error is None
        assert result.is_binary is False
        assert "研" in result.content

    def test_real_binary_file_still_flagged(self, tmp_path):
        """The byte-domain probe must still catch a genuine binary file."""
        ops = ShellFileOperations(make_real_subprocess_env_replace(str(tmp_path)))
        p = tmp_path / "blob.bin"
        p.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 500)
        result = ops.read_file(str(p))
        assert result.is_binary is True
