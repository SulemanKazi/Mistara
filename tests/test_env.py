"""`.env` loading.

A key that silently fails to load surfaces as a confusing auth error a long way
from the cause, so the parsing edge cases are pinned here.
"""

from __future__ import annotations

import os

import pytest

from mistara.core.env import (
    credential_status,
    find_dotenv,
    load_dotenv,
    parse_dotenv,
)


class TestParsing:
    def test_plain_assignment(self):
        assert parse_dotenv("ANTHROPIC_API_KEY=sk-ant-123") == {
            "ANTHROPIC_API_KEY": "sk-ant-123"
        }

    def test_export_prefix_is_tolerated(self):
        assert parse_dotenv("export FOO=bar") == {"FOO": "bar"}

    @pytest.mark.parametrize("quote", ['"', "'"])
    def test_quotes_are_stripped(self, quote):
        assert parse_dotenv(f"FOO={quote}bar baz{quote}") == {"FOO": "bar baz"}

    def test_comments_and_blank_lines_are_skipped(self):
        text = "# a comment\n\nFOO=bar\n   # indented comment\nBAZ=qux\n"
        assert parse_dotenv(text) == {"FOO": "bar", "BAZ": "qux"}

    def test_trailing_comment_on_unquoted_value(self):
        assert parse_dotenv("FOO=bar # why")["FOO"] == "bar"

    def test_hash_inside_quotes_survives(self):
        """API keys can contain anything; a quoted `#` is part of the secret."""
        assert parse_dotenv('FOO="se#cret"')["FOO"] == "se#cret"

    def test_equals_in_value_is_preserved(self):
        assert parse_dotenv("FOO=a=b=c")["FOO"] == "a=b=c"

    def test_lines_without_an_equals_are_ignored(self):
        assert parse_dotenv("just some noise\nFOO=bar") == {"FOO": "bar"}

    def test_empty_value_is_allowed(self):
        assert parse_dotenv("FOO=") == {"FOO": ""}


class TestLoading:
    def test_values_land_in_the_environment(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text("MISTARA_TEST_KEY=from-file\n")
        monkeypatch.delenv("MISTARA_TEST_KEY", raising=False)
        assert load_dotenv(env) == env
        assert os.environ["MISTARA_TEST_KEY"] == "from-file"

    def test_real_environment_wins_by_default(self, tmp_path, monkeypatch):
        """So a one-off override works without editing the file."""
        env = tmp_path / ".env"
        env.write_text("MISTARA_TEST_KEY=from-file\n")
        monkeypatch.setenv("MISTARA_TEST_KEY", "from-shell")
        load_dotenv(env)
        assert os.environ["MISTARA_TEST_KEY"] == "from-shell"

    def test_override_is_opt_in(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text("MISTARA_TEST_KEY=from-file\n")
        monkeypatch.setenv("MISTARA_TEST_KEY", "from-shell")
        load_dotenv(env, override=True)
        assert os.environ["MISTARA_TEST_KEY"] == "from-file"

    def test_missing_file_is_not_an_error(self, tmp_path):
        assert load_dotenv(tmp_path / "nope.env") is None

    def test_search_walks_up_from_a_subdirectory(self, tmp_path):
        (tmp_path / ".env").write_text("FOO=bar\n")
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        assert find_dotenv(nested) == tmp_path / ".env"

    def test_search_returns_none_when_absent(self, tmp_path):
        assert find_dotenv(tmp_path) is None


class TestCredentialStatus:
    def test_reports_presence_without_revealing_the_secret(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-do-not-print-me")
        status = credential_status()
        assert status["anthropic"] == "set via ANTHROPIC_API_KEY"
        assert "sk-ant-do-not-print-me" not in " ".join(status.values())

    def test_absent_credentials_say_so(self, tmp_path, monkeypatch):
        # Config dir must be redirected too, or a real `ant auth login` profile
        # on the developer's machine leaks into the assertion.
        for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path))
        assert credential_status()["anthropic"] == "not set"

    def test_auth_token_is_recognised_as_an_alternative(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path))
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "oauth-token")
        assert credential_status()["anthropic"] == "set via ANTHROPIC_AUTH_TOKEN"

    def test_an_ant_profile_counts_as_credentials(self, tmp_path, monkeypatch):
        """The SDK falls back to a stored OAuth profile, so reporting only on
        env vars would tell a logged-in user their credentials are missing."""
        for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
            monkeypatch.delenv(name, raising=False)
        creds = tmp_path / "credentials"
        creds.mkdir()
        (creds / "default.json").write_text("{}")
        monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path))

        assert credential_status()["anthropic"] == "set via ant profile (default)"

    def test_an_explicit_key_takes_precedence_over_a_profile(
        self, tmp_path, monkeypatch
    ):
        creds = tmp_path / "credentials"
        creds.mkdir()
        (creds / "default.json").write_text("{}")
        monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-explicit")
        assert credential_status()["anthropic"] == "set via ANTHROPIC_API_KEY"

    def test_no_profile_directory_is_not_an_error(self, tmp_path, monkeypatch):
        for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path / "absent"))
        assert credential_status()["anthropic"] == "not set"
