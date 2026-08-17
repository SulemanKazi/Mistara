"""Minimal `.env` loading.

Deliberately hand-rolled rather than pulling a dependency: the format we need is
a few lines of `KEY=value`, and the failure mode of a missing key is a confusing
auth error, so it is worth having the parsing be something we can read.

Real environment variables always win over the file. That ordering matters — it
lets you override a checked-in default for one command without editing anything:

    ANTHROPIC_API_KEY=sk-other uv run mistara extract ...
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILENAME = ".env"

#: How far up the tree to look. Enough to find the project root from a
#: subdirectory without wandering into the user's home.
_MAX_PARENTS = 4


def find_dotenv(start: Path | None = None) -> Path | None:
    """Find the nearest `.env`, walking up from `start` (default: cwd)."""
    here = (start or Path.cwd()).resolve()
    for directory in [here, *list(here.parents)[:_MAX_PARENTS]]:
        candidate = directory / ENV_FILENAME
        if candidate.is_file():
            return candidate
    return None


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse `.env` content. Tolerates `export`, quotes, comments, blank lines."""
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            # Only strip trailing comments on unquoted values; a `#` inside
            # quotes is part of the secret.
            value = value.split(" #", 1)[0].rstrip()
        values[key] = value
    return values


def load_dotenv(path: Path | str | None = None, *, override: bool = False) -> Path | None:
    """Load a `.env` into `os.environ`. Returns the file used, or None.

    Existing environment variables are left alone unless `override` is set.
    """
    target = Path(path) if path is not None else find_dotenv()
    if target is None or not target.is_file():
        return None
    for key, value in parse_dotenv(target.read_text(encoding="utf-8")).items():
        if override or key not in os.environ:
            os.environ[key] = value
    return target


def anthropic_profile() -> Path | None:
    """Find an `ant auth login` profile, if one exists.

    The Anthropic SDK resolves credentials as API key → auth token → stored
    OAuth profile, so a profile alone is sufficient and no environment variable
    need be set. Reporting only on env vars would tell a logged-in user their
    credentials are missing.
    """
    config_dir = Path(
        os.environ.get("ANTHROPIC_CONFIG_DIR", Path.home() / ".config" / "anthropic")
    )
    profile = os.environ.get("ANTHROPIC_PROFILE", "default")
    candidate = config_dir / "credentials" / f"{profile}.json"
    if candidate.is_file():
        return candidate
    creds = config_dir / "credentials"
    if creds.is_dir():
        return next(iter(sorted(creds.glob("*.json"))), None)
    return None


def credential_status() -> dict[str, str]:
    """Report which provider credentials are visible, without revealing them."""
    out: dict[str, str] = {}
    for label, names in {
        "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
        "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        "openai": ("OPENAI_API_KEY",),
    }.items():
        found = next((n for n in names if os.environ.get(n)), None)
        out[label] = f"set via {found}" if found else "not set"

    if out["anthropic"] == "not set" and (profile := anthropic_profile()) is not None:
        out["anthropic"] = f"set via ant profile ({profile.stem})"
    return out
