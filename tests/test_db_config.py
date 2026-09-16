# tests/test_db_config.py
"""app.db.DATABASE_URL must be sourced from app.config.settings, not
os.environ. pydantic-settings loads a dotenv file into the Settings object
without ever writing it into os.environ, so an `os.environ.get(...)` read
in app/db.py silently ignores .env in any process that did not separately
export it (e.g. a real `uvicorn app.main:app` run — tests/conftest.py
loads .env into os.environ itself, which is why the fast tier never
caught this)."""
import os
import subprocess
from pathlib import Path

import pytest

from app import config as config_module
from app import db as db_module


def test_settings_database_url_comes_from_env_file_not_os_environ(
        tmp_path, monkeypatch):
    """A Settings built against a throwaway .env file returns that file's
    database_url — proving app/db.py's DATABASE_URL is sourced from the
    Settings object, not from os.environ. No need to reload app.db to
    prove it: a reload would rebind a *new* engine/Session on the module,
    but every module that already did `from app.db import Session`
    (app/routes/docsearch.py among them) would go on using the old ones,
    silently un-disposing the real engine's pool for the rest of the
    session. Constructing a Settings directly proves the same behaviour
    with no module-state mutation that outlives this test.

    This fails if app/db.py reverts to os.environ.get: os.environ has
    DATABASE_URL deleted below, so that read would fall through to the
    hardcoded default instead of the temp file's value."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "anthropic_api_key=test-key\n"
        "database_url=postgresql+asyncpg://u:p@otherhost/otherdb\n")

    fresh_settings = config_module.Settings(_env_file=str(env_file))

    assert "DATABASE_URL" not in os.environ
    assert fresh_settings.database_url == (
        "postgresql+asyncpg://u:p@otherhost/otherdb")


@pytest.mark.integration
def test_alembic_upgrade_head_works_without_an_anthropic_key(tmp_path):
    """migrations/env.py imports app.db, which used to import
    app.config.settings — a full Settings() with anthropic_api_key
    required. That made `alembic upgrade head` fail outright in any
    directory with no .env and no ANTHROPIC_API_KEY exported, coupling a
    database migration to an LLM key it never uses. Reproduces exactly
    that directory without ever touching the real .env: pydantic-settings
    resolves env_file=".env" against the process's current working
    directory, not against app/config.py, so running alembic
    with its cwd set to an empty tmp_path makes Settings() see no .env at
    all, the same as a fresh checkout. alembic.ini's script_location uses
    the %(here)s token, so it still finds migrations/ regardless of cwd —
    passed explicitly via -c so this doesn't depend on alembic's own
    default of looking for alembic.ini in the current directory."""
    root = Path(__file__).resolve().parent.parent
    clean_env = {"PATH": os.environ.get("PATH", ""),
                 "DATABASE_URL": db_module.DATABASE_URL}
    result = subprocess.run(
        [str(root / ".venv/bin/alembic"), "-c", str(root / "alembic.ini"),
         "upgrade", "head"],
        cwd=tmp_path, env=clean_env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "ValidationError" not in result.stderr
    assert "anthropic" not in result.stderr.lower()
