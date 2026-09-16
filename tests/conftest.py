# tests/conftest.py
"""Loads .env (e.g. a machine-local DATABASE_URL port override) before
anything below imports app.*, so os.environ is fully set up first and
app.db builds its engine against the right database. .env does not exist
in CI or on a fresh checkout; that's fine — anthropic_api_key defaults to
blank, so app.config.settings still builds, and the fast tier never spends
the placeholder key set below anyway: call_llm is monkeypatched
everywhere."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(path):
        if not path.exists():
            return
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-never-sent")

# Root-cause fix, not a symptom patch: app.db's engine is a module-level
# singleton, but pytest-asyncio hands each async test its own event loop by
# default, and a pooled asyncpg connection opened on one loop cannot be
# reused on another ("Future ... attached to a different loop"). Settings'
# db_pool_size=0 selects NullPool (app/db.py), which makes every checkout a
# fresh connection that's never handed to a later loop, so the fast and
# integration tiers can even run in one invocation. Must be set before
# anything below (or any test module) imports app.db.
os.environ.setdefault("DB_POOL_SIZE", "0")

import pytest
from sqlalchemy import delete


@pytest.fixture(autouse=True)
def _calls_log_to_tmp(tmp_path, monkeypatch):
    """call_llm appends a line to app.loop.CALLS_LOG,
    which resolves to calls.jsonl at the project root. Autouse rather than
    opt-in because a test that forgets it does not fail -- it quietly
    writes production-shaped telemetry into the repo, and nobody can tell
    those lines from real ones afterwards."""
    from app import loop
    monkeypatch.setattr(loop, "CALLS_LOG", tmp_path / "calls.jsonl")


@pytest.fixture(autouse=True)
def _no_mcp_servers_by_default(monkeypatch):
    """app.routes.chat's lifespan spawns a subprocess per entry in
    settings.mcp_servers, and app.config.py's own default points at this
    repo's own MCP server so a real `uvicorn app.main:app` run works out
    of the box. Every test that builds a TestClient(app) — including
    test_ask_route.py and test_docsearch_route.py, which know nothing
    about MCP — runs that same lifespan, so blanking the list here keeps
    the whole suite subprocess-free by default. A test that wants a real
    connection starts one directly, the way test_mcp_protocol.py does,
    rather than through the app lifespan."""
    from app.config import settings
    monkeypatch.setattr(settings, "mcp_servers", [])


@pytest.fixture
async def clean_chunks_table():
    """Opt-in, not autouse: tests that insert
    real rows and never delete them would make `chunks` grow every run,
    so any test that writes to it should request this fixture and get a
    truncate afterward. Making it autouse for every `integration`-marked
    test was wrong — most integration tests (the live embedder, the live
    API call, the alembic checks) never touch the table, and an autouse
    truncate made a reachable database a hidden prerequisite even for
    those, contradicting the rule that the live-model and
    live-API tests need only a model and a key. A DB test whose truncate
    can't reach the database should still fail loudly, not swallow the
    error — so this does not catch anything."""
    yield
    from app.db import Session
    from app.models import Chunk, Document
    async with Session() as session, session.begin():
        await session.execute(delete(Chunk))
        await session.execute(delete(Document))
