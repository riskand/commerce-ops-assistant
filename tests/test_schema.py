# tests/test_schema.py
import subprocess

import pytest
import sqlalchemy

from app.db import DATABASE_URL
from app.embedder import DIMS
from app.models import Chunk


def test_the_python_attribute_and_the_column_name_differ():
    # Mapper.columns is keyed by the Python/ORM attribute name ("meta"),
    # while the column object it holds carries the real database column
    # name ("metadata"). Assert both halves so either one drifting back
    # into agreement would fail this test.
    assert "meta" in Chunk.__mapper__.columns.keys()
    assert Chunk.__mapper__.columns["meta"].name == "metadata"


def test_the_vector_column_matches_the_embedder():
    """DIMS and Vector(1024) are one decision written in two places."""
    assert Chunk.__table__.c.embedding.type.dim == DIMS


@pytest.mark.integration
def test_upgrade_head_produces_the_declared_columns():
    subprocess.run([".venv/bin/alembic", "upgrade", "head"],
                   check=True, capture_output=True)
    sync_url = DATABASE_URL.replace("+asyncpg", "+psycopg")
    engine = sqlalchemy.create_engine(sync_url)
    inspector = sqlalchemy.inspect(engine)
    actual = {c["name"] for c in inspector.get_columns("chunks")}
    engine.dispose()
    expected = set(Chunk.__table__.columns.keys())
    assert expected == actual
