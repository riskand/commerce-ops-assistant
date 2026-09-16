# tests/test_db_url.py
"""migrations/env.py is not a package on the path, so this helper lives
in app/db_url.py (importable via the pythonpath the tests already use)
rather than in migrations/ itself."""
import pytest

from app.db_url import to_psycopg_url


@pytest.mark.parametrize("given", [
    "postgresql://postgres:dev@localhost/docsearch",
    "postgresql+asyncpg://postgres:dev@localhost/docsearch",
    "postgresql+psycopg://postgres:dev@localhost/docsearch",
])
def test_any_postgres_driver_normalises_to_psycopg(given):
    assert (to_psycopg_url(given) ==
            "postgresql+psycopg://postgres:dev@localhost/docsearch")


def test_a_nonsense_scheme_raises():
    with pytest.raises(ValueError, match="DATABASE_URL"):
        to_psycopg_url("mysql://postgres:dev@localhost/docsearch")


def test_a_non_url_raises():
    with pytest.raises(ValueError, match="DATABASE_URL"):
        to_psycopg_url("not-a-url-at-all")
