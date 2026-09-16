# app/db_url.py
"""Alembic runs synchronously on psycopg; the app runs async on asyncpg
(see app/db.py). DATABASE_URL can arrive in any of the forms SQLAlchemy
accepts for Postgres — a bare "postgresql://", the async "+asyncpg" one,
or already "+psycopg" — and a naive `.replace("+asyncpg", "+psycopg")`
silently no-ops on the first of those, leaving Alembic to guess a DBAPI.
to_psycopg_url() makes the driver explicit instead of incidental."""

_DIALECT = "postgresql"
_SYNC_DRIVER = "psycopg"


def to_psycopg_url(database_url: str) -> str:
    """Normalise a Postgres DATABASE_URL to the +psycopg driver.

    Raises ValueError, naming DATABASE_URL and the offending value, for
    anything that isn't a postgresql:// URL of some driver."""
    scheme, sep, rest = database_url.partition("://")
    dialect = scheme.split("+", 1)[0]
    if not sep or dialect != _DIALECT:
        raise ValueError(
            f"DATABASE_URL must be a {_DIALECT}:// URL, "
            f"got {database_url!r}")
    return f"{_DIALECT}+{_SYNC_DRIVER}://{rest}"
