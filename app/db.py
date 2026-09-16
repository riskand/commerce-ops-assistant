# app/db.py
"""Engine and session factory. The URL comes from app.config.settings."""
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import settings

DATABASE_URL = settings.database_url

_pool_kwargs = ({"poolclass": NullPool} if settings.db_pool_size == 0
                else {"pool_size": settings.db_pool_size})

engine = create_async_engine(DATABASE_URL, **_pool_kwargs)
Session = async_sessionmaker(engine, expire_on_commit=False)
