# app/models.py
import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (BigInteger, Boolean, DateTime, FetchedValue, Index,
                        Integer, Text, UniqueConstraint, func, text)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        Index("ix_chunks_fts", "fts", postgresql_using="gin"),
    )

    id:          Mapped[int] = mapped_column(BigInteger, primary_key=True)
    tenant_id:   Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    ordinal:     Mapped[int] = mapped_column(Integer)
    content:     Mapped[str] = mapped_column(Text)
    embedding:   Mapped[list[float]] = mapped_column(Vector(1024))  # model's dims
    meta:        Mapped[dict] = mapped_column(
                     "metadata", JSONB, server_default=text("'{}'"))
    fts:         Mapped[str] = mapped_column(TSVECTOR, nullable=True,
                                             server_default=FetchedValue())
    embedding_model: Mapped[str] = mapped_column(   # per chunk, not per document
                         Text, server_default=text("'unknown'"))


class Document(Base):
    __tablename__ = "documents"

    id:         Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True),
                                                  primary_key=True)
    tenant_id:  Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    body:       Mapped[str] = mapped_column(Text)   # "text" would shadow text()
    status:     Mapped[str] = mapped_column(Text,
                                            server_default=text("'queued'"))
    chunks:     Mapped[int] = mapped_column(Integer, server_default=text("0"))
    error:      Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
                    DateTime(timezone=True), server_default=func.now())


class Order(Base):
    __tablename__ = "orders"

    id:         Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True),
                                                  primary_key=True)
    tenant_id:  Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    order_ref:  Mapped[str] = mapped_column(Text, index=True)   # e.g. "SO-1042"
    status:     Mapped[str] = mapped_column(Text)
    channel:    Mapped[str] = mapped_column(Text)
    line_items: Mapped[list] = mapped_column(JSONB, server_default=text("'[]'"))


class SyncError(Base):
    __tablename__ = "sync_errors"

    id:         Mapped[int] = mapped_column(BigInteger, primary_key=True)
    tenant_id:  Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    order_ref:  Mapped[str] = mapped_column(Text)
    channel:    Mapped[str] = mapped_column(Text, index=True)
    code:       Mapped[str] = mapped_column(Text)               # e.g. "8541"
    detail:     Mapped[str] = mapped_column(Text)
    occurred_at: Mapped[datetime] = mapped_column(
                     DateTime(timezone=True), server_default=func.now())


class PromptVersion(Base):
    __tablename__ = "prompt_versions"
    # The one-active-version-per-name rule is a partial unique index, so
    # Postgres refuses a second active row rather than trusting every
    # future caller to deactivate first. Declared here as well as in the
    # migration for the reason ix_chunks_fts taught this repo: Alembic
    # diffs the model against the database, so an index the model does
    # not know about reads as drift and the next --autogenerate proposes
    # dropping it.
    __table_args__ = (
        UniqueConstraint("name", "version",
                         name="uq_prompt_versions_name_version"),
        Index("ix_prompt_versions_one_active", "name", unique=True,
              postgresql_where=text("active")),
    )

    id:         Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name:       Mapped[str] = mapped_column(Text, index=True)
    version:    Mapped[int] = mapped_column(Integer)
    template:   Mapped[str] = mapped_column(Text)
    # Derived from the template, stored so the API can show a prompt's
    # contract without parsing it. app/prompts.py validates renders
    # against the template itself, never against this column, so the two
    # can never disagree about behaviour -- a test asserts they agree
    # about documentation.
    variables:  Mapped[list] = mapped_column(JSONB, server_default=text("'[]'"))
    active:     Mapped[bool] = mapped_column(Boolean,
                                             server_default=text("false"))
    created_at: Mapped[datetime] = mapped_column(
                    DateTime(timezone=True), server_default=func.now())
