# app/config.py
import sys
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).resolve().parent.parent


def _default_mcp_servers(data: dict) -> list[dict]:
    """This repo's own MCP server, launched as the tests launch it."""
    return [{"name": "saas-tools", "command": sys.executable,
             "args": ["-m", "mcp_server.server"], "cwd": str(_ROOT),
             "env": {"MCP_TENANT_ID": data["mcp_tenant_id"],
                     "DATABASE_URL": data["database_url"]}}]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    anthropic_api_key: str = ""
    llm_api_url: str = "https://api.anthropic.com/v1/messages"
    llm_model: str = "claude-sonnet-5"
    llm_fallback_model: str = "claude-haiku-4-5-20251001"
    app_url: str = "http://localhost:8000"   # this API, as scripts reach it
    database_url: str = (
        "postgresql+asyncpg://postgres:dev@localhost/docsearch")
    db_pool_size: int = 10   # 0 selects NullPool in app/db.py
    serving_url: str = ""    # blank: models run in this process
    vllm_url: str = ""       # blank: no local generator
    vllm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    embedding_model: str = ""   # blank: app/embedder.py's MODEL
    min_rerank_score: float = -5.0   # measured against the eval set
    contextual_retrieval: bool = False   # measured against the eval set
    mcp_tenant_id: str = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    mcp_servers: list[dict] = Field(default_factory=_default_mcp_servers)
    mcp_tool_allowlist: list[str] = []   # empty: every discovered tool

settings = Settings()
