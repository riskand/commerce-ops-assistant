# tests/test_config.py
"""Settings.model_config sets extra="ignore" precisely so that a .env
file carrying keys Settings doesn't declare (DATABASE_URL, say) does not
crash every import of app.config. Without it, pydantic-settings' default
of extra="forbid" applies to dotenv keys too, and anyone who puts
DATABASE_URL in the same .env as the API key gets a ValidationError with
no obvious cause."""
from app.config import Settings


def test_an_unrelated_dotenv_key_does_not_raise(tmp_path, monkeypatch):
    # Isolate from the real process environment (conftest.py has already
    # loaded the real .env) so this only exercises the temp dotenv below.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "ANTHROPIC_API_KEY=test-key\n"
        "DATABASE_URL=postgresql+asyncpg://postgres:dev@localhost/docsearch\n"
    )

    settings = Settings(_env_file=dotenv)

    assert settings.anthropic_api_key == "test-key"


def test_default_mcp_server_forwards_tenant_and_database_from_settings(
        tmp_path, monkeypatch):
    """The MCP stdio client hands a spawned server get_default_environment()
    -- HOME, LOGNAME, PATH, SHELL, TERM, USER -- and nothing else unless
    the caller passes an explicit env, so MCP_TENANT_ID and DATABASE_URL
    exported in this process's own environment (as opposed to sitting in
    a .env file in the child's cwd) never reached the server subprocess
    before this test's fix: _default_mcp_servers() built no "env" key at
    all, and the server silently fell back to the hardcoded demo tenant
    and localhost/docsearch. An operator who scopes a deployment purely
    by environment variable got another tenant's rows served to the
    model, with no error. This test fails against that code: an empty
    dotenv (isolated from the real .env with a nonexistent path) plus
    process-environment MCP_TENANT_ID/DATABASE_URL is exactly the
    "environment, not .env file" deployment shape mcp_server/README.md
    documents as supported."""
    monkeypatch.setenv("MCP_TENANT_ID", "22222222-2222-4222-8222-222222222222")
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:dev@localhost/other_tenant_db")

    settings = Settings(_env_file=tmp_path / "no-such-.env")

    server = settings.mcp_servers[0]
    assert server["env"]["MCP_TENANT_ID"] == (
        "22222222-2222-4222-8222-222222222222")
    assert server["env"]["DATABASE_URL"] == (
        "postgresql+asyncpg://postgres:dev@localhost/other_tenant_db")
