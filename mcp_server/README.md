# mcp_server — saas-tools

Three tools over the same `orders`, `sync_errors` and `chunks` tables a
support engineer would query directly: `search_docs`, `get_order`,
`list_sync_errors`. Run it:

    cd examples/ai-lab
    python -m mcp_server.server

It speaks MCP over stdio — one process per conversation, no network
surface. `docker compose up -d` first; the tools hit the same Postgres
the rest of `app/` uses.

## Tenant scoping

None of the three tools takes a `tenant_id` argument. Each one calls a
private `_tenant()` helper that reads `settings.mcp_tenant_id`
(`app/config.py`) instead:

```python
def _tenant() -> uuid.UUID:
    """The tenant this server speaks for, fixed at startup."""
    return uuid.UUID(settings.mcp_tenant_id)
```

This is deliberate, not an oversight. Every argument a tool schema
exposes is a value the model can be talked into supplying — the model
reads the tool's name, description and schema and nothing else, and a
`tenant_id` parameter is exactly the kind of thing a confused or
adversarial prompt can override. A support engineer asking "what's
wrong with order SO-1042" never intends to also grant the model the
choice of *whose* SO-1042 to look up. So authorization for tenant scope
lives in the server, resolved once from configuration the model never
sees or touches, not in an argument the model fills in.
`tests/test_mcp_tools.py` holds this as a regression test:
`test_no_tool_accepts_a_tenant_argument`,
`test_tools_cannot_reach_another_tenants_rows`, and
`test_search_docs_cannot_reach_another_tenants_chunks`, which covers
the one tool whose corpus (`chunks`) is shared across every tenant, so
a wrong variable in the line that forwards `_tenant()` into
`hybrid_search` would leak silently rather than error.

**What a local stdio server implies.** This process has exactly one
caller — whoever spawned it — and exactly one tenant, fixed the moment
it starts: `mcp_tenant_id` is read once from a `.env` file sitting in
this process's own working directory, at import time, and never changes
for the life of the process. Exporting `MCP_TENANT_ID` in the *parent*
process's environment is not enough on its own — the MCP stdio client
hands a spawned server a fixed, minimal environment (`HOME`, `LOGNAME`,
`PATH`, `SHELL`, `TERM`, `USER`) and nothing else unless the caller
passes one explicitly — which is why `app/config.py`'s default
`mcp_servers` entry forwards `MCP_TENANT_ID` and `DATABASE_URL` from its
own already-resolved settings rather than leaving the child to inherit
them. That is enough for a developer's laptop or a single-tenant
deployment, where "which tenant" is a fact about the deployment, not the
request.

**What changes for a remote HTTP service.** This server is stdio
only — no network surface, no caller identity beyond "whoever can spawn
this process." A remote server (streamable HTTP) drops that assumption entirely: many callers
share one running server, so tenant identity can no longer be a
startup-time constant. It has to come from the authenticated caller on
*each* request — an OAuth token or API key the transport validates
before a tool ever executes — never from a request parameter, and never
from configuration shared across tenants. `_tenant()` in that world
reads the current request's validated principal, not `settings`. The
same rule holds for any worker's credentials: resolve tenant scope per
execution, at the boundary that has actual
authority to say who is asking, and never hand the model a parameter
that could ask on someone else's behalf.

## Registering with a real host

```bash
cd examples/ai-lab
claude mcp add saas-tools -- .venv/bin/python -m mcp_server.server
```

## Inspecting it by hand

```bash
cd examples/ai-lab
npx @modelcontextprotocol/inspector .venv/bin/python -m mcp_server.server
```

Connect, open **Tools**, and confirm all three descriptions read
unambiguously — that rendering is exactly how the model sees them, and
is the fastest way to catch a vague description before a model
misuses the tool it names.
