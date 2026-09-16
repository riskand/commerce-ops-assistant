# commerce-ops-assistant

A multi-tenant operations assistant for a commerce platform, built directly
against the Anthropic Messages API and the Model Context Protocol rather than
on top of an agent framework.

It answers operational questions over two kinds of source: a document corpus
searched with hybrid retrieval, and live platform data reached through MCP
tools. Everything that talks to a model goes through one gateway function with
bounded retries and a fallback model, every prompt is a versioned row rather
than a string literal, and the eval suite runs as a build gate.

The domain is deliberately concrete, because a service with no domain is hard
to judge: sellers listing on several marketplaces, orders that fail to sync,
and the policy documents an operator has to read to work out why.

## What is in here

| Path | What it does |
|---|---|
| `app/loop.py` | The only function that talks to the model. Bounded retries by status class, exponential backoff with jitter, `Retry-After` honoured, fallback model on sustained failure, and one structured JSON line per call recording latency, tokens and cost |
| `app/routes/ask.py` | An agent tool-use loop over a fixed local tool schema, with a step budget |
| `app/routes/chat.py` | The same loop over tools discovered at runtime from connected MCP servers, with a prompt-injection guard bound to the gateway |
| `app/routes/docsearch.py` | Ingest, document status, and query. Ingestion runs off the request path |
| `app/routes/prompts.py` | The prompt registry: list, read, diff two versions, activate one |
| `app/retrieval.py` | Hybrid search. Postgres full text and pgvector similarity fused with Reciprocal Rank Fusion, then cross-encoder reranking over the shortlist |
| `app/chunker.py` | Paragraph and sentence aware chunking with overlap |
| `app/mcp_client.py` | MCP tool definitions translated to the Anthropic tools array, call routing, per-call timeouts and a per-server circuit breaker |
| `mcp_server/` | Three tools over the same database: `search_docs`, `get_order`, `list_sync_errors` |
| `evals/` | An eval harness with an LLM judge, a recorded baseline, and `python -m evals.gate` which exits non-zero when a metric regresses |
| `serving/` | The embedding and reranking tier in its own process, so the API container carries no torch. Speaks an OpenAI-compatible generation endpoint too |
| `experiments/` | Two measurement harnesses: does the model call the right tool, and does a tool result get obeyed as an instruction |
| `migrations/` | Alembic. pgvector columns, a generated full text column, prompt versions |

## Tenancy

Every query is scoped to a tenant at the database boundary, not by filtering
afterwards. The MCP server resolves its tenant once at startup from its own
environment and never accepts one as a tool argument, so a model cannot ask a
question on another tenant's behalf. `mcp_server/README.md` sets out what has
to change for a remote, multi-caller deployment.

## Running it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env            # add a real ANTHROPIC_API_KEY
docker compose up -d            # postgres with pgvector, and pgadmin
.venv/bin/alembic upgrade head
.venv/bin/uvicorn app.main:app --reload
```

The in-process embedder and reranker are a separate install, because they pull
torch:

```bash
.venv/bin/pip install -r requirements-local.txt
```

Without them, point `SERVING_URL` at the `serving/` process instead and the
API container stays small.

## Tests

```bash
.venv/bin/pytest -q                      # fast tier: no database, no model, no key
.venv/bin/pytest -q -m integration -k "not live"   # needs docker compose up -d
.venv/bin/pytest -q -m integration -k live         # needs a real key and the local models
```

The fast tier is the default in `pyproject.toml` and is the one a build should
fail on first: it needs nothing, so a broken import is caught in seconds rather
than after the evals have spent money finding out. `.github/workflows/evals.yml`
runs both tiers and then the eval gate.

## Evals

```bash
python -m evals.corpus                   # load the corpus
python -m evals.run --out results.json   # recall@k, and an LLM judge on answers
python -m evals.gate results.json        # non-zero when a metric drops below baseline
```

Prompts are code, so they get a test suite, and a test suite nothing fails a
build over is a report nobody reads.

## Notes on a few decisions

**One gateway.** Every model call in the application goes through
`app.loop.call_llm`. A defence, a route or an experiment that builds its own
request is a defence that skips the retry layer and fails whenever the API is
briefly busy.

**The injection guard is not in the prompt registry.** Every other prompt is,
and the registry ships an HTTP endpoint that edits prompts. Putting a security
control behind an endpoint that can rewrite it hands anyone who reaches the API
a way to turn it off, so that one prompt lives in source, where changing it is a
deploy and a code review.

**Tool results are data, never instructions.** `experiments/injection.py`
measures whether that actually holds, by planting a poisoned document and asking
a question designed to retrieve it, with the guard and without.

**Circuit breaker per MCP server.** One flaky server should cost one call, not
every chat turn, so a server is cut off after consecutive failures and let back
in by a single probe.

## Licence

MIT. See `LICENSE`.
