# Contributing to ScyllaVec

Thank you for your interest in contributing! This document explains how to get
involved.

## Quick start

```bash
# Clone and start the full stack
git clone <repo-url>
cd scyllavec
bash scripts/dev.sh

# Create a virtual environment and install dev dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[langchain,llamaindex]"
pip install pytest httpx grpcio

# Run unit tests (no running stack required)
pytest tests/test_sharding.py tests/test_langchain.py tests/test_llamaindex.py -v

# Run integration tests (requires running stack)
pytest tests/test_api.py -v
```

## Ways to contribute

- **Bug reports** — open an issue with reproduction steps and the output of
  `GET /health`
- **Feature requests** — open an issue describing the use-case before writing
  code
- **Pull requests** — see the workflow below

## Pull request workflow

1. Fork the repo and create a branch: `git checkout -b feat/my-feature`
2. Make your changes following the conventions below
3. Add or update tests — PRs that reduce coverage will not be merged
4. Run the full test suite locally and confirm it passes
5. Open a PR against `main` with a clear description of what and why

## Code conventions

### Python
- Follow existing style (no strict formatter enforced yet, but keep it
  readable)
- Type-annotate all public functions
- Use `from __future__ import annotations` at the top of every new file

### Rust
- Run `cargo fmt` before committing
- Run `cargo clippy -- -D warnings` and fix any warnings

### Commits
- One logical change per commit
- Short imperative subject line (`fix: …`, `feat: …`, `docs: …`)

## Architecture overview

```
Client
  │  REST (port 8000)
  ▼
FastAPI (api/)          ← Python, handles auth-less REST + metadata filtering
  │
  ├─ CQL ──────────► ScyllaDB   ← vector metadata + HNSW snapshots/WAL
  │
  └─ gRPC ──┬──────► sidecar-0  ← Rust, in-memory HNSW shard 0
             └──────► sidecar-1  ← Rust, in-memory HNSW shard 1
```

Routing is deterministic: `hash(vector_id) % n_shards` for writes;
parallel fan-out + merge for searches.

## Project structure

| Path | What lives there |
|------|-----------------|
| `api/` | FastAPI service + shard-aware gRPC client |
| `sidecar/` | Rust HNSW sidecar (one binary, run N times) |
| `sdk/scyllavec/` | Python SDK + LangChain + LlamaIndex adapters |
| `tests/` | Unit tests (mocked) + integration tests |
| `benchmarks/` | Comparison harness vs pgvector, Chroma, Weaviate |
| `schema/` | ScyllaDB CQL schema |
| `scripts/` | `dev.sh` one-command startup, `seed.py` demo data |

## Reporting security issues

Please **do not** open a public issue for security vulnerabilities. Email the
maintainers directly (add contact here before going public).

## License

By contributing you agree that your contributions will be licensed under the
[Apache 2.0 License](LICENSE).
