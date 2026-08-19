# ScyllaVec

A vector store built on [ScyllaDB](https://www.scylladb.com/) — store, index, and search high-dimensional embeddings at scale, co-located with your operational data.

## Architecture

```
Client / AI Agent
      │
      ▼  REST (port 8000)
  FastAPI (Python)
      │                  ╔════════════════════════╗
      ├─ CQL ──────────► ║  ScyllaDB  (port 9042) ║
      │                  ║  vectors               ║
      └─ gRPC ─────────► ║  collections           ║
  Rust Sidecar           ║  index_snapshots       ║
  (port 50051)           ║  index_wal             ║
  in-memory HNSW         ╚════════════════════════╝
```

## Quick Start (local)

**Prerequisites:** Docker + Docker Compose, Python 3.11+

```bash
# Start everything (ScyllaDB + Rust sidecar + Python API)
bash scripts/dev.sh
```

That single command:
1. Starts ScyllaDB via Docker
2. Builds and starts the Rust HNSW sidecar
3. Builds and starts the Python FastAPI service
4. Waits for ScyllaDB to be healthy and applies the schema

### Try it out

```bash
# Seed with real sentence embeddings (no API key needed)
pip install sentence-transformers
python scripts/seed.py

# Open interactive API docs
open http://localhost:8000/docs

# Run integration tests
pip install pytest httpx
pytest tests/ -v
```

### Stop

```bash
docker compose down
```

---

## REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/collections` | Create a collection |
| `GET` | `/collections` | List collections |
| `DELETE` | `/collections/{name}` | Drop a collection |
| `POST` | `/collections/{name}/vectors` | Upsert vectors (batch) |
| `DELETE` | `/collections/{name}/vectors/{id}` | Delete a vector |
| `POST` | `/collections/{name}/search` | ANN search |
| `POST` | `/collections/{name}/snapshot` | Force HNSW snapshot |
| `GET` | `/health` | Liveness check |

### Example

```bash
# Create collection
curl -X POST http://localhost:8000/collections \
  -H "Content-Type: application/json" \
  -d '{"name":"docs","dimension":384,"metric":"cosine"}'

# Upsert vectors
curl -X POST http://localhost:8000/collections/docs/vectors \
  -H "Content-Type: application/json" \
  -d '{"vectors":[{"id":"v1","embedding":[0.1,0.2,...],"metadata":{"title":"hello"}}]}'

# Search
curl -X POST http://localhost:8000/collections/docs/search \
  -H "Content-Type: application/json" \
  -d '{"query":[0.1,0.2,...],"top_k":5}'
```

---

## Python SDK

```python
from sdk.scyllavec import ScyllaVecClient

with ScyllaVecClient("http://localhost:8000") as client:
    client.create_collection("docs", dimension=384, metric="cosine")

    client.upsert("docs", [
        {"id": "doc-1", "embedding": [...], "metadata": {"title": "Hello"}},
        {"id": "doc-2", "embedding": [...], "metadata": {"title": "World"}},
    ])

    results = client.search("docs", query=[...], top_k=5)
    for r in results:
        print(r["id"], r["distance"], r["metadata"])
```

---

## Project Structure

```
scyllavec/
├── docker-compose.yml        # Full local stack
├── pyproject.toml            # Python dependencies
├── schema/
│   └── 001_init.cql          # ScyllaDB schema
├── sidecar/                  # Rust HNSW index service
│   ├── Cargo.toml
│   ├── Dockerfile
│   ├── build.rs
│   ├── proto/
│   │   └── scyllavec.proto   # gRPC interface definition
│   └── src/
│       ├── main.rs
│       ├── index.rs          # In-memory HNSW + metric math
│       ├── db.rs             # ScyllaDB persistence + snapshots
│       └── grpc.rs           # gRPC handlers
├── api/                      # Python FastAPI REST service
│   ├── main.py               # Endpoints
│   ├── db.py                 # ScyllaDB connection
│   ├── sidecar_client.py     # gRPC client for sidecar
│   └── Dockerfile
├── sdk/
│   └── scyllavec/
│       ├── __init__.py
│       └── client.py         # Python SDK
├── tests/
│   └── test_api.py           # Integration tests
├── scripts/
│   ├── dev.sh                # One-command local startup
│   ├── init_schema.sh        # Apply CQL schema
│   └── seed.py               # Seed with sentence-transformers
└── benchmarks/               # (Phase 5) vs Pinecone, pgvector, Chroma
```

---

## Metrics supported

| Metric | Description |
|--------|-------------|
| `cosine` | Cosine similarity (recommended for NLP embeddings) |
| `euclidean` | L2 distance |
| `dot_product` | Dot product (for pre-normalized vectors) |

---

## Roadmap

- [x] Phase 1 — Storage layer (ScyllaDB schema, CQL read/write)
- [x] Phase 2 — HNSW sidecar (Rust, in-memory index, snapshot/WAL)
- [x] Phase 3 — REST API + Python SDK
- [ ] Phase 4 — Metadata filtering (server-side, pre/post-filter modes)
- [ ] Phase 5 — Benchmarks vs Pinecone, pgvector, Chroma, Weaviate
- [ ] Phase 6 — LangChain / LlamaIndex adapter
- [ ] Phase 7 — Multi-node index sharding
