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
      └─ gRPC ─┬────────► ║  collections           ║
  sidecar-0    │          ║  index_snapshots       ║
  sidecar-1    │          ║  index_wal             ║
  (HNSW shards)│          ╚════════════════════════╝
               │  (hash-partitioned; fan-out search)
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
| `POST` | `/collections/{name}/search` | ANN search (with optional metadata filter) |
| `POST` | `/collections/{name}/snapshot` | Force HNSW snapshot to ScyllaDB |
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
  -d '{"vectors":[{"id":"v1","embedding":[0.1,0.2,...],"metadata":{"title":"hello","category":"infra"}}]}'

# Search (no filter)
curl -X POST http://localhost:8000/collections/docs/search \
  -H "Content-Type: application/json" \
  -d '{"query":[0.1,0.2,...],"top_k":5}'

# Search with metadata filter
curl -X POST http://localhost:8000/collections/docs/search \
  -H "Content-Type: application/json" \
  -d '{"query":[0.1,0.2,...],"top_k":5,"filter":{"category":"infra"}}'
```

---

## Python SDK

```python
from scyllavec import ScyllaVecClient

with ScyllaVecClient("http://localhost:8000") as client:
    client.create_collection("docs", dimension=384, metric="cosine")

    client.upsert("docs", [
        {"id": "doc-1", "embedding": [...], "metadata": {"title": "Hello", "tag": "a"}},
        {"id": "doc-2", "embedding": [...], "metadata": {"title": "World", "tag": "b"}},
    ])

    # Plain search
    results = client.search("docs", query=[...], top_k=5)

    # Filtered search — only return vectors where tag == "a"
    results = client.search("docs", query=[...], top_k=5, filter={"tag": "a"})

    for r in results:
        print(r["id"], r["distance"], r["metadata"])
```

---

## Metadata Filtering

Pass a `filter` dict to any search call to restrict results to vectors whose metadata matches **all** supplied key/value pairs (exact equality, AND logic).

```python
# Single field
client.search("docs", query=[...], top_k=10, filter={"category": "science"})

# Multiple fields (both must match)
client.search("docs", query=[...], top_k=10, filter={"category": "science", "year": 2024})
```

Filtering is implemented as **post-filter**: the sidecar returns up to `top_k × 10` ANN candidates, the API applies the filter in Python, then returns the top `top_k` matching results in distance order.

---

## LangChain Integration

```bash
pip install langchain-core sentence-transformers
```

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from scyllavec.langchain import ScyllaVecVectorStore

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create a collection and index documents in one call
store = ScyllaVecVectorStore.from_texts(
    texts=["ScyllaDB is fast", "Vector search is powerful"],
    embedding=embeddings,
    collection="docs",
    dimension=384,
)

# Semantic search
docs = store.similarity_search("high performance database", k=3)

# Semantic search with metadata filter
docs = store.similarity_search("database", k=3, filter={"category": "infra"})

# With similarity scores
results = store.similarity_search_with_score("embeddings", k=5)
for doc, score in results:
    print(f"[{score:.4f}] {doc.page_content}")
```

---

## LlamaIndex Integration

```bash
pip install llama-index-core
```

```python
from llama_index.core import VectorStoreIndex, StorageContext
from scyllavec.llamaindex import ScyllaVecVectorStore

vector_store = ScyllaVecVectorStore(
    collection="docs",
    dimension=384,
    base_url="http://localhost:8000",
)

# Build an index from documents
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Query
query_engine = index.as_query_engine(similarity_top_k=5)
response = query_engine.query("What is ScyllaDB?")
```

---

## Project Structure

```
scyllavec/
├── docker-compose.yml        # Full local stack
├── pyproject.toml            # Python dependencies + optional extras
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
│   ├── main.py               # Endpoints + metadata filter logic
│   ├── db.py                 # ScyllaDB connection
│   ├── sidecar_client.py     # gRPC client for sidecar
│   └── Dockerfile
├── sdk/
│   └── scyllavec/
│       ├── __init__.py
│       ├── client.py         # Python SDK
│       ├── langchain.py      # LangChain VectorStore adapter
│       └── llamaindex.py     # LlamaIndex VectorStore adapter
├── tests/
│   ├── test_api.py           # Integration tests (requires running stack)
│   ├── test_langchain.py     # LangChain adapter unit tests (mocked)
│   ├── test_llamaindex.py    # LlamaIndex adapter unit tests (mocked)
│   └── test_sharding.py      # Phase 7 unit tests — routing, fan-out, health
├── scripts/
│   ├── dev.sh                # One-command local startup
│   ├── init_schema.sh        # Apply CQL schema
│   └── seed.py               # Seed with sentence-transformers
└── benchmarks/               # Phase 5 — vs pgvector, Chroma, Weaviate
```

---

## Metrics supported

| Metric | Description |
|--------|-------------|
| `cosine` | Cosine similarity (recommended for NLP embeddings) |
| `euclidean` | L2 distance |
| `dot_product` | Dot product (for pre-normalized vectors) |

---

## Multi-node sharding (Phase 7)

The stack runs two sidecar instances by default (`sidecar-0`, `sidecar-1`). The API service is configured via:

```
SIDECAR_URLS=sidecar-0:50051,sidecar-1:50051
```

Routing rules in [`api/sidecar_client.py`](api/sidecar_client.py):

| Operation | Strategy |
|-----------|----------|
| `upsert` / `delete` | `hash(vector_id) % n_shards` → single shard |
| `search` | parallel fan-out → merge by distance → global top-k |
| `evict` / `snapshot` | broadcast to all shards |
| `health` | all shards must report `"ok"` (otherwise `"degraded"`) |

Backward compatible: set `SIDECAR_URL=host:port` (single URL, old env var) to run a single-shard deployment.

---

## Roadmap

- [x] Phase 1 — Storage layer (ScyllaDB schema, CQL read/write)
- [x] Phase 2 — HNSW sidecar (Rust, in-memory index, snapshot/WAL)
- [x] Phase 3 — REST API + Python SDK
- [x] Phase 4 — Metadata filtering (post-filter, exact equality)
- [ ] Phase 5 — Benchmarks vs pgvector, Chroma, Weaviate → see [`benchmarks/`](benchmarks/README.md)
- [x] Phase 6 — LangChain / LlamaIndex adapter
- [x] Phase 7 — Multi-node index sharding
