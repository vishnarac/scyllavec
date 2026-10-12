# ScyllaVec Benchmarks

Compares ScyllaVec against **pgvector**, **Chroma**, and **Weaviate** on the same synthetic dataset using identical query workloads.

## What is measured

| Metric | Description |
|--------|-------------|
| **Insert throughput** | Vectors inserted per second at `--batch` size |
| **Query latency p50** | Median search latency over `--queries` queries |
| **Query latency p99** | 99th-percentile search latency |
| **Recall@K** | Fraction of true top-K neighbours returned by ANN vs brute-force |

## How to run

**1. Start the ScyllaVec stack** (if not already running):

```bash
bash scripts/dev.sh
```

**2. Start the competitor stack:**

```bash
docker compose -f benchmarks/docker-compose.yml up -d
```

Wait ~15 seconds for all services to be healthy.

**3. Install benchmark deps:**

```bash
python3 -m venv .bench-venv
source .bench-venv/bin/activate
pip install -r benchmarks/requirements.txt
```

**4. Run:**

```bash
# Default: 10k vectors, dim=384, batch=100, 200 queries
python benchmarks/run.py

# Larger dataset
python benchmarks/run.py --n 100000 --batch 500

# Single system only
python benchmarks/run.py --systems scyllavec pgvector
```

**5. Stop competitors when done:**

```bash
docker compose -f benchmarks/docker-compose.yml down -v
```

## Results

> Hardware: Apple M-series, 16 GB RAM, colima (Docker)
> Dataset: 5,000 vectors · dim=384 · batch=100 · 200 queries · cosine metric
> All systems run in Docker with default config (no tuning)

| System        | Insert (v/s) | p50 (ms) | p99 (ms) | Recall@10 |
|---------------|-------------:|---------:|---------:|----------:|
| **ScyllaVec** |          476 |      6.4 |     10.9 | **1.000** |
| pgvector      |          239 |      2.1 |      4.6 |     0.836 |
| Chroma        |        1,389 |      4.2 |     13.3 |     0.897 |
| Weaviate      |        1,878 |      4.8 |      8.0 |     0.989 |

**Key observations:**

- ScyllaVec achieves **Recall@10 = 1.000** (exact brute-force search) at this dataset size. pgvector uses HNSW (approximate) and Chroma/Weaviate trade recall for throughput.
- Insert throughput is bottlenecked by the HTTP+gRPC round-trip overhead per batch through the multi-layer ScyllaVec stack (REST API → gRPC sidecar → ScyllaDB). Chroma/Weaviate batch via REST directly into their own storage engine.
- Query latency reflects the additional network hop through the Rust sidecar. At larger datasets where HNSW approximation matters, ScyllaVec's brute-force will scale linearly — the HNSW integration path in `instant-distance` is the next performance milestone.

## Notes

- All systems run locally in Docker with default configurations (no tuning)
- ScyllaVec uses cosine metric; competitors configured to match
- Insert batches are `--batch` vectors per request/transaction
- Recall is computed against exact brute-force cosine similarity
- Query latency includes HTTP/network overhead to localhost
