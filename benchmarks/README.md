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
| **ScyllaVec** |          640 |      5.7 |     10.3 | **0.998** |
| pgvector      |          233 |      1.6 |      3.1 |     0.834 |
| Chroma        |        1,328 |      4.9 |      9.6 |     0.901 |
| Weaviate      |        1,849 |      4.6 |      7.3 |     0.989 |

ScyllaVec results use the real HNSW graph (`instant-distance`) in the Rust sidecar, with a single shard.

**Key observations:**

- ScyllaVec reaches **Recall@10 = 0.998** with HNSW, the highest of the four systems here, at the same default config used for the others.
- Insert throughput is bottlenecked by the HTTP+gRPC round-trip overhead per batch through the multi-layer ScyllaVec stack (REST API → gRPC sidecar → ScyllaDB). Chroma/Weaviate batch via REST directly into their own storage engine.
- Query latency (p50 5.7 ms) is slower than pgvector, Chroma and Weaviate. It includes the extra network hop through the Rust sidecar. `instant-distance` is batch-only, so the graph is rebuilt lazily on the first search after writes; the first query after an insert pays that rebuild cost.
- This is a small, synthetic, single-machine run (5k vectors). Treat it as a sanity check, not a scale claim.

## Notes

- All systems run locally in Docker with default configurations (no tuning)
- ScyllaVec uses cosine metric; competitors configured to match
- Insert batches are `--batch` vectors per request/transaction
- Recall is computed against exact brute-force cosine similarity
- Query latency includes HTTP/network overhead to localhost
