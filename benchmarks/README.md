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

> Run on: Apple M-series, 16 GB RAM, Docker Desktop (colima), 10k vectors, dim=384, batch=100, 200 queries

| System    | Insert (v/s) | p50 (ms) | p99 (ms) | Recall@10 |
|-----------|-------------:|---------:|---------:|----------:|
| ScyllaVec |            — |        — |        — |         — |
| pgvector  |            — |        — |        — |         — |
| Chroma    |            — |        — |        — |         — |
| Weaviate  |            — |        — |        — |         — |

*Results pending — run `python benchmarks/run.py` to populate this table.*

## Notes

- All systems run locally in Docker with default configurations (no tuning)
- ScyllaVec uses cosine metric; competitors configured to match
- Insert batches are `--batch` vectors per request/transaction
- Recall is computed against exact brute-force cosine similarity
- Query latency includes HTTP/network overhead to localhost
