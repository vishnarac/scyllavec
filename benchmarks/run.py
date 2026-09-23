#!/usr/bin/env python3
"""
ScyllaVec benchmark harness — Phase 5.

Measures insert throughput, query latency (p50/p99), and Recall@10
for ScyllaVec, pgvector, Chroma, and Weaviate using the same synthetic
dataset and query set.

Prerequisites
─────────────
1. ScyllaVec stack running:
     bash scripts/dev.sh

2. Competitor stack running:
     docker compose -f benchmarks/docker-compose.yml up -d

3. Python deps installed:
     pip install -r benchmarks/requirements.txt

Run
───
     python benchmarks/run.py [--n 10000] [--dim 384] [--batch 100] [--queries 200]
"""
from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
import time
import uuid
from typing import Callable

import numpy as np

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="ScyllaVec benchmark")
parser.add_argument("--n",       type=int, default=10_000, help="Dataset size (default: 10000)")
parser.add_argument("--dim",     type=int, default=384,    help="Vector dimension (default: 384)")
parser.add_argument("--batch",   type=int, default=100,    help="Upsert batch size (default: 100)")
parser.add_argument("--queries", type=int, default=200,    help="Number of search queries (default: 200)")
parser.add_argument("--top-k",   type=int, default=10,     help="top-k for recall (default: 10)")
parser.add_argument(
    "--systems",
    nargs="+",
    default=["scyllavec", "pgvector", "chroma", "weaviate"],
    help="Systems to benchmark",
)
args = parser.parse_args()

N       = args.n
DIM     = args.dim
BATCH   = args.batch
NQUERY  = args.queries
TOP_K   = args.top_k
SYSTEMS = args.systems

COLLECTION = "bench"

# ── Dataset generation ────────────────────────────────────────────────────────

print(f"Generating {N} random unit vectors (dim={DIM})...")
rng = np.random.default_rng(42)
raw = rng.standard_normal((N, DIM)).astype(np.float32)
norms = np.linalg.norm(raw, axis=1, keepdims=True)
VECTORS: np.ndarray = raw / norms          # unit vectors, shape (N, DIM)
IDS: list[str] = [f"v{i}" for i in range(N)]

query_idx = rng.choice(N, size=NQUERY, replace=False)
QUERIES: np.ndarray = VECTORS[query_idx]   # shape (NQUERY, DIM)

# ── Brute-force ground truth ──────────────────────────────────────────────────

def brute_force_top_k(query: np.ndarray, k: int) -> list[int]:
    """Return indices of k closest vectors by cosine distance (=1 - dot for unit vecs)."""
    scores = VECTORS @ query          # (N,) dot products
    return list(np.argpartition(-scores, k)[:k])


print("Computing brute-force ground truth...")
GROUND_TRUTH: list[set[int]] = [
    set(brute_force_top_k(QUERIES[i], TOP_K)) for i in range(NQUERY)
]

# ── Timing helpers ────────────────────────────────────────────────────────────

def timed_insert(fn: Callable, vectors, ids) -> float:
    """Insert all vectors in batches; return total elapsed seconds."""
    t0 = time.perf_counter()
    for start in range(0, len(vectors), BATCH):
        batch_vecs = vectors[start:start + BATCH].tolist()
        batch_ids  = ids[start:start + BATCH]
        fn(batch_vecs, batch_ids)
    return time.perf_counter() - t0


def timed_queries(fn: Callable) -> tuple[float, float, list[float]]:
    """Run NQUERY searches; return (p50_ms, p99_ms, raw_latencies_ms)."""
    latencies = []
    for i in range(NQUERY):
        t0 = time.perf_counter()
        fn(QUERIES[i].tolist(), TOP_K)
        latencies.append((time.perf_counter() - t0) * 1000)
    latencies.sort()
    p50 = statistics.median(latencies)
    p99 = latencies[int(0.99 * len(latencies))]
    return p50, p99, latencies


def recall(result_ids: list[str], truth_set: set[int]) -> float:
    """Fraction of ground-truth top-k IDs found in result_ids."""
    found = sum(1 for rid in result_ids if int(rid[1:]) in truth_set)
    return found / len(truth_set)


def measure_recall(search_fn: Callable) -> float:
    """Average Recall@K over all queries."""
    recalls = []
    for i in range(NQUERY):
        results = search_fn(QUERIES[i].tolist(), TOP_K)
        recalls.append(recall(results, GROUND_TRUTH[i]))
    return statistics.mean(recalls)


# ── Results accumulator ───────────────────────────────────────────────────────

results: dict[str, dict] = {}

def record(system: str, insert_vps: float, p50: float, p99: float, rec: float):
    results[system] = {
        "insert_vps": insert_vps,
        "p50_ms":     p50,
        "p99_ms":     p99,
        "recall":     rec,
    }

# ══════════════════════════════════════════════════════════════════════════════
# ScyllaVec
# ══════════════════════════════════════════════════════════════════════════════

if "scyllavec" in SYSTEMS:
    print("\n── ScyllaVec ──────────────────────────────────────")
    try:
        import httpx
    except ImportError:
        print("  SKIP — pip install httpx")
    else:
        BASE = "http://localhost:8000"
        client = httpx.Client(base_url=BASE, timeout=120)
        try:
            # Setup
            client.delete(f"/collections/{COLLECTION}")
            r = client.post("/collections", json={"name": COLLECTION, "dimension": DIM, "metric": "cosine"})
            assert r.status_code == 201, r.text

            def sv_insert(batch_vecs, batch_ids):
                payload = {"vectors": [
                    {"id": bid, "embedding": vec} for bid, vec in zip(batch_ids, batch_vecs)
                ]}
                r = client.post(f"/collections/{COLLECTION}/vectors", json=payload)
                assert r.status_code == 201, r.text

            def sv_search(query, k) -> list[str]:
                r = client.post(f"/collections/{COLLECTION}/search",
                                json={"query": query, "top_k": k})
                assert r.status_code == 200, r.text
                return [res["id"] for res in r.json()["results"]]

            elapsed = timed_insert(sv_insert, VECTORS, IDS)
            vps = N / elapsed
            print(f"  Insert: {vps:,.0f} vectors/sec")

            p50, p99, _ = timed_queries(sv_search)
            print(f"  Query:  p50={p50:.1f}ms  p99={p99:.1f}ms")

            rec = measure_recall(sv_search)
            print(f"  Recall@{TOP_K}: {rec:.3f}")

            record("ScyllaVec", vps, p50, p99, rec)
        except Exception as e:
            print(f"  SKIP — ScyllaVec stack not reachable: {e}")
        finally:
            client.close()

# ══════════════════════════════════════════════════════════════════════════════
# pgvector
# ══════════════════════════════════════════════════════════════════════════════

if "pgvector" in SYSTEMS:
    print("\n── pgvector ───────────────────────────────────────")
    try:
        import psycopg2
        from pgvector.psycopg2 import register_vector
    except ImportError:
        print("  SKIP — pip install psycopg2-binary pgvector")
    else:
        conn = psycopg2.connect(
            host="localhost", port=5432,
            dbname="bench", user="bench", password="bench"
        )
        conn.autocommit = True
        cur = conn.cursor()
        register_vector(conn)

        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f"DROP TABLE IF EXISTS {COLLECTION}")
        cur.execute(
            f"CREATE TABLE {COLLECTION} "
            f"(id text PRIMARY KEY, embedding vector({DIM}))"
        )
        cur.execute(
            f"CREATE INDEX ON {COLLECTION} "
            f"USING hnsw (embedding vector_cosine_ops)"
        )

        def pg_insert(batch_vecs, batch_ids):
            args = [(bid, vec) for bid, vec in zip(batch_ids, batch_vecs)]
            cur.executemany(
                f"INSERT INTO {COLLECTION} (id, embedding) VALUES (%s, %s) "
                f"ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding",
                args,
            )

        def pg_search(query, k) -> list[str]:
            cur.execute(
                f"SELECT id FROM {COLLECTION} "
                f"ORDER BY embedding <=> %s LIMIT %s",
                (query, k),
            )
            return [row[0] for row in cur.fetchall()]

        elapsed = timed_insert(pg_insert, VECTORS, IDS)
        vps = N / elapsed
        print(f"  Insert: {vps:,.0f} vectors/sec")

        p50, p99, _ = timed_queries(pg_search)
        print(f"  Query:  p50={p50:.1f}ms  p99={p99:.1f}ms")

        rec = measure_recall(pg_search)
        print(f"  Recall@{TOP_K}: {rec:.3f}")

        record("pgvector", vps, p50, p99, rec)
        cur.close()
        conn.close()

# ══════════════════════════════════════════════════════════════════════════════
# Chroma
# ══════════════════════════════════════════════════════════════════════════════

if "chroma" in SYSTEMS:
    print("\n── Chroma ─────────────────────────────────────────")
    try:
        import chromadb
    except ImportError:
        print("  SKIP — pip install chromadb")
    else:
        chroma = chromadb.HttpClient(host="localhost", port=8001)
        try:
            chroma.delete_collection(COLLECTION)
        except Exception:
            pass
        col = chroma.create_collection(
            COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

        def ch_insert(batch_vecs, batch_ids):
            col.add(ids=batch_ids, embeddings=batch_vecs)

        def ch_search(query, k) -> list[str]:
            res = col.query(query_embeddings=[query], n_results=k)
            return res["ids"][0]

        elapsed = timed_insert(ch_insert, VECTORS, IDS)
        vps = N / elapsed
        print(f"  Insert: {vps:,.0f} vectors/sec")

        p50, p99, _ = timed_queries(ch_search)
        print(f"  Query:  p50={p50:.1f}ms  p99={p99:.1f}ms")

        rec = measure_recall(ch_search)
        print(f"  Recall@{TOP_K}: {rec:.3f}")

        record("Chroma", vps, p50, p99, rec)

# ══════════════════════════════════════════════════════════════════════════════
# Weaviate
# ══════════════════════════════════════════════════════════════════════════════

if "weaviate" in SYSTEMS:
    print("\n── Weaviate ───────────────────────────────────────")
    try:
        import weaviate
        import weaviate.classes as wvc
    except ImportError:
        print("  SKIP — pip install weaviate-client")
    else:
        wv = weaviate.connect_to_local(port=8080)
        col_name = COLLECTION.capitalize()
        try:
            wv.collections.delete(col_name)
        except Exception:
            pass
        wv_col = wv.collections.create(
            col_name,
            vectorizer_config=wvc.config.Configure.Vectorizer.none(),
            vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                distance_metric=wvc.config.VectorDistances.COSINE,
            ),
        )

        def wv_insert(batch_vecs, batch_ids):
            with wv_col.batch.dynamic() as batch:
                for bid, vec in zip(batch_ids, batch_vecs):
                    batch.add_object(properties={"item_id": bid}, vector=vec)

        def wv_search(query, k) -> list[str]:
            res = wv_col.query.near_vector(near_vector=query, limit=k)
            return [obj.properties.get("item_id", "") for obj in res.objects]

        elapsed = timed_insert(wv_insert, VECTORS, IDS)
        vps = N / elapsed
        print(f"  Insert: {vps:,.0f} vectors/sec")

        p50, p99, _ = timed_queries(wv_search)
        print(f"  Query:  p50={p50:.1f}ms  p99={p99:.1f}ms")

        rec = measure_recall(wv_search)
        print(f"  Recall@{TOP_K}: {rec:.3f}")

        record("Weaviate", vps, p50, p99, rec)
        wv.close()

# ══════════════════════════════════════════════════════════════════════════════
# Results table
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'─'*65}")
print(f"{'System':<12} {'Insert (v/s)':>12} {'p50 (ms)':>10} {'p99 (ms)':>10} {f'Recall@{TOP_K}':>10}")
print(f"{'─'*65}")
for system, r in results.items():
    print(
        f"{system:<12} {r['insert_vps']:>12,.0f} "
        f"{r['p50_ms']:>10.1f} {r['p99_ms']:>10.1f} {r['recall']:>10.3f}"
    )
print(f"{'─'*65}")
print(f"\nDataset: {N:,} vectors, dim={DIM}, batch={BATCH}, queries={NQUERY}")
