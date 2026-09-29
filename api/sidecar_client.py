"""
Shard-aware gRPC client for the ScyllaVec HNSW sidecar.

Single-shard (backward-compatible):
    SIDECAR_URL=sidecar:50051          (old env var, still works)

Multi-shard:
    SIDECAR_URLS=sidecar-0:50051,sidecar-1:50051

Routing:
  - upsert / delete  → deterministic shard by hash(vector_uuid) % n_shards
  - search           → fan-out to all shards, merge by distance, return top-k
  - evict / snapshot → broadcast to all shards
  - health           → all shards must report "ok"
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import os
import sys
import pathlib
from typing import Any

import grpc

# Add generated proto stubs to path
sys.path.insert(0, str(pathlib.Path(__file__).parent / "proto_gen"))

import scyllavec_pb2 as pb
import scyllavec_pb2_grpc as pb_grpc

# ── Shard registry ────────────────────────────────────────────────────────────

def _parse_urls() -> list[str]:
    """Return list of bare host:port strings for all shards."""
    # Prefer SIDECAR_URLS (multi-shard); fall back to SIDECAR_URL (legacy)
    raw = os.getenv("SIDECAR_URLS") or os.getenv("SIDECAR_URL", "localhost:50051")
    urls = []
    for u in raw.split(","):
        u = u.strip()
        if "://" in u:
            u = u.split("://", 1)[1]
        if u:
            urls.append(u)
    return urls


class _ShardPool:
    """Lazy pool of gRPC stubs, one per shard URL."""

    def __init__(self) -> None:
        self._urls: list[str] = []
        self._channels: list[grpc.Channel] = []
        self._stubs: list[pb_grpc.VectorIndexStub] = []

    def _ensure_init(self) -> None:
        if self._stubs:
            return
        self._urls = _parse_urls()
        for url in self._urls:
            ch = grpc.insecure_channel(url)
            self._channels.append(ch)
            self._stubs.append(pb_grpc.VectorIndexStub(ch))

    @property
    def stubs(self) -> list[pb_grpc.VectorIndexStub]:
        self._ensure_init()
        return self._stubs

    @property
    def n(self) -> int:
        self._ensure_init()
        return len(self._stubs)

    def stub_for(self, vector_id: str) -> pb_grpc.VectorIndexStub:
        """Route a vector to its owning shard by consistent hash."""
        self._ensure_init()
        digest = int(hashlib.md5(vector_id.encode()).hexdigest(), 16)
        return self._stubs[digest % self.n]

    def shard_index_for(self, vector_id: str) -> int:
        self._ensure_init()
        digest = int(hashlib.md5(vector_id.encode()).hexdigest(), 16)
        return digest % self.n


_pool = _ShardPool()


# ── Public API ────────────────────────────────────────────────────────────────

def upsert(collection: str, vector_id: str, embedding: list[float]) -> bool:
    stub = _pool.stub_for(vector_id)
    resp = stub.Upsert(pb.UpsertRequest(
        collection=collection,
        vector_id=vector_id,
        embedding=embedding,
    ))
    return resp.success


def search(collection: str, query: list[float], top_k: int = 10) -> list[dict]:
    """Fan-out search to all shards, merge results, return global top-k."""
    stubs = _pool.stubs

    def _search_shard(stub: pb_grpc.VectorIndexStub) -> list[dict]:
        resp = stub.Search(pb.SearchRequest(
            collection=collection,
            query=query,
            top_k=top_k,
        ))
        return [{"vector_id": r.vector_id, "distance": r.distance} for r in resp.results]

    if len(stubs) == 1:
        return _search_shard(stubs[0])

    # Parallel fan-out
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(stubs)) as ex:
        futures = [ex.submit(_search_shard, s) for s in stubs]
        all_results: list[dict] = []
        for f in concurrent.futures.as_completed(futures):
            try:
                all_results.extend(f.result())
            except Exception:
                pass  # partial failure — continue with results from other shards

    # Merge: sort by distance ascending, deduplicate by vector_id, take top-k
    seen: set[str] = set()
    merged: list[dict] = []
    for r in sorted(all_results, key=lambda x: x["distance"]):
        if r["vector_id"] not in seen:
            seen.add(r["vector_id"])
            merged.append(r)
        if len(merged) == top_k:
            break
    return merged


def delete(collection: str, vector_id: str) -> bool:
    stub = _pool.stub_for(vector_id)
    resp = stub.Delete(pb.DeleteRequest(collection=collection, vector_id=vector_id))
    return resp.success


def snapshot(collection: str) -> dict:
    """Trigger snapshot on all shards; return aggregated node count."""
    total_nodes = 0
    last_version = 0
    for stub in _pool.stubs:
        resp = stub.Snapshot(pb.SnapshotRequest(collection=collection))
        total_nodes += resp.nodes
        last_version = max(last_version, resp.version)
    return {"success": True, "version": last_version, "nodes": total_nodes}


def evict(collection: str) -> bool:
    """Broadcast DropCollection to all shards."""
    ok = True
    for stub in _pool.stubs:
        resp = stub.DropCollection(pb.DropCollectionRequest(collection=collection))
        ok = ok and resp.success
    return ok


def health() -> str:
    """All shards must be healthy; return 'ok' or 'degraded'."""
    for stub in _pool.stubs:
        try:
            resp = stub.Health(pb.HealthRequest())
            if resp.status != "ok":
                return "degraded"
        except Exception:
            return "degraded"
    return "ok"
