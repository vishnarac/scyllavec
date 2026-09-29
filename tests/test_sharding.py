"""
Unit tests for Phase 7 — shard-aware routing in sidecar_client.py.
No running stack required — gRPC stubs are fully mocked.

Tests verify:
  - upsert/delete route to the correct shard (hash(id) % n)
  - search fans out to all shards and merges by distance
  - search deduplicates and truncates to top-k
  - evict/snapshot broadcast to all shards
  - health returns 'degraded' if any shard is unhealthy
  - single-shard mode is backward-compatible
"""
import sys
import os
import types

# ── Stub out proto modules before importing sidecar_client ───────────────────
# scyllavec_pb2 / scyllavec_pb2_grpc are generated inside Docker at build time
# and don't exist locally. We inject fake modules so the import succeeds.
#
# Use a MagicMock-backed module so that ANY attribute access (UpsertRequest,
# SearchRequest, DeleteRequest, …) returns a callable MagicMock automatically.
from unittest.mock import MagicMock

class _AutoMockModule(types.ModuleType):
    """A module whose every attribute access returns a fresh MagicMock."""
    def __getattr__(self, name: str):
        m = MagicMock(name=name)
        setattr(self, name, m)
        return m

_pb2 = _AutoMockModule("scyllavec_pb2")
_pb2_grpc = _AutoMockModule("scyllavec_pb2_grpc")
sys.modules.setdefault("scyllavec_pb2", _pb2)
sys.modules.setdefault("scyllavec_pb2_grpc", _pb2_grpc)

# Add api/ to path so `import api.sidecar_client` resolves
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hashlib
import importlib
from unittest.mock import MagicMock, patch  # MagicMock already imported above; patch added here
import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _shard_for(vector_id: str, n: int) -> int:
    digest = int(hashlib.md5(vector_id.encode()).hexdigest(), 16)
    return digest % n


def _make_search_result(vector_id: str, distance: float):
    r = MagicMock()
    r.vector_id = vector_id
    r.distance = distance
    return r


def _fresh_client(sidecar_urls: str):
    """Import a fresh copy of sidecar_client with env patched."""
    with patch.dict(os.environ, {"SIDECAR_URLS": sidecar_urls}, clear=False):
        # Force re-import so _pool is re-initialised
        import api.sidecar_client as mod
        importlib.reload(mod)
        return mod


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def two_shard_client():
    """Return a freshly loaded sidecar_client module with 2 mocked stubs."""
    with patch.dict(os.environ, {"SIDECAR_URLS": "s0:50051,s1:50051"}, clear=False):
        import api.sidecar_client as mod
        importlib.reload(mod)
        stub0, stub1 = MagicMock(), MagicMock()
        mod._pool._stubs = [stub0, stub1]
        mod._pool._urls = ["s0:50051", "s1:50051"]
        yield mod, stub0, stub1


@pytest.fixture()
def one_shard_client():
    """Return a freshly loaded sidecar_client module with 1 mocked stub."""
    with patch.dict(os.environ, {"SIDECAR_URLS": "s0:50051"}, clear=False):
        import api.sidecar_client as mod
        importlib.reload(mod)
        stub0 = MagicMock()
        mod._pool._stubs = [stub0]
        mod._pool._urls = ["s0:50051"]
        yield mod, stub0


# ── Routing tests ─────────────────────────────────────────────────────────────

def test_upsert_routes_to_correct_shard(two_shard_client):
    mod, stub0, stub1 = two_shard_client

    # Pick an ID that hashes to shard 0
    test_id = next(i for i in (f"v{n}" for n in range(100)) if _shard_for(i, 2) == 0)
    stub0.Upsert.return_value = MagicMock(success=True)

    mod.upsert("col", test_id, [0.1, 0.2])

    stub0.Upsert.assert_called_once()
    stub1.Upsert.assert_not_called()


def test_upsert_routes_other_ids_to_shard1(two_shard_client):
    mod, stub0, stub1 = two_shard_client

    test_id = next(i for i in (f"v{n}" for n in range(100)) if _shard_for(i, 2) == 1)
    stub1.Upsert.return_value = MagicMock(success=True)

    mod.upsert("col", test_id, [0.3, 0.4])

    stub1.Upsert.assert_called_once()
    stub0.Upsert.assert_not_called()


def test_delete_routes_to_correct_shard(two_shard_client):
    mod, stub0, stub1 = two_shard_client

    test_id = next(i for i in (f"v{n}" for n in range(100)) if _shard_for(i, 2) == 0)
    stub0.Delete.return_value = MagicMock(success=True)

    mod.delete("col", test_id)

    stub0.Delete.assert_called_once()
    stub1.Delete.assert_not_called()


# ── Fan-out search tests ──────────────────────────────────────────────────────

def test_search_fanout_merges_by_distance(two_shard_client):
    mod, stub0, stub1 = two_shard_client

    # Shard 0 returns distances 0.1, 0.5
    resp0 = MagicMock()
    resp0.results = [_make_search_result("a", 0.1), _make_search_result("c", 0.5)]
    stub0.Search.return_value = resp0

    # Shard 1 returns distances 0.2, 0.4
    resp1 = MagicMock()
    resp1.results = [_make_search_result("b", 0.2), _make_search_result("d", 0.4)]
    stub1.Search.return_value = resp1

    results = mod.search("col", [0.1, 0.2], top_k=3)

    assert len(results) == 3
    assert [r["vector_id"] for r in results] == ["a", "b", "d"]   # sorted by distance
    assert [r["distance"] for r in results] == [0.1, 0.2, 0.4]


def test_search_deduplicates(two_shard_client):
    mod, stub0, stub1 = two_shard_client

    # Both shards return the same ID (shouldn't happen in practice but must be safe)
    for stub in (stub0, stub1):
        resp = MagicMock()
        resp.results = [_make_search_result("dup", 0.1)]
        stub.Search.return_value = resp

    results = mod.search("col", [0.1], top_k=5)
    ids = [r["vector_id"] for r in results]
    assert ids.count("dup") == 1


def test_search_truncates_to_top_k(two_shard_client):
    mod, stub0, stub1 = two_shard_client

    resp0 = MagicMock()
    resp0.results = [_make_search_result(f"s0-{i}", float(i)) for i in range(5)]
    stub0.Search.return_value = resp0

    resp1 = MagicMock()
    resp1.results = [_make_search_result(f"s1-{i}", float(i) + 0.5) for i in range(5)]
    stub1.Search.return_value = resp1

    results = mod.search("col", [0.1], top_k=3)
    assert len(results) == 3


def test_search_partial_shard_failure(two_shard_client):
    """If one shard errors, results from the other shard are still returned."""
    mod, stub0, stub1 = two_shard_client

    stub0.Search.side_effect = Exception("shard 0 down")
    resp1 = MagicMock()
    resp1.results = [_make_search_result("b", 0.2)]
    stub1.Search.return_value = resp1

    results = mod.search("col", [0.1], top_k=5)
    assert len(results) == 1
    assert results[0]["vector_id"] == "b"


# ── Broadcast tests ───────────────────────────────────────────────────────────

def test_evict_broadcasts_to_all_shards(two_shard_client):
    mod, stub0, stub1 = two_shard_client
    stub0.DropCollection.return_value = MagicMock(success=True)
    stub1.DropCollection.return_value = MagicMock(success=True)

    result = mod.evict("col")

    assert result is True
    stub0.DropCollection.assert_called_once()
    stub1.DropCollection.assert_called_once()


def test_snapshot_aggregates_node_counts(two_shard_client):
    mod, stub0, stub1 = two_shard_client
    stub0.Snapshot.return_value = MagicMock(success=True, version=100, nodes=500)
    stub1.Snapshot.return_value = MagicMock(success=True, version=110, nodes=600)

    result = mod.snapshot("col")

    assert result["nodes"] == 1100        # sum
    assert result["version"] == 110       # max
    assert result["success"] is True


# ── Health tests ──────────────────────────────────────────────────────────────

def test_health_ok_when_all_shards_healthy(two_shard_client):
    mod, stub0, stub1 = two_shard_client
    stub0.Health.return_value = MagicMock(status="ok")
    stub1.Health.return_value = MagicMock(status="ok")
    assert mod.health() == "ok"


def test_health_degraded_when_one_shard_unhealthy(two_shard_client):
    mod, stub0, stub1 = two_shard_client
    stub0.Health.return_value = MagicMock(status="ok")
    stub1.Health.return_value = MagicMock(status="degraded")
    assert mod.health() == "degraded"


def test_health_degraded_when_shard_unreachable(two_shard_client):
    mod, stub0, stub1 = two_shard_client
    stub0.Health.return_value = MagicMock(status="ok")
    stub1.Health.side_effect = Exception("connection refused")
    assert mod.health() == "degraded"


# ── Backward-compat: single-shard ─────────────────────────────────────────────

def test_single_shard_search_no_fanout(one_shard_client):
    mod, stub0 = one_shard_client
    resp = MagicMock()
    resp.results = [_make_search_result("x", 0.1)]
    stub0.Search.return_value = resp

    results = mod.search("col", [0.1], top_k=1)

    assert len(results) == 1
    stub0.Search.assert_called_once()


def test_single_shard_upsert(one_shard_client):
    mod, stub0 = one_shard_client
    stub0.Upsert.return_value = MagicMock(success=True)
    mod.upsert("col", "any-id", [0.1])
    stub0.Upsert.assert_called_once()
