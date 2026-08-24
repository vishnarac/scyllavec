"""
Integration tests for ScyllaVec.
Requires the full stack running: docker compose up

Run with:  pytest tests/ -v
"""
import pytest
import random
import math
import httpx

BASE_URL = "http://localhost:8000"
COLLECTION = "test_col"
DIM = 8


def rand_vec(dim: int = DIM) -> list[float]:
    v = [random.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x**2 for x in v))
    return [x / norm for x in v]  # unit vector


@pytest.fixture(scope="module")
def client():
    with httpx.Client(base_url=BASE_URL, timeout=30) as c:
        yield c


@pytest.fixture(autouse=True, scope="module")
def setup_collection(client):
    # Clean slate
    client.delete(f"/collections/{COLLECTION}")
    r = client.post("/collections", json={
        "name": COLLECTION,
        "dimension": DIM,
        "metric": "cosine",
    })
    assert r.status_code == 201
    yield
    client.delete(f"/collections/{COLLECTION}")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["api"] == "ok"


def test_list_collections(client):
    r = client.get("/collections")
    assert r.status_code == 200
    names = [c["name"] for c in r.json()]
    assert COLLECTION in names


def test_upsert_and_search(client):
    # Insert 20 random vectors
    vectors = [
        {"id": f"vec-{i}", "embedding": rand_vec(), "metadata": {"index": i}}
        for i in range(20)
    ]
    r = client.post(f"/collections/{COLLECTION}/vectors", json={"vectors": vectors})
    assert r.status_code == 201
    assert r.json()["inserted"] == 20

    # Search with a known query — just verify shape
    query = rand_vec()
    r = client.post(f"/collections/{COLLECTION}/search", json={"query": query, "top_k": 5})
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 5
    for res in results:
        assert "id" in res
        assert "distance" in res
        assert "metadata" in res


def test_search_returns_nearest(client):
    """Insert a vector, search with the same vector — it must be rank 1."""
    target = rand_vec()
    r = client.post(f"/collections/{COLLECTION}/vectors", json={
        "vectors": [{"id": "exact-match", "embedding": target, "metadata": {}}]
    })
    assert r.status_code == 201

    r = client.post(f"/collections/{COLLECTION}/search", json={"query": target, "top_k": 1})
    assert r.status_code == 200
    top = r.json()["results"][0]
    assert top["id"] == "exact-match"
    assert top["distance"] < 1e-4


def test_delete_vector(client):
    r = client.delete(f"/collections/{COLLECTION}/vectors/vec-0")
    assert r.status_code == 204

    # Should no longer appear in results
    r = client.post(f"/collections/{COLLECTION}/search", json={"query": rand_vec(), "top_k": 50})
    ids = [res["id"] for res in r.json()["results"]]
    assert "vec-0" not in ids


def test_snapshot(client):
    r = client.post(f"/collections/{COLLECTION}/snapshot")
    assert r.status_code == 200
    assert r.json()["success"] is True


# ── Phase 4: metadata filtering ───────────────────────────────────────────────

def test_filter_returns_matching_only(client):
    """Filter on a metadata field — results must all carry that field value."""
    # Insert two groups: tag=a and tag=b
    vectors = (
        [{"id": f"a-{i}", "embedding": rand_vec(), "metadata": {"tag": "a"}} for i in range(5)]
        + [{"id": f"b-{i}", "embedding": rand_vec(), "metadata": {"tag": "b"}} for i in range(5)]
    )
    r = client.post(f"/collections/{COLLECTION}/vectors", json={"vectors": vectors})
    assert r.status_code == 201

    r = client.post(f"/collections/{COLLECTION}/search", json={
        "query": rand_vec(),
        "top_k": 10,
        "filter": {"tag": "a"},
    })
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) > 0
    assert all(res["metadata"]["tag"] == "a" for res in results), \
        "All filtered results must have tag=a"


def test_filter_no_match_returns_empty(client):
    """A filter that matches nothing must return an empty result list."""
    r = client.post(f"/collections/{COLLECTION}/search", json={
        "query": rand_vec(),
        "top_k": 5,
        "filter": {"tag": "nonexistent"},
    })
    assert r.status_code == 200
    assert r.json()["results"] == []


def test_filter_multi_key(client):
    """Multi-key filter: all conditions must hold simultaneously."""
    vectors = [
        {"id": "mk-1", "embedding": rand_vec(), "metadata": {"tag": "a", "score": 1}},
        {"id": "mk-2", "embedding": rand_vec(), "metadata": {"tag": "a", "score": 2}},
        {"id": "mk-3", "embedding": rand_vec(), "metadata": {"tag": "b", "score": 1}},
    ]
    r = client.post(f"/collections/{COLLECTION}/vectors", json={"vectors": vectors})
    assert r.status_code == 201

    r = client.post(f"/collections/{COLLECTION}/search", json={
        "query": rand_vec(),
        "top_k": 10,
        "filter": {"tag": "a", "score": 1},
    })
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) > 0
    for res in results:
        assert res["metadata"]["tag"] == "a"
        assert res["metadata"]["score"] == 1
