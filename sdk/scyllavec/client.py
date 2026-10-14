"""
ScyllaVec Python SDK — thin wrapper over the REST API.
"""
from __future__ import annotations
import httpx
from typing import Any


class ScyllaVecClient:
    """
    Client for the ScyllaVec REST API.

    Usage::

        client = ScyllaVecClient("http://localhost:8000")
        client.create_collection("docs", dimension=384)
        client.upsert("docs", [{"embedding": [0.1, 0.2, ...], "metadata": {"title": "hi"}}])
        results = client.search("docs", query=[0.1, 0.2, ...], top_k=5)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        headers = {"X-API-Key": api_key} if api_key else {}
        self._client = httpx.Client(base_url=base_url, timeout=timeout, headers=headers)

    # ── Collections ───────────────────────────────────────────────────────────

    def create_collection(
        self,
        name: str,
        dimension: int,
        metric: str = "cosine",
        description: str = "",
    ) -> dict:
        r = self._client.post("/collections", json={
            "name": name,
            "dimension": dimension,
            "metric": metric,
            "description": description,
        })
        r.raise_for_status()
        return r.json()

    def list_collections(self) -> list[dict]:
        r = self._client.get("/collections")
        r.raise_for_status()
        return r.json()

    def delete_collection(self, name: str) -> None:
        r = self._client.delete(f"/collections/{name}")
        r.raise_for_status()

    # ── Vectors ───────────────────────────────────────────────────────────────

    def upsert(
        self,
        collection: str,
        vectors: list[dict[str, Any]],
    ) -> dict:
        """
        vectors: list of {"id": optional str, "embedding": list[float], "metadata": dict}
        """
        r = self._client.post(f"/collections/{collection}/vectors", json={"vectors": vectors})
        r.raise_for_status()
        return r.json()

    def delete_vector(self, collection: str, vector_id: str) -> None:
        r = self._client.delete(f"/collections/{collection}/vectors/{vector_id}")
        r.raise_for_status()

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        collection: str,
        query: list[float],
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[dict]:
        r = self._client.post(f"/collections/{collection}/search", json={
            "query": query,
            "top_k": top_k,
            "filter": filter or {},
        })
        r.raise_for_status()
        return r.json()["results"]

    # ── Maintenance ───────────────────────────────────────────────────────────

    def snapshot(self, collection: str) -> dict:
        r = self._client.post(f"/collections/{collection}/snapshot")
        r.raise_for_status()
        return r.json()

    def health(self) -> dict:
        r = self._client.get("/health")
        r.raise_for_status()
        return r.json()

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
