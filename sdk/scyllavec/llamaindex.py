"""
LlamaIndex VectorStore adapter for ScyllaVec.

Usage::

    from llama_index.core import VectorStoreIndex, StorageContext
    from llama_index.core.vector_stores.types import VectorStoreQuery
    from scyllavec.llamaindex import ScyllaVecVectorStore

    vector_store = ScyllaVecVectorStore(
        collection="docs",
        base_url="http://localhost:8000",
        dimension=384,
    )

    # Use with VectorStoreIndex
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    # Or query directly
    query = VectorStoreQuery(query_embedding=[...], similarity_top_k=5)
    result = vector_store.query(query)
"""
from __future__ import annotations

from typing import Any, Sequence, cast

try:
    from llama_index.core.schema import BaseNode, TextNode
    from llama_index.core.vector_stores.types import (
        MetadataFilters,
        MetadataFilter,
        FilterOperator,
        VectorStoreQuery,
        VectorStoreQueryResult,
        VectorStore as LlamaVectorStore,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "llama-index-core is required for the LlamaIndex adapter. "
        "Install it with: pip install llama-index-core"
    ) from e

from .client import ScyllaVecClient


def _filters_to_dict(filters: MetadataFilters | None) -> dict[str, Any]:
    """Convert LlamaIndex MetadataFilters to ScyllaVec's flat equality filter dict.

    Only EQ (==) conditions are supported in post-filter mode.
    Other operators are silently skipped.
    """
    if not filters or not filters.filters:
        return {}
    result: dict[str, Any] = {}
    for f in filters.filters:
        if isinstance(f, MetadataFilter):
            if f.operator in (FilterOperator.EQ, "==", None):
                result[f.key] = f.value
    return result


class ScyllaVecVectorStore(LlamaVectorStore):
    """LlamaIndex-compatible VectorStore backed by ScyllaVec."""

    stores_text: bool = True
    flat_metadata: bool = True

    def __init__(
        self,
        collection: str,
        dimension: int,
        base_url: str = "http://localhost:8000",
        metric: str = "cosine",
    ) -> None:
        self._collection = collection
        self._base_url = base_url
        self._dimension = dimension
        self._metric = metric
        self._client = ScyllaVecClient(base_url)
        # Ensure the collection exists
        try:
            self._client.create_collection(collection, dimension=dimension, metric=metric)
        except Exception:
            pass  # already exists

    # ── LlamaIndex interface ──────────────────────────────────────────────────

    def add(self, nodes: Sequence[BaseNode], **kwargs: Any) -> list[str]:
        items = []
        for node in nodes:
            embedding = node.embedding  # None if not set; get_embedding() raises
            if not embedding:
                continue
            meta = dict(node.metadata)
            if hasattr(node, "text"):
                meta["text"] = cast(TextNode, node).text
            items.append({
                "id": node.node_id,
                "embedding": embedding,
                "metadata": meta,
            })
        if not items:
            return []
        result = self._client.upsert(self._collection, items)
        return result["ids"]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        self._client.delete_vector(self._collection, ref_doc_id)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        if not query.query_embedding:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        top_k = query.similarity_top_k
        f = _filters_to_dict(query.filters)

        results = self._client.search(
            self._collection,
            query.query_embedding,
            top_k=top_k,
            filter=f if f else None,
        )

        nodes: list[BaseNode] = []
        similarities: list[float] = []
        ids: list[str] = []

        for r in results:
            meta = dict(r["metadata"])
            text = meta.pop("text", "")
            node = TextNode(
                text=text,
                id_=r["id"],
                metadata=meta,
                embedding=None,
            )
            nodes.append(node)
            # ScyllaVec returns distance (lower = more similar).
            # LlamaIndex expects similarity (higher = more similar).
            # Convert: similarity = 1 - distance (valid for cosine in [0, 2] range).
            similarities.append(1.0 - r["distance"])
            ids.append(r["id"])

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
