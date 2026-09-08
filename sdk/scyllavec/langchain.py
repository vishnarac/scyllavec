"""
LangChain VectorStore adapter for ScyllaVec.

Usage::

    from langchain_community.embeddings import HuggingFaceEmbeddings
    from scyllavec.langchain import ScyllaVecVectorStore

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # From scratch
    store = ScyllaVecVectorStore.from_texts(
        texts=["Hello world", "ScyllaDB is fast"],
        embedding=embeddings,
        collection="docs",
        dimension=384,
    )

    # Attach to an existing collection
    store = ScyllaVecVectorStore(
        collection="docs",
        embedding=embeddings,
        base_url="http://localhost:8000",
    )

    results = store.similarity_search("fast database", k=3)
    results_with_scores = store.similarity_search_with_score("fast database", k=3)
"""
from __future__ import annotations

from typing import Any, Iterable

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "langchain-core is required for the LangChain adapter. "
        "Install it with: pip install langchain-core"
    ) from e

from .client import ScyllaVecClient


class ScyllaVecVectorStore(VectorStore):
    """LangChain-compatible VectorStore backed by ScyllaVec."""

    def __init__(
        self,
        collection: str,
        embedding: Embeddings,
        base_url: str = "http://localhost:8000",
        metric: str = "cosine",
    ) -> None:
        self._collection = collection
        self._embedding = embedding
        self._metric = metric
        self._client = ScyllaVecClient(base_url)

    # ── LangChain interface ───────────────────────────────────────────────────

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        texts = list(texts)
        vectors = self._embedding.embed_documents(texts)
        metadatas = metadatas or [{} for _ in texts]

        items = []
        for i, (text, vec, meta) in enumerate(zip(texts, vectors, metadatas)):
            item: dict[str, Any] = {
                "embedding": vec,
                "metadata": {**meta, "text": text},
            }
            if ids:
                item["id"] = ids[i]
            items.append(item)

        result = self._client.upsert(self._collection, items)
        return result["ids"]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k=k, filter=filter)]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        vec = self._embedding.embed_query(query)
        results = self._client.search(self._collection, vec, top_k=k, filter=filter)
        return [
            (
                Document(
                    page_content=r["metadata"].pop("text", ""),
                    metadata=r["metadata"],
                    id=r["id"],
                ),
                r["distance"],
            )
            for r in results
        ]

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        collection: str = "langchain",
        dimension: int | None = None,
        metric: str = "cosine",
        base_url: str = "http://localhost:8000",
        **kwargs: Any,
    ) -> "ScyllaVecVectorStore":
        if dimension is None:
            dimension = len(embedding.embed_query(texts[0]))
        client = ScyllaVecClient(base_url)
        try:
            client.create_collection(collection, dimension=dimension, metric=metric)
        except Exception:
            pass  # already exists
        store = cls(collection=collection, embedding=embedding, base_url=base_url, metric=metric)
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store
