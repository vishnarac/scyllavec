"""
Unit tests for the LangChain adapter.
No running stack required — ScyllaVecClient is fully mocked.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sdk"))

from unittest.mock import MagicMock, patch
import pytest

from scyllavec.langchain import ScyllaVecVectorStore


# ── Fixtures ─────────────────────────────────────────────────────────────────

class FakeEmbeddings:
    """Deterministic embeddings: each word in text → its ASCII sum as a float."""
    def embed_documents(self, texts):
        return [[float(sum(ord(c) for c in t))] * 4 for t in texts]

    def embed_query(self, text):
        return [float(sum(ord(c) for c in text))] * 4


@pytest.fixture()
def store():
    with patch("scyllavec.langchain.ScyllaVecClient") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        s = ScyllaVecVectorStore(
            collection="test",
            embedding=FakeEmbeddings(),
        )
        s._mock_client = mock_client
        yield s


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_add_texts_calls_upsert(store):
    store._mock_client.upsert.return_value = {"ids": ["id-0", "id-1"], "inserted": 2}
    ids = store.add_texts(["hello", "world"], metadatas=[{"src": "a"}, {"src": "b"}])
    assert ids == ["id-0", "id-1"]
    store._mock_client.upsert.assert_called_once()
    call_args = store._mock_client.upsert.call_args
    vectors = call_args[0][1]
    assert len(vectors) == 2
    assert vectors[0]["metadata"]["text"] == "hello"
    assert vectors[0]["metadata"]["src"] == "a"


def test_add_texts_passes_ids(store):
    store._mock_client.upsert.return_value = {"ids": ["my-id"], "inserted": 1}
    store.add_texts(["hi"], ids=["my-id"])
    vectors = store._mock_client.upsert.call_args[0][1]
    assert vectors[0]["id"] == "my-id"


def test_similarity_search_returns_documents(store):
    store._mock_client.search.return_value = [
        {"id": "doc-1", "distance": 0.1, "metadata": {"text": "ScyllaDB is fast", "src": "a"}},
        {"id": "doc-2", "distance": 0.3, "metadata": {"text": "Rust is safe", "src": "b"}},
    ]
    docs = store.similarity_search("database", k=2)
    assert len(docs) == 2
    assert docs[0].page_content == "ScyllaDB is fast"
    assert docs[0].metadata["src"] == "a"
    assert docs[0].id == "doc-1"


def test_similarity_search_with_score(store):
    store._mock_client.search.return_value = [
        {"id": "doc-1", "distance": 0.05, "metadata": {"text": "hello"}},
    ]
    results = store.similarity_search_with_score("hello", k=1)
    assert len(results) == 1
    doc, score = results[0]
    assert doc.page_content == "hello"
    assert score == pytest.approx(0.05)


def test_similarity_search_passes_filter(store):
    store._mock_client.search.return_value = []
    store.similarity_search("query", k=5, filter={"tag": "science"})
    store._mock_client.search.assert_called_once_with(
        "test", [pytest.approx(x) for x in FakeEmbeddings().embed_query("query")],
        top_k=5, filter={"tag": "science"},
    )


def test_from_texts_creates_collection_and_upserts():
    with patch("scyllavec.langchain.ScyllaVecClient") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.upsert.return_value = {"ids": ["0", "1"], "inserted": 2}

        store = ScyllaVecVectorStore.from_texts(
            texts=["foo", "bar"],
            embedding=FakeEmbeddings(),
            collection="new-col",
            dimension=4,
        )

        mock_client.create_collection.assert_called_once_with(
            "new-col", dimension=4, metric="cosine"
        )
        mock_client.upsert.assert_called_once()
        assert isinstance(store, ScyllaVecVectorStore)
