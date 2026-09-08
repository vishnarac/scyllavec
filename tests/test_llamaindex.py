"""
Unit tests for the LlamaIndex adapter.
No running stack required — ScyllaVecClient is fully mocked.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sdk"))

from unittest.mock import MagicMock, patch
import pytest

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
)

from scyllavec.llamaindex import ScyllaVecVectorStore, _filters_to_dict


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def store():
    with patch("scyllavec.llamaindex.ScyllaVecClient") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        s = ScyllaVecVectorStore(collection="test", dimension=4)
        s._mock_client = mock_client
        yield s


# ── _filters_to_dict helper ───────────────────────────────────────────────────

def test_filters_to_dict_empty():
    assert _filters_to_dict(None) == {}
    assert _filters_to_dict(MetadataFilters(filters=[])) == {}


def test_filters_to_dict_eq():
    f = MetadataFilters(filters=[
        MetadataFilter(key="tag", value="science", operator=FilterOperator.EQ),
        MetadataFilter(key="year", value=2024, operator=FilterOperator.EQ),
    ])
    assert _filters_to_dict(f) == {"tag": "science", "year": 2024}


def test_filters_to_dict_skips_non_eq():
    f = MetadataFilters(filters=[
        MetadataFilter(key="tag", value="a", operator=FilterOperator.EQ),
        MetadataFilter(key="score", value=5, operator=FilterOperator.GT),
    ])
    result = _filters_to_dict(f)
    assert result == {"tag": "a"}   # GT is skipped


# ── add ───────────────────────────────────────────────────────────────────────

def test_add_upserts_nodes(store):
    store._mock_client.upsert.return_value = {"ids": ["node-1"], "inserted": 1}
    node = TextNode(text="ScyllaDB is fast", id_="node-1", embedding=[0.1, 0.2, 0.3, 0.4])
    ids = store.add([node])
    assert ids == ["node-1"]
    call = store._mock_client.upsert.call_args[0]
    items = call[1]
    assert items[0]["id"] == "node-1"
    assert items[0]["metadata"]["text"] == "ScyllaDB is fast"
    assert items[0]["embedding"] == [0.1, 0.2, 0.3, 0.4]


def test_add_skips_nodes_without_embedding(store):
    store._mock_client.upsert.return_value = {"ids": [], "inserted": 0}
    node = TextNode(text="no embedding", id_="n1")  # no embedding set
    ids = store.add([node])
    assert ids == []
    store._mock_client.upsert.assert_not_called()


# ── delete ────────────────────────────────────────────────────────────────────

def test_delete_calls_client(store):
    store.delete("node-1")
    store._mock_client.delete_vector.assert_called_once_with("test", "node-1")


# ── query ─────────────────────────────────────────────────────────────────────

def test_query_returns_results(store):
    store._mock_client.search.return_value = [
        {"id": "n1", "distance": 0.1, "metadata": {"text": "hello", "src": "a"}},
        {"id": "n2", "distance": 0.4, "metadata": {"text": "world", "src": "b"}},
    ]
    q = VectorStoreQuery(query_embedding=[0.1, 0.2, 0.3, 0.4], similarity_top_k=2)
    result = store.query(q)
    assert len(result.nodes) == 2
    assert result.nodes[0].get_content() == "hello"
    assert result.nodes[0].metadata["src"] == "a"
    assert result.similarities[0] == pytest.approx(0.9)   # 1 - 0.1
    assert result.similarities[1] == pytest.approx(0.6)   # 1 - 0.4
    assert result.ids == ["n1", "n2"]


def test_query_no_embedding_returns_empty(store):
    q = VectorStoreQuery(query_embedding=None, similarity_top_k=5)
    result = store.query(q)
    assert result.nodes == []
    assert result.ids == []
    store._mock_client.search.assert_not_called()


def test_query_passes_metadata_filter(store):
    store._mock_client.search.return_value = []
    f = MetadataFilters(filters=[
        MetadataFilter(key="tag", value="ai", operator=FilterOperator.EQ)
    ])
    q = VectorStoreQuery(
        query_embedding=[0.1, 0.2, 0.3, 0.4],
        similarity_top_k=3,
        filters=f,
    )
    store.query(q)
    store._mock_client.search.assert_called_once_with(
        "test", [0.1, 0.2, 0.3, 0.4], top_k=3, filter={"tag": "ai"}
    )
