"""
gRPC client stub for communicating with the Rust HNSW sidecar.
"""
import grpc
import os
import sys
import pathlib

# Add generated proto stubs to path
sys.path.insert(0, str(pathlib.Path(__file__).parent / "proto_gen"))

import scyllavec_pb2 as pb
import scyllavec_pb2_grpc as pb_grpc

_channel: grpc.Channel | None = None
_stub: pb_grpc.VectorIndexStub | None = None


def get_stub() -> pb_grpc.VectorIndexStub:
    global _channel, _stub
    if _stub is None:
        url = os.getenv("SIDECAR_URL", "localhost:50051")
        # Strip any scheme prefix — gRPC channels need bare host:port
        if "://" in url:
            url = url.split("://", 1)[1]
        _channel = grpc.insecure_channel(url)
        _stub = pb_grpc.VectorIndexStub(_channel)
    return _stub


def upsert(collection: str, vector_id: str, embedding: list[float]) -> bool:
    stub = get_stub()
    resp = stub.Upsert(pb.UpsertRequest(
        collection=collection,
        vector_id=vector_id,
        embedding=embedding,
    ))
    return resp.success


def search(collection: str, query: list[float], top_k: int = 10) -> list[dict]:
    stub = get_stub()
    resp = stub.Search(pb.SearchRequest(
        collection=collection,
        query=query,
        top_k=top_k,
    ))
    return [{"vector_id": r.vector_id, "distance": r.distance} for r in resp.results]


def delete(collection: str, vector_id: str) -> bool:
    stub = get_stub()
    resp = stub.Delete(pb.DeleteRequest(collection=collection, vector_id=vector_id))
    return resp.success


def snapshot(collection: str) -> dict:
    stub = get_stub()
    resp = stub.Snapshot(pb.SnapshotRequest(collection=collection))
    return {"success": resp.success, "version": resp.version, "nodes": resp.nodes}


def evict(collection: str) -> bool:
    stub = get_stub()
    resp = stub.DropCollection(pb.DropCollectionRequest(collection=collection))
    return resp.success


def health() -> str:
    stub = get_stub()
    resp = stub.Health(pb.HealthRequest())
    return resp.status
