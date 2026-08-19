"""
ScyllaVec REST API — FastAPI application.

Endpoints:
  POST   /collections                   create a collection
  GET    /collections                   list collections
  DELETE /collections/{name}            drop a collection

  POST   /collections/{name}/vectors    upsert vectors (batch)
  DELETE /collections/{name}/vectors/{id}  delete a vector

  POST   /collections/{name}/search     ANN search
  POST   /collections/{name}/snapshot   force HNSW snapshot to ScyllaDB

  GET    /health                        liveness check
"""
from __future__ import annotations
import uuid
import json
import logging
from typing import Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from db import init_db, get_session
import sidecar_client as sidecar

log = logging.getLogger(__name__)

# ── Pydantic models ──────────────────────────────────────────────────────────

class CollectionCreate(BaseModel):
    name: str
    dimension: int
    metric: str = Field(default="cosine", pattern="^(cosine|euclidean|dot_product)$")
    description: str = ""

class VectorItem(BaseModel):
    id: str | None = None          # auto-generated if omitted
    embedding: list[float]
    metadata: dict[str, Any] = {}

class UpsertRequest(BaseModel):
    vectors: list[VectorItem]

class SearchRequest(BaseModel):
    query: list[float]
    top_k: int = Field(default=10, ge=1, le=1000)
    filter: dict[str, Any] = {}    # metadata filter (phase 2)

# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(title="ScyllaVec API", version="0.1.0", lifespan=lifespan)

# ── Collections ───────────────────────────────────────────────────────────────

@app.post("/collections", status_code=status.HTTP_201_CREATED)
def create_collection(body: CollectionCreate):
    session = get_session()
    session.execute(
        """INSERT INTO collections (name, dimension, metric, description, created_at)
           VALUES (%s, %s, %s, %s, toTimestamp(now()))
           IF NOT EXISTS""",
        (body.name, body.dimension, body.metric, body.description),
    )
    return {"name": body.name, "dimension": body.dimension, "metric": body.metric}


@app.get("/collections")
def list_collections():
    session = get_session()
    rows = session.execute("SELECT name, dimension, metric, description, created_at FROM collections")
    return [dict(r._asdict()) for r in rows]


@app.delete("/collections/{name}", status_code=status.HTTP_204_NO_CONTENT)
def delete_collection(name: str):
    session = get_session()
    session.execute("DELETE FROM collections WHERE name = %s", (name,))
    session.execute("DELETE FROM vectors WHERE collection = %s", (name,))
    session.execute("DELETE FROM index_snapshots WHERE collection = %s", (name,))

# ── Vectors ───────────────────────────────────────────────────────────────────

@app.post("/collections/{name}/vectors", status_code=status.HTTP_201_CREATED)
def upsert_vectors(name: str, body: UpsertRequest):
    session = get_session()
    inserted = []
    for item in body.vectors:
        vid = item.id or str(uuid.uuid4())
        vid_uuid = uuid.UUID(vid)
        meta_json = json.dumps(item.metadata)

        # Persist to ScyllaDB via sidecar (which also updates the HNSW index)
        ok = sidecar.upsert(name, vid, item.embedding)
        if not ok:
            raise HTTPException(500, detail=f"Sidecar upsert failed for id={vid}")

        # Store metadata separately (sidecar only handles embeddings)
        session.execute(
            "UPDATE vectors SET metadata = %s WHERE collection = %s AND id = %s",
            (meta_json, name, vid_uuid),
        )
        inserted.append(vid)

    return {"inserted": len(inserted), "ids": inserted}


@app.delete("/collections/{name}/vectors/{vector_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_vector(name: str, vector_id: str):
    sidecar.delete(name, vector_id)

# ── Search ────────────────────────────────────────────────────────────────────

@app.post("/collections/{name}/search")
def search_vectors(name: str, body: SearchRequest):
    session = get_session()

    # ANN search via sidecar
    ann_results = sidecar.search(name, body.query, body.top_k)

    if not ann_results:
        return {"results": []}

    # Fetch metadata for returned IDs
    ids = [uuid.UUID(r["vector_id"]) for r in ann_results]
    id_list = ", ".join(["%s"] * len(ids))
    rows = session.execute(
        f"SELECT id, metadata FROM vectors WHERE collection = %s AND id IN ({id_list})",
        [name, *ids],
    )
    meta_map = {str(r.id): json.loads(r.metadata or "{}") for r in rows}

    results = [
        {
            "id": r["vector_id"],
            "distance": r["distance"],
            "metadata": meta_map.get(r["vector_id"], {}),
        }
        for r in ann_results
    ]
    return {"results": results}

# ── Snapshot ──────────────────────────────────────────────────────────────────

@app.post("/collections/{name}/snapshot")
def trigger_snapshot(name: str):
    result = sidecar.snapshot(name)
    return result

# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    sidecar_status = sidecar.health()
    return {"api": "ok", "sidecar": sidecar_status}
