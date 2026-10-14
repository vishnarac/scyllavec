"""
Unit tests for API key auth middleware.
No running stack required — uses FastAPI's TestClient.
"""
from __future__ import annotations
import sys
import os
import types
from unittest.mock import MagicMock, patch

# ── Stub proto modules (same pattern as test_sharding.py) ────────────────────
class _AutoMockModule(types.ModuleType):
    def __getattr__(self, name: str):
        m = MagicMock(name=name)
        setattr(self, name, m)
        return m

for _mod in ("scyllavec_pb2", "scyllavec_pb2_grpc"):
    sys.modules.setdefault(_mod, _AutoMockModule(_mod))

# ── Stub out grpc and cassandra so api/main.py imports cleanly ────────────────
sys.modules.setdefault("grpc", _AutoMockModule("grpc"))
for _mod in ("cassandra", "cassandra.cluster", "cassandra.policies"):
    sys.modules.setdefault(_mod, _AutoMockModule(_mod))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import importlib
import pytest
from fastapi.testclient import TestClient


def _make_client(api_key_env: str = "") -> TestClient:
    """Load api.main with the given API_KEY env value and return a TestClient."""
    with patch.dict(os.environ, {"API_KEY": api_key_env}, clear=False):
        # Stub db and sidecar_client so the app can start without real services
        fake_db = types.ModuleType("db")
        fake_db.init_db = lambda: None
        fake_db.get_session = lambda: MagicMock()
        fake_sc = _AutoMockModule("sidecar_client")
        fake_sc.health = lambda: "ok"

        with patch.dict(sys.modules, {"db": fake_db, "sidecar_client": fake_sc}):
            import api.main as mod
            importlib.reload(mod)
            return TestClient(mod.app, raise_server_exceptions=False)


# ── /health is always public ──────────────────────────────────────────────────

def test_health_public_no_key_configured():
    client = _make_client(api_key_env="")
    assert client.get("/health").status_code == 200


def test_health_public_even_when_key_configured():
    client = _make_client(api_key_env="secret")
    assert client.get("/health").status_code == 200


# ── Auth disabled (no API_KEY set) ────────────────────────────────────────────

def test_no_auth_collections_accessible_without_key():
    client = _make_client(api_key_env="")
    assert client.get("/collections").status_code == 200


# ── Auth enabled ──────────────────────────────────────────────────────────────

def test_auth_rejects_missing_key():
    client = _make_client(api_key_env="secret")
    assert client.get("/collections").status_code == 401


def test_auth_rejects_wrong_key():
    client = _make_client(api_key_env="secret")
    assert client.get("/collections", headers={"X-API-Key": "wrong"}).status_code == 401


def test_auth_accepts_correct_key():
    client = _make_client(api_key_env="secret")
    assert client.get("/collections", headers={"X-API-Key": "secret"}).status_code == 200


def test_auth_protects_upsert():
    client = _make_client(api_key_env="secret")
    r = client.post("/collections/test/vectors", json={"vectors": []})
    assert r.status_code == 401


def test_auth_protects_search():
    client = _make_client(api_key_env="secret")
    r = client.post("/collections/test/search", json={"query": [0.1]})
    assert r.status_code == 401


def test_auth_protects_delete_collection():
    client = _make_client(api_key_env="secret")
    assert client.delete("/collections/test").status_code == 401
