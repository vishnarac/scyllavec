"""
ScyllaDB connection pool — shared across the FastAPI app.
Uses the pure-Python cassandra-driver which is compatible with ScyllaDB.
"""
from cassandra.cluster import Cluster, Session
from cassandra.policies import DCAwareRoundRobinPolicy, TokenAwarePolicy
from cassandra.auth import PlainTextAuthProvider
import os
import logging

log = logging.getLogger(__name__)

_session: Session | None = None


def get_session() -> Session:
    global _session
    if _session is None:
        raise RuntimeError("DB not initialised — call init_db() first")
    return _session


def init_db() -> Session:
    global _session
    raw_hosts = os.getenv("SCYLLA_HOSTS", "127.0.0.1").split(",")
    keyspace = os.getenv("SCYLLA_KEYSPACE", "scyllavec")

    # Strip any :port suffix — Cluster takes hosts and port separately
    hosts = [h.split(":")[0] for h in raw_hosts]
    port = int(raw_hosts[0].split(":")[1]) if ":" in raw_hosts[0] else 9042

    log.info("Connecting to ScyllaDB: hosts=%s port=%d keyspace=%s", hosts, port, keyspace)
    cluster = Cluster(
        contact_points=hosts,
        port=port,
        load_balancing_policy=TokenAwarePolicy(DCAwareRoundRobinPolicy()),
        protocol_version=4,
    )
    session = cluster.connect(keyspace)
    _session = session
    log.info("ScyllaDB connected")
    return session
