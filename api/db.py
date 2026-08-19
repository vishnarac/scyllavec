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
    hosts = os.getenv("SCYLLA_HOSTS", "127.0.0.1").split(",")
    keyspace = os.getenv("SCYLLA_KEYSPACE", "scyllavec")

    log.info("Connecting to ScyllaDB: hosts=%s keyspace=%s", hosts, keyspace)
    cluster = Cluster(
        contact_points=hosts,
        load_balancing_policy=TokenAwarePolicy(DCAwareRoundRobinPolicy()),
        protocol_version=4,
    )
    session = cluster.connect(keyspace)
    _session = session
    log.info("ScyllaDB connected")
    return session
