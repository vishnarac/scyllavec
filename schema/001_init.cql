-- ScyllaVec schema — run via scripts/init_schema.sh

CREATE KEYSPACE IF NOT EXISTS scyllavec
    WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1}
    AND durable_writes = true;

USE scyllavec;

-- Namespace for a set of vectors (like a table in a vector DB)
CREATE TABLE IF NOT EXISTS collections (
    name        text PRIMARY KEY,
    dimension   int,
    metric      text,        -- cosine | euclidean | dot_product
    description text,
    created_at  timestamp
);

-- Core vector storage — embedding co-located with metadata
CREATE TABLE IF NOT EXISTS vectors (
    collection  text,
    id          uuid,
    embedding   list<float>,
    metadata    text,        -- JSON string (flexible schema)
    created_at  timestamp,
    PRIMARY KEY (collection, id)
) WITH compaction = {'class': 'LeveledCompactionStrategy'}
  AND comment = 'Primary vector + metadata store';

-- HNSW index snapshots — serialized graph persisted as blob
CREATE TABLE IF NOT EXISTS index_snapshots (
    collection  text PRIMARY KEY,
    snapshot    blob,
    version     bigint,
    node_count  bigint,
    updated_at  timestamp
);

-- Write-ahead log for crash recovery of in-memory HNSW index
CREATE TABLE IF NOT EXISTS index_wal (
    collection  text,
    seq         timeuuid,
    op          text,        -- insert | delete
    vector_id   uuid,
    embedding   list<float>,
    PRIMARY KEY (collection, seq)
) WITH CLUSTERING ORDER BY (seq ASC)
  AND default_time_to_live = 86400;  -- WAL entries expire after 24h
