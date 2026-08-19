use scylla::{Session, SessionBuilder};
use std::sync::Arc;
use tracing::info;
use uuid::Uuid;
use flate2::{write::GzEncoder, read::GzDecoder, Compression};
use std::io::{Write, Read};

use crate::index::{CollectionIndex, IndexRegistry, Metric};

/// Build a ScyllaDB session from env-supplied hosts.
pub async fn connect(hosts: &str, keyspace: &str) -> Arc<Session> {
    info!("Connecting to ScyllaDB at {}", hosts);
    let host_list: Vec<&str> = hosts.split(',').collect();
    let session = SessionBuilder::new()
        .known_nodes(&host_list)
        .use_keyspace(keyspace, false)
        .build()
        .await
        .expect("Failed to connect to ScyllaDB");
    info!("Connected to ScyllaDB keyspace '{}'", keyspace);
    Arc::new(session)
}

/// Load all vectors for a collection from ScyllaDB into the in-memory index.
pub async fn load_collection(
    session: &Session,
    collection: &str,
    index: &mut CollectionIndex,
) -> anyhow::Result<()> {
    let rows = session
        .query(
            "SELECT id, embedding FROM vectors WHERE collection = ?",
            (collection,),
        )
        .await?
        .rows_typed::<(Uuid, Vec<f32>)>()?;

    let mut count = 0usize;
    for row in rows {
        let (id, emb) = row?;
        index.upsert(id, emb);
        count += 1;
    }
    info!("Loaded {} vectors for collection '{}'", count, collection);
    Ok(())
}

/// Persist a single vector + WAL entry to ScyllaDB.
pub async fn persist_vector(
    session: &Session,
    collection: &str,
    id: Uuid,
    embedding: &[f32],
) -> anyhow::Result<()> {
    let emb_list: Vec<f32> = embedding.to_vec();
    let now = chrono::Utc::now();

    // Write to vectors table
    session
        .query(
            "INSERT INTO vectors (collection, id, embedding, created_at) VALUES (?, ?, ?, ?)",
            (collection, id, &emb_list, now),
        )
        .await?;

    // Append WAL entry
    let seq = uuid::Uuid::now_v7();
    session
        .query(
            "INSERT INTO index_wal (collection, seq, op, vector_id, embedding) VALUES (?, now(), 'insert', ?, ?)",
            (collection, id, &emb_list),
        )
        .await?;

    Ok(())
}

/// Delete a vector from ScyllaDB.
pub async fn delete_vector(
    session: &Session,
    collection: &str,
    id: Uuid,
) -> anyhow::Result<()> {
    session
        .query(
            "DELETE FROM vectors WHERE collection = ? AND id = ?",
            (collection, id),
        )
        .await?;
    session
        .query(
            "INSERT INTO index_wal (collection, seq, op, vector_id, embedding) VALUES (?, now(), 'delete', ?, [])",
            (collection, id),
        )
        .await?;
    Ok(())
}

/// Serialize and compress the collection's vector map, store as blob.
pub async fn save_snapshot(
    session: &Session,
    collection: &str,
    index: &CollectionIndex,
) -> anyhow::Result<()> {
    let raw = bincode::serialize(&index.vectors)?;
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&raw)?;
    let compressed = encoder.finish()?;

    let version: i64 = chrono::Utc::now().timestamp_millis();
    let node_count = index.len() as i64;

    session
        .query(
            "INSERT INTO index_snapshots (collection, snapshot, version, node_count, updated_at) VALUES (?, ?, ?, ?, ?)",
            (collection, compressed.as_slice(), version, node_count, chrono::Utc::now()),
        )
        .await?;

    info!("Snapshot saved for '{}' — {} nodes, version {}", collection, node_count, version);
    Ok(())
}
