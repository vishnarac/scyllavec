use scylla::{Session, SessionBuilder};
use scylla::frame::value::CqlTimestamp;
use std::sync::Arc;
use tracing::info;
use uuid::Uuid;
use flate2::{write::GzEncoder, Compression};
use std::io::Write;

use crate::index::{CollectionIndex, Metric};

/// Fetch collection metadata (dimension, metric) from ScyllaDB.
/// Returns `None` if the collection does not exist.
pub async fn get_collection_meta(
    session: &Session,
    collection: &str,
) -> anyhow::Result<Option<(i32, Metric)>> {
    let result = session
        .query(
            "SELECT dimension, metric FROM collections WHERE name = ?",
            (collection,),
        )
        .await?;

    if let Some(rows) = result.rows {
        if let Some(row) = rows.into_iter().next() {
            let (dim, metric_str): (i32, String) = row.into_typed()?;
            let metric = match metric_str.as_str() {
                "euclidean" => Metric::Euclidean,
                "dot_product" => Metric::DotProduct,
                _ => Metric::Cosine,
            };
            return Ok(Some((dim, metric)));
        }
    }
    Ok(None)
}

/// Build a ScyllaDB session from env-supplied hosts.
/// Retries connecting to the keyspace until it exists (it may not be created yet
/// when the sidecar starts, since schema is applied after container health checks).
pub async fn connect(hosts: &str, keyspace: &str) -> Arc<Session> {
    info!("Connecting to ScyllaDB at {}", hosts);
    let host_list: Vec<&str> = hosts.split(',').collect();

    // Connect without a keyspace first — just to validate connectivity
    let session = SessionBuilder::new()
        .known_nodes(&host_list)
        .build()
        .await
        .expect("Failed to connect to ScyllaDB");

    // Retry switching to the keyspace — it may not exist yet
    loop {
        match session.use_keyspace(keyspace, false).await {
            Ok(_) => {
                info!("Connected to ScyllaDB keyspace '{}'", keyspace);
                return Arc::new(session);
            }
            Err(e) => {
                info!("Keyspace '{}' not ready yet ({}), retrying in 5s...", keyspace, e);
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            }
        }
    }
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
    let now_ms = CqlTimestamp(chrono::Utc::now().timestamp_millis());

    // Write to vectors table
    session
        .query(
            "INSERT INTO vectors (collection, id, embedding, created_at) VALUES (?, ?, ?, ?)",
            (collection, id, &emb_list, now_ms),
        )
        .await?;

    // Append WAL entry (seq is driven by the server-side now() timeuuid)
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
    let updated_at = CqlTimestamp(version);

    session
        .query(
            "INSERT INTO index_snapshots (collection, snapshot, version, node_count, updated_at) VALUES (?, ?, ?, ?, ?)",
            (collection, compressed.as_slice(), version, node_count, updated_at),
        )
        .await?;

    info!("Snapshot saved for '{}' — {} nodes, version {}", collection, node_count, version);
    Ok(())
}
