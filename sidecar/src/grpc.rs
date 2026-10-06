use tonic::{Request, Response, Status};
use uuid::Uuid;
use std::str::FromStr;
use std::sync::Arc;
use scylla::Session;

use crate::index::IndexRegistry;
use crate::db;

// Generated gRPC stubs (via build.rs)
pub mod proto {
    tonic::include_proto!("scyllavec");
}

use proto::vector_index_server::VectorIndex;
use proto::*;

pub struct VectorIndexService {
    pub registry: IndexRegistry,
    pub session: Arc<Session>,
}

#[tonic::async_trait]
impl VectorIndex for VectorIndexService {
    async fn upsert(
        &self,
        request: Request<UpsertRequest>,
    ) -> Result<Response<UpsertResponse>, Status> {
        let req = request.into_inner();
        let id = Uuid::from_str(&req.vector_id)
            .map_err(|_| Status::invalid_argument("invalid UUID"))?;

        // Persist to ScyllaDB
        db::persist_vector(&self.session, &req.collection, id, &req.embedding)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        // Update in-memory index — lazy-load if not yet registered.
        // Must load existing vectors first so a post-restart upsert doesn't
        // silently discard the pre-restart contents of the index.
        {
            let reg = self.registry.read().await;
            if !reg.contains_key(&req.collection) {
                drop(reg);
                let meta = db::get_collection_meta(&self.session, &req.collection)
                    .await
                    .map_err(|e| Status::internal(e.to_string()))?;
                match meta {
                    Some((dim, metric)) => {
                        let mut reg_w = self.registry.write().await;
                        // Double-check after acquiring write lock
                        if !reg_w.contains_key(&req.collection) {
                            let mut idx =
                                crate::index::CollectionIndex::new(dim as usize, metric);
                            db::load_collection(&self.session, &req.collection, &mut idx)
                                .await
                                .map_err(|e| Status::internal(e.to_string()))?;
                            reg_w.insert(req.collection.clone(), idx);
                        }
                    }
                    None => {
                        return Err(Status::not_found(format!(
                            "collection '{}' not found — create it first via the REST API",
                            req.collection
                        )));
                    }
                }
            }
        }
        let mut reg = self.registry.write().await;
        reg.get_mut(&req.collection).unwrap().upsert(id, req.embedding);

        Ok(Response::new(UpsertResponse { success: true }))
    }

    async fn delete(
        &self,
        request: Request<DeleteRequest>,
    ) -> Result<Response<DeleteResponse>, Status> {
        let req = request.into_inner();
        let id = Uuid::from_str(&req.vector_id)
            .map_err(|_| Status::invalid_argument("invalid UUID"))?;

        db::delete_vector(&self.session, &req.collection, id)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        // Lazy-load the collection so the in-memory index is consistent even
        // if this delete arrives before any search on a freshly restarted sidecar.
        {
            let reg = self.registry.read().await;
            if !reg.contains_key(&req.collection) {
                drop(reg);
                if let Ok(Some((dim, metric))) =
                    db::get_collection_meta(&self.session, &req.collection).await
                {
                    let mut reg_w = self.registry.write().await;
                    if !reg_w.contains_key(&req.collection) {
                        let mut idx = crate::index::CollectionIndex::new(dim as usize, metric);
                        let _ = db::load_collection(&self.session, &req.collection, &mut idx).await;
                        reg_w.insert(req.collection.clone(), idx);
                    }
                }
            }
        }
        let mut reg = self.registry.write().await;
        if let Some(idx) = reg.get_mut(&req.collection) {
            idx.delete(&id);
        }

        Ok(Response::new(DeleteResponse { success: true }))
    }

    async fn search(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let req = request.into_inner();

        // Lazy-load the collection if not yet in the registry
        {
            let reg = self.registry.read().await;
            if !reg.contains_key(&req.collection) {
                drop(reg);
                let meta = db::get_collection_meta(&self.session, &req.collection)
                    .await
                    .map_err(|e| Status::internal(e.to_string()))?;
                match meta {
                    Some((dim, metric)) => {
                        let mut reg_w = self.registry.write().await;
                        // Double-check after acquiring write lock
                        if !reg_w.contains_key(&req.collection) {
                            let mut idx =
                                crate::index::CollectionIndex::new(dim as usize, metric);
                            db::load_collection(&self.session, &req.collection, &mut idx)
                                .await
                                .map_err(|e| Status::internal(e.to_string()))?;
                            reg_w.insert(req.collection.clone(), idx);
                        }
                    }
                    None => {
                        return Err(Status::not_found(format!(
                            "collection '{}' not found",
                            req.collection
                        )));
                    }
                }
            }
        }

        let reg = self.registry.read().await;
        let idx = reg.get(&req.collection).unwrap();
        let results = idx.search(&req.query, req.top_k as usize)
            .into_iter()
            .map(|(id, dist)| SearchResult {
                vector_id: id.to_string(),
                distance: dist,
            })
            .collect();

        Ok(Response::new(SearchResponse { results }))
    }

    async fn snapshot(
        &self,
        request: Request<SnapshotRequest>,
    ) -> Result<Response<SnapshotResponse>, Status> {
        let req = request.into_inner();
        let reg = self.registry.read().await;
        let idx = reg.get(&req.collection)
            .ok_or_else(|| Status::not_found(format!("collection '{}' not found", req.collection)))?;

        let nodes = idx.len() as u64;
        db::save_snapshot(&self.session, &req.collection, idx)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(SnapshotResponse {
            success: true,
            version: chrono::Utc::now().timestamp_millis() as u64,
            nodes,
        }))
    }

    async fn drop_collection(
        &self,
        request: Request<DropCollectionRequest>,
    ) -> Result<Response<DropCollectionResponse>, Status> {
        let req = request.into_inner();
        let mut reg = self.registry.write().await;
        reg.remove(&req.collection);
        Ok(Response::new(DropCollectionResponse { success: true }))
    }

    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        Ok(Response::new(HealthResponse { status: "ok".into() }))
    }
}
