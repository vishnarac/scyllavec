use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use instant_distance::{Builder, HnswMap, Point, Search};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Metric used for similarity scoring.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Metric {
    Euclidean,
    Cosine,
    DotProduct,
}

impl Metric {
    pub fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Metric::Euclidean => a.iter().zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt(),
            Metric::Cosine => {
                let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let na: f32 = a.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
                let nb: f32 = b.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
                1.0 - dot / (na * nb + 1e-9)
            }
            Metric::DotProduct => {
                -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
            }
        }
    }
}

/// A float vector that carries its metric so the HNSW graph uses the right
/// distance function during both build and search.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VecPoint {
    pub data: Vec<f32>,
    pub metric: Metric,
}

impl Point for VecPoint {
    fn distance(&self, other: &Self) -> f32 {
        self.metric.distance(&self.data, &other.data)
    }
}

/// In-memory index for a single collection.
///
/// Write path:  upsert / delete update `vectors` (HashMap) and set `dirty`.
/// Read path:   search() rebuilds the HNSW graph from `vectors` when dirty,
///              then queries it.  At small sizes (<= BRUTE_FORCE_THRESHOLD)
///              we skip the rebuild and fall back to brute-force to avoid the
///              overhead of building a graph for a handful of vectors.
pub struct CollectionIndex {
    pub metric: Metric,
    pub dimension: usize,
    /// Authoritative vector store — also used for HNSW rebuild and snapshots.
    pub vectors: HashMap<Uuid, Vec<f32>>,
    /// HNSW graph, rebuilt lazily when `dirty` is true.
    hnsw: Option<HnswMap<VecPoint, Uuid>>,
    /// True whenever `vectors` has changed since the last HNSW build.
    dirty: bool,
}

/// Below this count we skip HNSW and do brute-force — building a graph for
/// fewer than ~50 vectors costs more than it saves.
const BRUTE_FORCE_THRESHOLD: usize = 50;

impl CollectionIndex {
    pub fn new(dimension: usize, metric: Metric) -> Self {
        Self {
            metric,
            dimension,
            vectors: HashMap::new(),
            hnsw: None,
            dirty: false,
        }
    }

    pub fn upsert(&mut self, id: Uuid, embedding: Vec<f32>) {
        self.vectors.insert(id, embedding);
        self.dirty = true;
    }

    pub fn delete(&mut self, id: &Uuid) {
        if self.vectors.remove(id).is_some() {
            self.dirty = true;
        }
    }

    /// Rebuild the HNSW graph from the current `vectors` map.
    /// Called lazily from `search()` when `dirty` is true and the collection
    /// is large enough to benefit from HNSW.
    fn rebuild(&mut self) {
        let n = self.vectors.len();
        if n < BRUTE_FORCE_THRESHOLD {
            // Too small — brute-force is faster; keep hnsw as None.
            self.hnsw = None;
            self.dirty = false;
            return;
        }

        let mut points: Vec<VecPoint> = Vec::with_capacity(n);
        let mut values: Vec<Uuid> = Vec::with_capacity(n);
        for (id, emb) in &self.vectors {
            points.push(VecPoint { data: emb.clone(), metric: self.metric.clone() });
            values.push(*id);
        }

        self.hnsw = Some(Builder::default().build(points, values));
        self.dirty = false;
    }

    /// Search for the top-k nearest neighbours to `query`.
    /// Uses HNSW when the collection is large enough, brute-force otherwise.
    /// Returns `Vec<(uuid, distance)>` sorted ascending by distance.
    pub fn search(&mut self, query: &[f32], top_k: usize) -> Vec<(Uuid, f32)> {
        if self.vectors.is_empty() {
            return vec![];
        }

        // Rebuild HNSW graph if the index is stale.
        if self.dirty {
            self.rebuild();
        }

        match &self.hnsw {
            Some(map) => {
                let qpoint = VecPoint { data: query.to_vec(), metric: self.metric.clone() };
                let mut search = Search::default();
                map.search(&qpoint, &mut search)
                    .take(top_k)
                    .map(|item| (*item.value, item.distance))
                    .collect()
            }
            None => {
                // Brute-force path (small collections or empty hnsw).
                let mut scores: Vec<(Uuid, f32)> = self.vectors.iter()
                    .map(|(id, emb)| (*id, self.metric.distance(query, emb)))
                    .collect();
                scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                scores.truncate(top_k);
                scores
            }
        }
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }
}

/// Registry of all collection indexes, shared across gRPC handlers.
pub type IndexRegistry = Arc<RwLock<HashMap<String, CollectionIndex>>>;

pub fn new_registry() -> IndexRegistry {
    Arc::new(RwLock::new(HashMap::new()))
}
