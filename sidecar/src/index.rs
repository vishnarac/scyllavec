use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use instant_distance::Point;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A single float vector treated as an HNSW point.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VecPoint(pub Vec<f32>);

impl Point for VecPoint {
    fn distance(&self, other: &Self) -> f32 {
        // Euclidean distance (default); metric override applied at query layer
        self.0.iter()
            .zip(other.0.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

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
                // negate so lower = more similar (distance semantics)
                -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
            }
        }
    }
}

/// In-memory index for a single collection.
pub struct CollectionIndex {
    pub metric: Metric,
    pub dimension: usize,
    /// Raw vectors keyed by UUID — source of truth for re-builds
    pub vectors: HashMap<Uuid, Vec<f32>>,
}

impl CollectionIndex {
    pub fn new(dimension: usize, metric: Metric) -> Self {
        Self { metric, dimension, vectors: HashMap::new() }
    }

    pub fn upsert(&mut self, id: Uuid, embedding: Vec<f32>) {
        self.vectors.insert(id, embedding);
    }

    pub fn delete(&mut self, id: &Uuid) {
        self.vectors.remove(id);
    }

    /// Brute-force top-k search (used until HNSW is wired in next phase).
    /// Returns Vec<(uuid, distance)> sorted ascending by distance.
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(Uuid, f32)> {
        let mut scores: Vec<(Uuid, f32)> = self.vectors.iter()
            .map(|(id, emb)| (*id, self.metric.distance(query, emb)))
            .collect();
        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scores.truncate(top_k);
        scores
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
