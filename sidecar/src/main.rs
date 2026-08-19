mod db;
mod grpc;
mod index;

use std::net::SocketAddr;
use tonic::transport::Server;
use tracing::info;
use serde::Deserialize;

use grpc::proto::vector_index_server::VectorIndexServer;
use grpc::VectorIndexService;
use index::new_registry;

#[derive(Deserialize, Debug)]
struct Config {
    #[serde(default = "default_hosts")]
    scylla_hosts: String,
    #[serde(default = "default_keyspace")]
    scylla_keyspace: String,
    #[serde(default = "default_port")]
    sidecar_grpc_port: String,
}

fn default_hosts() -> String { "127.0.0.1:9042".into() }
fn default_keyspace() -> String { "scyllavec".into() }
fn default_port() -> String { "50051".into() }

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cfg: Config = envy::from_env().expect("Failed to read config from env");
    info!("Starting ScyllaVec sidecar — config: {:?}", cfg);

    let session = db::connect(&cfg.scylla_hosts, &cfg.scylla_keyspace).await;
    let registry = new_registry();

    let addr: SocketAddr = format!("0.0.0.0:{}", cfg.sidecar_grpc_port)
        .parse()
        .expect("Invalid gRPC port");

    let svc = VectorIndexService {
        registry: registry.clone(),
        session: session.clone(),
    };

    info!("gRPC server listening on {}", addr);
    Server::builder()
        .add_service(VectorIndexServer::new(svc))
        .serve(addr)
        .await?;

    Ok(())
}
