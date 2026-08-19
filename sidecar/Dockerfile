FROM rust:1.78-slim AS builder

WORKDIR /app

# Required for tonic-build (protobuf compiler)
RUN apt-get update && apt-get install -y protobuf-compiler && rm -rf /var/lib/apt/lists/*

COPY sidecar/Cargo.toml sidecar/Cargo.lock* ./
COPY sidecar/build.rs ./
COPY sidecar/proto ./proto
COPY sidecar/src ./src

RUN cargo build --release

# ── Runtime image ─────────────────────────────────────────────────────────────
FROM debian:bookworm-slim
WORKDIR /app
COPY --from=builder /app/target/release/sidecar ./sidecar
CMD ["./sidecar"]
