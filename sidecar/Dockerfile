FROM rust:1.89-slim AS builder

WORKDIR /app

# Required for tonic-build (protobuf compiler)
RUN apt-get update && apt-get install -y protobuf-compiler && rm -rf /var/lib/apt/lists/*

# Build context is ./sidecar — paths are relative to that directory
COPY Cargo.toml Cargo.lock* ./
COPY build.rs ./
COPY proto ./proto
COPY src ./src

RUN cargo build --release

# ── Runtime image ─────────────────────────────────────────────────────────────
FROM debian:bookworm-slim
WORKDIR /app
COPY --from=builder /app/target/release/sidecar ./sidecar
CMD ["./sidecar"]
