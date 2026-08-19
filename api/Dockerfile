FROM python:3.11-slim

WORKDIR /app

# Install protobuf compiler for stub generation
RUN apt-get update && apt-get install -y protobuf-compiler && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY api/ ./api/
COPY sidecar/proto/ ./proto/

# Generate Python gRPC stubs from .proto
RUN mkdir -p api/proto_gen && \
    python -m grpc_tools.protoc \
        -I proto \
        --python_out=api/proto_gen \
        --grpc_python_out=api/proto_gen \
        proto/scyllavec.proto

WORKDIR /app/api
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
