#!/usr/bin/env bash
# dev.sh — one command to start the full local stack
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "🚀  Starting ScyllaVec local stack..."
cd "$ROOT"

docker compose up -d --build

echo ""
echo "⏳  Waiting for ScyllaDB to be healthy..."
until docker compose exec scylla cqlsh -e "DESCRIBE KEYSPACES" &>/dev/null 2>&1; do
  sleep 4
  printf "."
done
echo ""

echo "📦  Applying schema..."
docker compose exec -T scylla cqlsh < schema/001_init.cql

echo ""
echo "✅  Stack is up!"
echo ""
echo "   REST API   → http://localhost:8000"
echo "   API docs   → http://localhost:8000/docs"
echo "   Sidecar 0  → localhost:50051"
echo "   Sidecar 1  → localhost:50052"
echo "   ScyllaDB   → localhost:9042"
echo ""
echo "Run seed:   python scripts/seed.py"
echo "Run tests:  pytest tests/ -v"
echo "Stop:       docker compose down"
