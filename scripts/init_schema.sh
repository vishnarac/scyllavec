#!/usr/bin/env bash
# init_schema.sh — apply CQL schema to a running ScyllaDB instance
set -euo pipefail

SCHEMA_FILE="$(dirname "$0")/../schema/001_init.cql"

echo "⏳  Waiting for ScyllaDB at localhost:9042..."
until docker compose exec scylla cqlsh -e "DESCRIBE KEYSPACES" &>/dev/null 2>&1; do
  sleep 3
  printf "."
done
echo ""

echo "✅  ScyllaDB ready. Applying schema..."
docker compose exec -T scylla cqlsh < "$SCHEMA_FILE"
echo "✅  Schema applied."
