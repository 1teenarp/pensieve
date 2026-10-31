#!/usr/bin/env bash
# usage: scripts/download_tlc.sh 2024-01 2024-02 ...   (or: $(echo 2024-{01..12}))
set -euo pipefail
# DATA_ROOT comes from .env; load it so the script also works outside make
if [ -z "${DATA_ROOT:-}" ]; then
  [ -f .env ] || { echo "DATA_ROOT is not set and no .env found; run from the repo root" >&2; exit 1; }
  set -a; . ./.env; set +a
fi
DEST="${DATA_ROOT}/landing/tlc/yellow"; mkdir -p "$DEST"
for m in "$@"; do
  curl -fSL -o "$DEST/yellow_tripdata_${m}.parquet" \
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_${m}.parquet"
done
