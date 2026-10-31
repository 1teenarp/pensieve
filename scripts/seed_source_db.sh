#!/usr/bin/env bash
set -euo pipefail
curl -fsSL https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv -o /tmp/zones.csv
./dc exec -T source-db psql -U ops -d ops \
  -c "\copy taxi_zones(location_id, borough, zone, service_zone) FROM STDIN CSV HEADER" < /tmp/zones.csv
