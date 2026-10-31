#!/usr/bin/env bash
set -euo pipefail
P=http://localhost:8181
TOKEN=$(curl -sf $P/api/catalog/v1/oauth/tokens \
  -d grant_type=client_credentials -d scope=PRINCIPAL_ROLE:ALL \
  -d client_id="$POLARIS_CLIENT_ID" -d client_secret="$POLARIS_CLIENT_SECRET" | jq -r .access_token)
H=(-H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json")
LOC="s3://${WAREHOUSE_BUCKET}/lake"

curl -sf -X POST $P/api/management/v1/catalogs "${H[@]}" -d @- <<EOF
{"catalog": {"name": "lake", "type": "INTERNAL",
  "properties": {"default-base-location": "$LOC"},
  "storageConfigInfo": {"storageType": "S3", "endpoint": "http://seaweedfs:8333",
    "pathStyleAccess": true, "stsUnavailable": true, "region": "us-east-1",
    "allowedLocations": ["$LOC"]}}}
EOF

curl -sf -X PUT $P/api/management/v1/catalogs/lake/catalog-roles/catalog_admin/grants "${H[@]}" \
  -d '{"grant": {"type": "catalog", "privilege": "CATALOG_MANAGE_CONTENT"}}'
curl -sf -X PUT $P/api/management/v1/principal-roles/service_admin/catalog-roles/lake "${H[@]}" \
  -d '{"catalogRole": {"name": "catalog_admin"}}'
echo "catalog 'lake' ready"
