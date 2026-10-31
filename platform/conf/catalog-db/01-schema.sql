-- Polaris connects with currentSchema=POLARIS_SCHEMA (unquoted, so Postgres folds it to
-- polaris_schema) but never issues CREATE SCHEMA itself — the docs require the schema to
-- exist before Polaris connects (releases/1.8.0/metastores/relational-jdbc).
-- Runs once on first init of the catalog-db volume (docker-entrypoint-initdb.d).
CREATE SCHEMA IF NOT EXISTS POLARIS_SCHEMA;
