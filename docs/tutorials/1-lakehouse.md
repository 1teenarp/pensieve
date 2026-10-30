# Build a Spark Lakehouse on One Box

Oct 29, 2026 · @Praneet

## Start here

By the end you'll have one repo where `make up SHAPE=large` brings up a full lakehouse on your EPYC box: S3-compatible storage, an Iceberg catalog, a Spark cluster you size per shape, Trino for interactive SQL, notebooks over Spark Connect, and Airflow scheduling the ETL. Every piece maps to an AWS service, so the skills carry over.

**Before Phase 1, finish Tutorial 0** (Set up Pensieve: Host, Repo and Working Agreement). It prepares the host, creates the `pensieve` repo with its layout, conventions, `versions.env`, `./dc` and Makefile, and sets up the agent workflow. Each phase here is then one branch, one pull request, one journal entry and one tag, from `t1-p01` to `t1-p10`. To hand a phase to the agent: `Read AGENTS.md, then do phase t1-p01 from docs/tutorials/1-lakehouse.md.`

**How each phase works:**

1. **Build** — files to create and commands to run.
2. **Verify** — a check that proves it works. Don't move on until it passes.
3. **Commit** — a suggested commit message, so your Git history reads like the tutorial.
4. **Explore (optional)** — experiments that use what you've built so far to understand it more deeply.

&#91;embedded content: lakehouse architecture · clients, engines, catalog, storage\]

Engines ask Polaris where a table's current metadata is, then read and write files in SeaweedFS directly. Postgres stores only the catalog's own state.

### Pinned versions (as of Oct 2026)

| Component | Version | Role |
| --- | --- | --- |
| [Apache Spark](https://hub.docker.com/r/amd64/spark/) | 4.1.2, image `spark:4.1.2-scala2.13-java17-python3-ubuntu` | Compute |
| [Apache Iceberg](https://iceberg.apache.org/multi-engine-support/) | 1.11.0, `iceberg-spark-runtime-4.1_2.13` | Table format |
| [Apache Polaris](https://www.fivetran.com/blog/the-impact-of-apache-polaris-graduating-to-top-level-apache-project) | Latest 1.x, `apache/polaris` | Iceberg REST catalog |
| [SeaweedFS](https://duckdb.org/docs/current/guides/network_cloud_storage/seaweedfs_import) | Pin a recent tag of `chrislusf/seaweedfs` | S3-compatible object store |
| PostgreSQL | 17 | Catalog database and a sample source database |
| [Trino](https://polaris.apache.org/guides/trino/) | Pin a recent `trinodb/trino` tag | Interactive SQL |
| [Apache Airflow](https://www.getorchestra.io/guides/airflow-version) | 3.3.2 | Orchestration |

Two choices worth knowing:

- **Spark 4.1, not 4.2.** [Spark 4.2.0](https://spark.apache.org/releases/spark-release-4-2-0.html) shipped in July 2026, but Iceberg 1.11 only publishes runtimes up to Spark 4.1. Move to 4.2 once Iceberg ships `iceberg-spark-runtime-4.2`.
- **SeaweedFS, not MinIO.** MinIO's community edition went into [maintenance mode in late 2025](https://dev.to/alexandrev/minio-in-maintenance-mode-what-it-means-for-the-community-edition-oem-users-and-open-source-3o4j) and no longer publishes official images. SeaweedFS is Apache-2.0 and has a one-process `mini` mode that suits one box.

### What maps to what on AWS

| On your box | On AWS | Job |
| --- | --- | --- |
| SeaweedFS | S3 | Stores Parquet data and Iceberg metadata files |
| Apache Polaris | Glue Data Catalog | Tracks which tables exist and where their current metadata is |
| Spark standalone in Docker | EMR or Glue jobs | Batch ETL |
| Trino | Athena (built on Trino) | Interactive SQL on the catalog |
| Airflow 3 | Managed Workflows for Apache Airflow | Scheduling and chaining jobs |
| PostgreSQL | RDS | Catalog storage, and a stand-in for an app database |
| Prometheus and Grafana | CloudWatch or Amazon Managed Prometheus | Metrics |

## Phase 1: S3-compatible object storage

Goal: a SeaweedFS S3 endpoint at `http://seaweedfs:8333` inside the `lake` network, with `warehouse` and `landing` buckets. This is your S3.

### Build

`conf/seaweedfs/s3.json.template` (rendered to `s3.json`, which Git ignores):

```json
{
  "identities": [
    {
      "name": "lakeadmin",
      "credentials": [{ "accessKey": "${S3_ACCESS_KEY}", "secretKey": "${S3_SECRET_KEY}" }],
      "actions": ["Admin", "Read", "Write", "List", "Tagging"]
    }
  ]
}
```

`compose.storage.yaml`:

```yaml
services:
  seaweedfs:
    image: chrislusf/seaweedfs:${SEAWEEDFS_TAG}
    command: mini -dir=/data -s3.config=/etc/seaweedfs/s3.json
    ports:
      - "8333:8333"        # S3 API, published so host tools can reach it
    volumes:
      - ${DATA_ROOT}/seaweedfs:/data
      - ./conf/seaweedfs/s3.json:/etc/seaweedfs/s3.json:ro
    restart: unless-stopped
    networks: [lake]

networks:
  lake:
    external: true
```

`mini` runs SeaweedFS's master, volume server, filer and S3 gateway in one process. It also starts its own Iceberg REST catalog on port 8181 inside the container. You won't use that one, since Polaris is the catalog.

Save it as `platform/compose.storage.yaml`. There's no root `compose.yaml` to edit: `./dc` passes Compose every `platform/compose.*.yaml` file, so a new stack is picked up as soon as its file exists. Every stack joins the same external `lake` network, created by `make net`, which avoids clashes between files that would each try to define it.

Add to the `Makefile` (install `gettext-base` for `envsubst`, and `jq` for later):

```makefile
conf:
	envsubst < platform/conf/seaweedfs/s3.json.template > platform/conf/seaweedfs/s3.json

buckets:
	for b in $(WAREHOUSE_BUCKET) landing; do \
	  echo "s3.bucket.create -name $$b" | $(DC) exec -T seaweedfs weed shell -master=localhost:9333; \
	done

# usage: make s3 ARGS="ls s3://warehouse/ --recursive"
s3:
	docker run --rm --network lake \
	  -e AWS_ACCESS_KEY_ID=$(S3_ACCESS_KEY) -e AWS_SECRET_ACCESS_KEY=$(S3_SECRET_KEY) \
	  -e AWS_DEFAULT_REGION=us-east-1 -v $(PWD):/work -w /work \
	  amazon/aws-cli:$(AWS_CLI_TAG) --endpoint-url http://seaweedfs:8333 s3 $(ARGS)
```

`make up` works from this phase on: it starts whichever core stack files exist so far, and only adds the worker-scaling flag once the Spark stack exists.

### Verify

```bash
make buckets
make s3 ARGS="ls"                          # shows warehouse and landing
echo hello > /tmp/hi.txt && cp /tmp/hi.txt .
make s3 ARGS="cp hi.txt s3://landing/hi.txt"
make s3 ARGS="ls s3://landing/"
```

### Commit

`feat(storage): SeaweedFS S3 endpoint with warehouse and landing buckets`

### Explore (optional)

- **Look at how objects are really stored.** List `${DATA_ROOT}/seaweedfs`. You'll find a few large volume files, not one file per object. Read up on why object stores pack small objects together, and what that means for millions of tiny Parquet files.
- **Measure upload throughput.** Copy a 5 GB file with `make s3`, then raise `multipart_chunksize` and the number of concurrent requests and compare. Spark writes work the same way: many parallel multipart uploads.
- **Peek at SeaweedFS's built-in catalog.** Run `./dc exec seaweedfs wget -qO- localhost:8181/v1/config`. That's the same REST protocol Polaris will speak. Note it so you can compare the two in Phase 3.
- **Break it on purpose.** Use a wrong secret key in `make s3` and read the error. You'll see the same `403` shape from Spark later, and recognizing it saves time.

## Phase 2: A Spark standalone cluster in Docker

Goal: one master, N workers sized by your shape file, and a history server, all from one custom image. `make up SHAPE=small` gives 2 workers; another shape gives a different cluster.

### Build

`docker/spark/Dockerfile` bakes in the Iceberg runtime, the Iceberg AWS bundle (for S3) and the Postgres JDBC driver, so jobs never download jars at runtime:

```dockerfile
ARG SPARK_BASE
FROM ${SPARK_BASE}
USER root
ARG ICEBERG_VERSION=1.11.0
ARG PG_JDBC_VERSION=42.7.7
ARG MVN=https://repo1.maven.org/maven2
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl python3-pip numactl iproute2 procps && rm -rf /var/lib/apt/lists/*
RUN cd /opt/spark/jars && \
    curl -fsSLO $MVN/org/apache/iceberg/iceberg-spark-runtime-4.1_2.13/$ICEBERG_VERSION/iceberg-spark-runtime-4.1_2.13-$ICEBERG_VERSION.jar && \
    curl -fsSLO $MVN/org/apache/iceberg/iceberg-aws-bundle/$ICEBERG_VERSION/iceberg-aws-bundle-$ICEBERG_VERSION.jar && \
    curl -fsSLO $MVN/org/postgresql/postgresql/$PG_JDBC_VERSION/postgresql-$PG_JDBC_VERSION.jar
# newer pip needs the flag, older pip rejects it
COPY requirements.lock /tmp/requirements.lock      # requirements.in: pyarrow, pandas, requests
RUN pip3 install --no-cache-dir -r /tmp/requirements.lock || \
    pip3 install --no-cache-dir --break-system-packages -r /tmp/requirements.lock
COPY lake-submit /opt/spark/bin/lake-submit
RUN chmod +x /opt/spark/bin/lake-submit && mkdir -p /opt/spark/events /scratch && \
    chown -R spark:spark /opt/spark/events /scratch
ENV PATH=/opt/spark/bin:$PATH PYTHONPATH=/opt/libs   # libs/ is mounted at /opt/libs, so `import pensieve` works
USER spark
```

`docker/spark/lake-submit` wraps `spark-submit`:

```bash
#!/usr/bin/env bash
# Executors connect back to the driver. A container's hostname often isn't resolvable
# by other containers, so advertise the container's IP instead.
exec /opt/spark/bin/spark-submit --conf spark.driver.host="$(hostname -i)" "$@"
```

`conf/spark/spark-defaults.conf.template` (Phase 3 adds the catalog):

```properties
spark.master                          spark://spark-master:7077
spark.driver.memory                   8g
spark.executor.cores                  ${EXECUTOR_CORES}
spark.executor.memory                 ${EXECUTOR_MEMORY}
spark.cores.max                       ${APP_MAX_CORES}
spark.serializer                      org.apache.spark.serializer.KryoSerializer
spark.sql.adaptive.enabled            true
spark.eventLog.enabled                true
spark.eventLog.dir                    file:///opt/spark/events
spark.history.fs.logDirectory         file:///opt/spark/events
spark.ui.reverseProxy                 true
spark.ui.reverseProxyUrl              http://${HOST_NAME}:8080
```

`spark.cores.max` matters more than it looks. In standalone mode, an application grabs every free core unless capped. Without the cap, your first notebook session would starve every batch job.

The reverse proxy lets the master UI on port 8080 link to worker and application UIs. That's useful because scaled workers can't all publish the same port.

`compose.spark.yaml`:

```yaml
x-spark: &spark
  image: lakehouse/spark:4.1.2
  build: { context: ./docker/spark, args: [SPARK_BASE, ICEBERG_VERSION] }
  networks: [lake]
  environment:
    SOURCE_DB_PASSWORD: ${SOURCE_DB_PASSWORD}
  volumes:
    - ./conf/spark:/opt/spark/conf:ro
    - ../pipelines/jobs:/opt/jobs:ro
    - ../libs:/opt/libs:ro
    - ${DATA_ROOT}/spark-events:/opt/spark/events
    - ${DATA_ROOT}/landing:/data/landing

services:
  spark-master:
    <<: *spark
    command: /opt/spark/bin/spark-class org.apache.spark.deploy.master.Master --host spark-master --port 7077 --webui-port 8080
    ports: ["8080:8080"]

  spark-worker:
    <<: *spark
    command: >
      /opt/spark/bin/spark-class org.apache.spark.deploy.worker.Worker
      --cores ${SPARK_WORKER_CORES} --memory ${SPARK_WORKER_MEMORY} spark://spark-master:7077
    environment:
      SOURCE_DB_PASSWORD: ${SOURCE_DB_PASSWORD}
      SPARK_LOCAL_DIRS: /scratch
      SPARK_WORKER_OPTS: -Dspark.worker.cleanup.enabled=true -Dspark.worker.cleanup.appDataTtl=86400
    volumes:                      # a YAML merge replaces lists, so repeat them
      - ./conf/spark:/opt/spark/conf:ro
      - ../pipelines/jobs:/opt/jobs:ro
      - ../libs:/opt/libs:ro
      - ${DATA_ROOT}/spark-events:/opt/spark/events
      - ${DATA_ROOT}/landing:/data/landing
      - ${SCRATCH_DIR}:/scratch
    ports: ["8081-8096:8081"]     # one host port per scaled worker
    cpus: ${SPARK_WORKER_CORES}
    mem_limit: ${SPARK_WORKER_MEM_LIMIT}
    depends_on: [spark-master]

  spark-history:
    <<: *spark
    command: /opt/spark/bin/spark-class org.apache.spark.deploy.history.HistoryServer
    ports: ["18080:18080"]

  spark-client:                   # throwaway driver container for batch jobs
    <<: *spark
    profiles: [tools]
    entrypoint: []

networks:
  lake:
    external: true
```

The Docker limits (`cpus`, `mem_limit`) and Spark's own limits (`--cores`, `--memory`) do different jobs. Spark's limits decide what it schedules. Docker's limits stop a runaway process from hurting the rest of the box. Keep the Docker memory limit about 15% above the worker memory, for JVM overhead.

Save it as `platform/compose.spark.yaml`, and add these targets to `make/core.mk`:

```makefile
conf:
	envsubst < platform/conf/seaweedfs/s3.json.template > platform/conf/seaweedfs/s3.json
	envsubst < platform/conf/spark/spark-defaults.conf.template > platform/conf/spark/spark-defaults.conf

# usage: make submit JOB=ingest/tlc_bronze.py ARGS="--month 2024-01"
submit:
	$(DC) run --rm spark-client lake-submit $(SUBMIT_OPTS) /opt/jobs/$(JOB) $(ARGS)

sql:
	$(DC) run --rm spark-client /opt/spark/bin/spark-sql --conf spark.driver.host=$$(hostname -i)
```

The `sql` target is a placeholder that won't resolve the right IP from the host. Phase 3 replaces it with a container-side version.

### Verify

```bash
make up
make ps                                    # master, 2 workers, history
# open http://pensive:8080 — 2 workers, 16 cores total
./dc run --rm spark-client sh -c \
  'lake-submit --class org.apache.spark.examples.SparkPi /opt/spark/examples/jars/spark-examples_*.jar 2000'
# open http://pensive:18080 — the finished SparkPi run is listed
```

Then resize: create `shapes/large.env` with, say, 4 workers of 12 cores and 100g, and run `make up SHAPE=large`. The master UI should show the new totals.

### Commit

`feat(spark): standalone cluster image, sized workers, history server`

### Explore (optional)

- **Kill a worker mid-job.** Start SparkPi with a big slice count, then `docker kill` one worker. In the application UI, watch the lost tasks get rescheduled. This is fault tolerance you'll never see in DuckDB.
- **Same cores, different shapes.** Run one job with 2 workers of 16 cores, then 8 workers of 4 cores. Compare stage times and the number of JVMs in `docker stats`.
- **Read an event log.** Open a file in `${DATA_ROOT}/spark-events` with `head`. It's JSON, one event per line. The history server just replays it, which is why it can show jobs that ended long ago.
- **Feel the core grab.** Set `APP_MAX_CORES` to the full cluster, start two jobs at once, and watch the second one wait in the master UI. Then set it back.

## Phase 3: An Iceberg REST catalog with Apache Polaris

Goal: Polaris running on Postgres, a catalog named `lake` whose tables live in `s3://warehouse/lake`, and Spark creating and reading Iceberg tables through it. This is your Glue Data Catalog.

A catalog stores surprisingly little: for each table, a pointer to its current metadata file. The data and the metadata live in object storage. That pointer is what makes commits atomic, and it's why every engine must agree on one catalog.

### Build

`compose.catalog.yaml`:

```yaml
x-polaris-db: &polaris-db
  POLARIS_PERSISTENCE_TYPE: relational-jdbc
  QUARKUS_DATASOURCE_JDBC_URL: jdbc:postgresql://catalog-db:5432/polaris
  QUARKUS_DATASOURCE_USERNAME: polaris
  QUARKUS_DATASOURCE_PASSWORD: ${CATALOG_DB_PASSWORD}

services:
  catalog-db:
    image: postgres:${POSTGRES_TAG}
    environment:
      POSTGRES_USER: polaris
      POSTGRES_PASSWORD: ${CATALOG_DB_PASSWORD}
      POSTGRES_DB: polaris
    volumes:
      - ${PG_ROOT}/catalog-db:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U polaris"]
      interval: 5s
    networks: [lake]

  polaris-bootstrap:            # one-time setup, run by `make catalog-bootstrap`
    image: apache/polaris-admin-tool:${POLARIS_TAG}
    profiles: [setup]
    environment: *polaris-db
    command: ["bootstrap", "--realm=POLARIS", "--credential=POLARIS,${POLARIS_CLIENT_ID},${POLARIS_CLIENT_SECRET}"]
    depends_on:
      catalog-db: { condition: service_healthy }
    networks: [lake]

  polaris:
    image: apache/polaris:${POLARIS_TAG}
    environment:
      <<: *polaris-db
      POLARIS_REALM_CONTEXT_REALMS: POLARIS
      POLARIS_READINESS_IGNORE_SEVERE_ISSUES: "true"   # dev-only, see Phase 10
      AWS_ACCESS_KEY_ID: ${S3_ACCESS_KEY}               # Polaris reads metadata files itself
      AWS_SECRET_ACCESS_KEY: ${S3_SECRET_KEY}
      AWS_REGION: us-east-1
    ports: ["8181:8181", "8182:8182"]                  # REST API, health
    depends_on:
      catalog-db: { condition: service_healthy }
    networks: [lake]

networks:
  lake:
    external: true
```

Bootstrap is a separate one-time step, not a dependency of `polaris`. Running it on every `up` would fail once the realm exists. The JDBC settings follow [Polaris's relational-JDBC docs](https://polaris.apache.org/in-dev/unreleased/metastores/relational-jdbc/).

`scripts/create-catalog.sh` creates the `lake` catalog and grants your admin principal rights to manage its tables:

```bash
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
```

`stsUnavailable: true` tells Polaris that SeaweedFS has no STS service, so it won't try to hand out temporary credentials. Spark and Trino use static S3 keys instead ([Polaris S3 storage docs](https://polaris.apache.org/in-dev/unreleased/configuration/configuring-polaris-for-production/configuring-aws-s3-cloud-storage-specific/)). On real AWS you'd let Polaris vend short-lived credentials per table, which is safer.

Append the catalog to `conf/spark/spark-defaults.conf.template`:

```properties
spark.sql.extensions                          org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions
spark.sql.defaultCatalog                      lake
spark.sql.catalog.lake                        org.apache.iceberg.spark.SparkCatalog
spark.sql.catalog.lake.type                   rest
spark.sql.catalog.lake.uri                    http://polaris:8181/api/catalog
spark.sql.catalog.lake.warehouse              lake
spark.sql.catalog.lake.credential             ${POLARIS_CLIENT_ID}:${POLARIS_CLIENT_SECRET}
spark.sql.catalog.lake.scope                  PRINCIPAL_ROLE:ALL
spark.sql.catalog.lake.io-impl                org.apache.iceberg.aws.s3.S3FileIO
spark.sql.catalog.lake.s3.endpoint            http://seaweedfs:8333
spark.sql.catalog.lake.s3.path-style-access   true
spark.sql.catalog.lake.s3.access-key-id       ${S3_ACCESS_KEY}
spark.sql.catalog.lake.s3.secret-access-key   ${S3_SECRET_KEY}
spark.sql.catalog.lake.client.region          us-east-1
```

Note that `warehouse` is the Polaris catalog name, not an S3 path. Save the Compose file as `platform/compose.catalog.yaml`, then extend `make/core.mk` and fix the `sql` target so the driver IP is read inside the container:

```makefile
catalog-bootstrap:
	$(DC) run --rm polaris-bootstrap

catalog-create:
	until curl -sf localhost:8182/q/health >/dev/null; do sleep 2; done
	./scripts/create-catalog.sh

sql:
	$(DC) run --rm spark-client sh -c 'spark-sql --conf spark.driver.host=$$(hostname -i)'
```

### Verify

```bash
make up && make catalog-bootstrap && make catalog-create
make sql
```

In the `spark-sql` prompt:

```sql
CREATE NAMESPACE lake.demo;
CREATE TABLE lake.demo.hello (id BIGINT, msg STRING) USING iceberg;
INSERT INTO lake.demo.hello VALUES (1, 'hi'), (2, 'there');
SELECT * FROM lake.demo.hello;
SELECT snapshot_id, operation, summary FROM lake.demo.hello.snapshots;
```

Then `make s3 ARGS="ls --recursive s3://warehouse/lake/"` shows `data/` (Parquet) and `metadata/` (JSON and Avro) for the table.

**Troubleshooting.**

- `NoClassDefFoundError ... FormatModelRegistry` on reads: a [known Iceberg 1.11.0 issue](https://github.com/apache/iceberg/issues/16856). Use a newer 1.11.x patch if one exists, or fall back to Spark 4.0.2 with `iceberg-spark-runtime-4.0_2.13`.
- `403` or "bucket not found": path-style access is off, or the keys in `.env` don't match `s3.json`. Re-run `make conf`.
- "Unable to load region": `client.region` is missing.
- `401` from Polaris: the bootstrap credentials and `.env` disagree.

### Commit

`feat(catalog): Polaris on Postgres with lake catalog; Spark wired to Iceberg REST`

### Explore (optional)

- **See how little a catalog holds.** Run `./dc exec catalog-db psql -U polaris -d polaris` and look at the tables. Find the row for `demo.hello` and its metadata location. Everything else is in S3.
- **Walk the metadata tree by hand.** Download the latest `metadata.json`, then follow it to the manifest list and a manifest (both Avro; read them with `fastavro` in Python). That's how a query finds files without listing directories.
- **Least privilege, IAM-style.** Create a principal `etl` and a principal role that can write only to one namespace, using the management API. Point a Spark session at it and confirm it can't drop tables elsewhere.
- **Concurrent commits.** Run two `INSERT`s into the same table at the same time from two sessions. Iceberg uses optimistic concurrency: one commit retries against the new snapshot. Look for both snapshots afterwards.
- **Compare with SeaweedFS's catalog.** Point a second Spark catalog at `http://seaweedfs:8181`. What does Polaris give you (access control, multiple catalogs, federation) that a built-in catalog doesn't?

## Phase 4: Submitting jobs and working interactively

Goal: two ways in. Batch jobs run as throwaway driver containers with `make submit`, each sized per job. Interactive work goes through a long-running Spark Connect server, so a Jupyter notebook holds no JVM and no cluster connection of its own.

### Build: batch jobs

Every job follows one shape: arguments in, one session, writes by table name only. Never hardcode S3 paths in jobs. That's what lets the same code run on AWS later.

`libs/pensieve/lake/spark.py`:

```python
from pyspark.sql import SparkSession

def get_spark(app_name: str) -> SparkSession:
    # Cluster, catalog and S3 settings come from spark-defaults.conf.
    return SparkSession.builder.appName(app_name).getOrCreate()
```

`jobs/demo/row_count.py`:

```python
import argparse, sys
sys.path.insert(0, "/opt/jobs")
from pensieve.lake.spark import get_spark

p = argparse.ArgumentParser()
p.add_argument("--table", required=True)
args = p.parse_args()

spark = get_spark(f"row_count:{args.table}")
print(args.table, spark.table(args.table).count())
spark.stop()
```

Sizing per job uses ordinary Spark flags through `SUBMIT_OPTS`:

```bash
make submit JOB=demo/row_count.py ARGS="--table lake.demo.hello"
make submit JOB=demo/row_count.py ARGS="--table lake.demo.hello" \
  SUBMIT_OPTS="--conf spark.cores.max=32 --executor-cores 8 --executor-memory 48g"
```

So there are two levels of size: the **cluster shape** (`SHAPE=`, how many workers exist) and the **job size** (`SUBMIT_OPTS`, how much of the cluster one job takes). On EMR these map to the cluster or capacity configuration and to per-step Spark settings.

### Build: Spark Connect and Jupyter

Add to `compose.spark.yaml`:

```yaml
  spark-connect:
    <<: *spark
    command: >
      /opt/spark/sbin/start-connect-server.sh
      --conf spark.driver.host=spark-connect
      --conf spark.cores.max=${CONNECT_MAX_CORES:-16}
      --conf spark.dynamicAllocation.enabled=true
      --conf spark.dynamicAllocation.shuffleTracking.enabled=true
      --conf spark.dynamicAllocation.executorIdleTimeout=120s
    environment:
      SPARK_NO_DAEMONIZE: "true"           # keep the server in the foreground
      SOURCE_DB_PASSWORD: ${SOURCE_DB_PASSWORD}
    ports: ["15002:15002", "4040:4040"]   # Connect gRPC, driver UI
    depends_on: [spark-master]
```

Dynamic allocation lets the shared server give executors back when notebooks sit idle. In standalone mode it needs shuffle tracking, because there's no external shuffle service here.

`docker/jupyter/Dockerfile`:

```dockerfile
ARG JUPYTER_TAG
FROM quay.io/jupyter/minimal-notebook:${JUPYTER_TAG}
# Match the client version to the server (4.1.2).
COPY requirements.lock /tmp/requirements.lock
# requirements.in: pyspark-client==4.1.2, pandas, pyarrow, grpcio, grpcio-status, protobuf, googleapis-common-protos
RUN pip install --no-cache-dir -r /tmp/requirements.lock
```

`compose.notebook.yaml`:

```yaml
services:
  jupyter:
    build: { context: ./docker/jupyter, args: [JUPYTER_TAG] }
    ports: ["8890:8888"]
    volumes:
      - ../notebooks:/home/jovyan/work
    networks: [lake]

networks:
  lake:
    external: true
```

In a notebook:

```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.remote("sc://spark-connect:15002").getOrCreate()
spark.sql("SHOW NAMESPACES IN lake").show()
df = spark.table("lake.demo.hello")
df.toPandas()          # results come back as Arrow batches over gRPC
```

### Verify

- `make submit JOB=demo/row_count.py ARGS="--table lake.demo.hello"` prints `2`.
- Start the notebook stack with make up-with S=notebook. The Jupyter token is in `./dc logs jupyter`. Open `http://pensive:8890`, run the cell above, and see the table.
- In the master UI, `Spark Connect server` shows as one long-running application, separate from your batch jobs.

### Commit

`feat(jobs): job template, make submit, Spark Connect server and Jupyter`

### Explore (optional)

- **Where does the code run?** With Connect, the notebook sends a query plan and the server's driver runs it. Run `import pyspark; pyspark.__file__` in the notebook, then try `spark.sparkContext`. It fails, because there's no local JVM. That's the whole design.
- **Starve, then share.** Raise `CONNECT_MAX_CORES` to the whole cluster, run a heavy notebook query, and submit a batch job at the same time. Watch the batch job wait. Then try FAIR scheduling pools inside the Connect server for several notebook users.
- **Arrow in action.** Time `df.toPandas()` on a few million rows with Connect, then in a batch job with `spark.sql.execution.arrow.pyspark.enabled` set to `false` and to `true`. This connects back to your Arrow question.
- **A job with tests.** Move the logic of a job into a pure function that takes a DataFrame and returns one. Test it with `pytest` and a local `SparkSession` on tiny inputs. You'll run these tests in CI in Phase 10.

## Phase 5: Ingesting from several sources into bronze, silver and gold

Goal: three real sources flowing into layered Iceberg tables. NYC taxi trips arrive as Parquet files, the zone lookup lives in an app database (Postgres), and hourly weather comes from a public REST API. The gold table answers a real question: how do trips and tips change with the weather, by borough?

| Layer | Rule | Tables |
| --- | --- | --- |
| Bronze | Raw as received, plus ingest metadata. Append or replace by batch. | `bronze.tlc_yellow`, `bronze.weather_hourly` |
| Silver | Typed, cleaned, deduplicated, partitioned for queries | `silver.trips`, `silver.zones` |
| Gold | Business-level aggregates, ready for dashboards | `gold.daily_zone_weather` |

New Compose files go in `platform/` and new jobs in `pipelines/jobs/`. The sources stack is part of the core, so `make up` starts it.

### Source 1: Parquet files in a landing zone

`scripts/download_tlc.sh`:

```bash
#!/usr/bin/env bash
# usage: scripts/download_tlc.sh 2024-01 2024-02 ...   (or: $(echo 2024-{01..12}))
set -euo pipefail
DEST="${DATA_ROOT}/landing/tlc/yellow"; mkdir -p "$DEST"
for m in "$@"; do
  curl -fSL -o "$DEST/yellow_tripdata_${m}.parquet" \
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_${m}.parquet"
done
```

The files' schemas drift between months: column names change case and some types change. That's normal for real data. The bronze job fixes it by casting every file to one canonical schema.

`jobs/ingest/tlc_bronze.py`:

```python
import argparse, sys
sys.path.insert(0, "/opt/jobs")
from pyspark.sql import functions as F
from pensieve.lake.spark import get_spark

CANONICAL = {
    "vendorid": "int", "tpep_pickup_datetime": "timestamp", "tpep_dropoff_datetime": "timestamp",
    "passenger_count": "double", "trip_distance": "double", "ratecodeid": "double",
    "store_and_fwd_flag": "string", "pulocationid": "int", "dolocationid": "int",
    "payment_type": "bigint", "fare_amount": "double", "extra": "double", "mta_tax": "double",
    "tip_amount": "double", "tolls_amount": "double", "improvement_surcharge": "double",
    "total_amount": "double", "congestion_surcharge": "double", "airport_fee": "double",
}

p = argparse.ArgumentParser()
p.add_argument("--month", required=True)                      # e.g. 2024-01
p.add_argument("--landing", default="file:///data/landing/tlc/yellow")
a = p.parse_args()

spark = get_spark(f"tlc_bronze:{a.month}")
raw = spark.read.parquet(f"{a.landing}/yellow_tripdata_{a.month}.parquet")
actual = {c.lower(): c for c in raw.columns}
df = raw.select(
    *[(F.col(actual[n]) if n in actual else F.lit(None)).cast(t).alias(n) for n, t in CANONICAL.items()],
    F.col("_metadata.file_path").alias("_source_file"),
).withColumn("_ingest_month", F.lit(a.month)).withColumn("_ingested_at", F.current_timestamp())

spark.sql("CREATE NAMESPACE IF NOT EXISTS lake.bronze")
if spark.catalog.tableExists("lake.bronze.tlc_yellow"):
    df.writeTo("lake.bronze.tlc_yellow").overwritePartitions()    # rerun = same result
else:
    df.writeTo("lake.bronze.tlc_yellow").partitionedBy(F.col("_ingest_month")).create()
```

`overwritePartitions()` replaces only the month being loaded. Running the job twice gives the same table, which is the property Airflow retries depend on.

The `file:///data/landing` path works because every worker mounts the same folder. On AWS the landing zone would be an S3 prefix read through `s3a://`; see the exercises.

### Source 2: An app database over JDBC

`compose.sources.yaml`:

```yaml
services:
  source-db:
    image: postgres:${POSTGRES_TAG}
    environment:
      POSTGRES_USER: ops
      POSTGRES_PASSWORD: ${SOURCE_DB_PASSWORD}
      POSTGRES_DB: ops
    volumes:
      - ${PG_ROOT}/source-db:/var/lib/postgresql/data
      - ./conf/source-db:/docker-entrypoint-initdb.d:ro
    networks: [lake]

networks:
  lake:
    external: true
```

`conf/source-db/01-schema.sql`:

```sql
CREATE TABLE taxi_zones (
  location_id  INT PRIMARY KEY,
  borough      TEXT,
  zone         TEXT,
  service_zone TEXT,
  updated_at   TIMESTAMPTZ DEFAULT now()
);
```

Seed it with `scripts/seed_source_db.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
curl -fsSL https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv -o /tmp/zones.csv
./dc exec -T source-db psql -U ops -d ops \
  -c "\copy taxi_zones(location_id, borough, zone, service_zone) FROM STDIN CSV HEADER" < /tmp/zones.csv
```

`jobs/ingest/zones_from_pg.py` upserts the dimension with `MERGE`, so changes in the app database flow through:

```python
import os, sys
sys.path.insert(0, "/opt/jobs")
from pensieve.lake.spark import get_spark

spark = get_spark("zones_from_pg")
src = (spark.read.format("jdbc")
       .option("url", "jdbc:postgresql://source-db:5432/ops")
       .option("dbtable", "public.taxi_zones")
       .option("user", "ops").option("password", os.environ["SOURCE_DB_PASSWORD"])
       .load())
src.createOrReplaceTempView("zones_src")

spark.sql("CREATE NAMESPACE IF NOT EXISTS lake.silver")
spark.sql("""CREATE TABLE IF NOT EXISTS lake.silver.zones
             (location_id INT, borough STRING, zone STRING, service_zone STRING, updated_at TIMESTAMP)
             USING iceberg""")
spark.sql("""MERGE INTO lake.silver.zones t USING zones_src s
             ON t.location_id = s.location_id
             WHEN MATCHED AND s.updated_at > t.updated_at THEN UPDATE SET *
             WHEN NOT MATCHED THEN INSERT *""")
```

### Source 3: A REST API

APIs return small amounts of data, so call them once from the driver, not from executors. Then hand the result to Spark; `createDataFrame` from pandas goes through Arrow.

`jobs/ingest/weather_bronze.py`:

```python
import argparse, sys
import pandas as pd, requests
sys.path.insert(0, "/opt/jobs")
from pyspark.sql import functions as F
from pensieve.lake.spark import get_spark

p = argparse.ArgumentParser()
p.add_argument("--start", required=True)   # 2024-01-01
p.add_argument("--end", required=True)     # 2024-12-31
a = p.parse_args()

r = requests.get("https://archive-api.open-meteo.com/v1/archive", timeout=60, params={
    "latitude": 40.71, "longitude": -74.01, "start_date": a.start, "end_date": a.end,
    "hourly": "temperature_2m,precipitation", "timezone": "America/New_York"})
r.raise_for_status()
pdf = pd.DataFrame(r.json()["hourly"])

spark = get_spark("weather_bronze")
df = (spark.createDataFrame(pdf)
      .withColumn("ts", F.to_timestamp("time")).drop("time")
      .withColumn("_ingested_at", F.current_timestamp()))
spark.sql("CREATE NAMESPACE IF NOT EXISTS lake.bronze")
df.writeTo("lake.bronze.weather_hourly").createOrReplace()
```

### Silver and gold

`jobs/transform/trips_silver.py` cleans one month and partitions by pickup day:

```python
import argparse, sys
sys.path.insert(0, "/opt/jobs")
from pyspark.sql import functions as F
from pensieve.lake.spark import get_spark

p = argparse.ArgumentParser(); p.add_argument("--month", required=True); a = p.parse_args()
spark = get_spark(f"trips_silver:{a.month}")

spark.sql("""CREATE TABLE IF NOT EXISTS lake.silver.trips (
  vendor_id INT, pickup_ts TIMESTAMP, dropoff_ts TIMESTAMP, passenger_count INT,
  trip_distance DOUBLE, pu_zone_id INT, do_zone_id INT, payment_type INT,
  fare_amount DOUBLE, tip_amount DOUBLE, total_amount DOUBLE)
  USING iceberg PARTITIONED BY (days(pickup_ts))""")

trips = (spark.table("lake.bronze.tlc_yellow").where(F.col("_ingest_month") == a.month)
  .where(F.date_format("tpep_pickup_datetime", "yyyy-MM") == a.month)   # drop mis-dated rows
  .where("trip_distance > 0 AND total_amount >= 0 AND tpep_dropoff_datetime > tpep_pickup_datetime")
  .select(F.col("vendorid").alias("vendor_id"), F.col("tpep_pickup_datetime").alias("pickup_ts"),
          F.col("tpep_dropoff_datetime").alias("dropoff_ts"), F.col("passenger_count").cast("int"),
          "trip_distance", F.col("pulocationid").alias("pu_zone_id"),
          F.col("dolocationid").alias("do_zone_id"), F.col("payment_type").cast("int"),
          "fare_amount", "tip_amount", "total_amount")
  .dropDuplicates())
trips.writeTo("lake.silver.trips").overwritePartitions()   # replaces that month's days only
```

`jobs/transform/daily_zone_weather.py` rebuilds gold:

```python
import sys
sys.path.insert(0, "/opt/jobs")
from pensieve.lake.spark import get_spark

spark = get_spark("gold_daily_zone_weather")
spark.sql("CREATE NAMESPACE IF NOT EXISTS lake.gold")
spark.sql("""
CREATE OR REPLACE TABLE lake.gold.daily_zone_weather USING iceberg PARTITIONED BY (months(day)) AS
WITH w AS (
  SELECT to_date(ts) AS day, avg(temperature_2m) AS avg_temp_c, sum(precipitation) AS precip_mm
  FROM lake.bronze.weather_hourly GROUP BY 1)
SELECT to_date(t.pickup_ts) AS day, z.borough, count(*) AS trips,
       round(avg(t.trip_distance), 2) AS avg_miles, round(sum(t.total_amount), 2) AS revenue,
       round(avg(t.tip_amount / nullif(t.fare_amount, 0)), 3) AS avg_tip_ratio,
       first(w.avg_temp_c) AS avg_temp_c, first(w.precip_mm) AS precip_mm
FROM lake.silver.trips t
JOIN lake.silver.zones z ON t.pu_zone_id = z.location_id
LEFT JOIN w ON w.day = to_date(t.pickup_ts)
GROUP BY 1, 2""")
```

### Verify

```bash
scripts/download_tlc.sh $(echo 2024-{01..06})
make up && scripts/seed_source_db.sh
for m in 2024-01 2024-02 2024-03 2024-04 2024-05 2024-06; do
  make submit JOB=ingest/tlc_bronze.py ARGS="--month $m"
  make submit JOB=transform/trips_silver.py ARGS="--month $m"
done
make submit JOB=ingest/zones_from_pg.py
make submit JOB=ingest/weather_bronze.py ARGS="--start 2024-01-01 --end 2024-06-30"
make submit JOB=transform/daily_zone_weather.py
```

In `make sql`: `SELECT * FROM lake.gold.daily_zone_weather ORDER BY precip_mm DESC LIMIT 10;` Then rerun one month's bronze and silver jobs and confirm the row counts don't change.

### Commit

`feat(etl): bronze/silver/gold from files, Postgres JDBC and REST API`

### Explore (optional)

- **Read the landing zone from S3.** Upload the Parquet files to `s3://landing/` and read them with `s3a://`. You'll need `hadoop-aws` plus the AWS SDK bundle that matches the Hadoop version inside Spark. Find it with `ls /opt/spark/jars | grep hadoop-client` and match exactly; a mismatch is the classic S3A failure.
- **Parallel JDBC reads.** Generate a million-row table in `source-db`, then read it with `partitionColumn`, `lowerBound`, `upperBound` and `numPartitions`. Watch connections in `pg_stat_activity`. You'll see why teams read replicas, not primaries.
- **SCD Type 2.** Change some zones in Postgres. Replace the `MERGE` with one that closes the old row (`valid_to`) and inserts a new one, so history survives.
- **Make it big.** Download several years of the much larger `fhvhv_tripdata` files, or generate synthetic data with `spark.range(10_000_000_000)`. Find the point where shuffle starts spilling to the scratch SSD.
- **Streaming.** Add Kafka (or Redpanda) to compose and write a Structured Streaming job into a bronze Iceberg table. Look at how many small files it creates; Phase 8 cleans them up.

## Phase 6: Trino for interactive SQL

Goal: query every table in the `lake` catalog from Trino, with no copy of the data. Athena is managed Trino, so this is the closest local match, and it proves the catalog really is shared between engines.

### Build

`conf/trino/catalog/lake.properties`. Trino substitutes `${ENV:...}` from its environment, so no secrets land in Git:

```properties
connector.name=iceberg
iceberg.catalog.type=rest
iceberg.rest-catalog.uri=http://polaris:8181/api/catalog
iceberg.rest-catalog.warehouse=lake
iceberg.rest-catalog.security=OAUTH2
iceberg.rest-catalog.oauth2.credential=${ENV:POLARIS_CLIENT_ID}:${ENV:POLARIS_CLIENT_SECRET}
iceberg.rest-catalog.oauth2.scope=PRINCIPAL_ROLE:ALL
iceberg.rest-catalog.oauth2.server-uri=http://polaris:8181/api/catalog/v1/oauth/tokens
fs.s3.enabled=true
s3.endpoint=http://seaweedfs:8333
s3.region=us-east-1
s3.path-style-access=true
s3.aws-access-key=${ENV:S3_ACCESS_KEY}
s3.aws-secret-key=${ENV:S3_SECRET_KEY}
```

These properties follow [Polaris's Trino guide](https://polaris.apache.org/guides/trino/). Older Trino releases call the S3 switch `fs.native-s3.enabled`; if Trino rejects `fs.s3.enabled`, check which one your tag expects.

`compose.query.yaml`:

```yaml
services:
  trino:
    image: trinodb/trino:${TRINO_TAG}
    ports: ["8088:8080"]                 # 8080 is taken by the Spark master UI
    environment:
      POLARIS_CLIENT_ID: ${POLARIS_CLIENT_ID}
      POLARIS_CLIENT_SECRET: ${POLARIS_CLIENT_SECRET}
      S3_ACCESS_KEY: ${S3_ACCESS_KEY}
      S3_SECRET_KEY: ${S3_SECRET_KEY}
    volumes:
      - ./conf/trino/catalog:/etc/trino/catalog:ro
    mem_limit: 64g                       # the image sizes its heap from this limit
    networks: [lake]

networks:
  lake:
    external: true
```

Add a Makefile target:

```makefile
trino:
	$(DC) exec trino trino --catalog lake
```

### Verify

```sql
SHOW SCHEMAS;
SELECT borough, sum(trips) AS trips, round(avg(avg_tip_ratio), 3) AS tip_ratio
FROM gold.daily_zone_weather
WHERE precip_mm > 5
GROUP BY borough ORDER BY trips DESC;
```

Open `http://pensive:8088` to see the query in Trino's UI, then run the same query in `make sql` and compare.

### Commit

`feat(query): Trino on the Polaris catalog`

### Explore (optional)

- **Two engines, one table.** Insert a row from Trino into `lake.demo.hello`, then read it in Spark. Then do the opposite. Both commit through Polaris, so each sees the other's snapshots.
- **Why Trino feels faster interactively.** Run the same aggregation on `silver.trips` in both engines. Trino pipelines data between stages in memory and starts in milliseconds; Spark plans stages, launches tasks and writes shuffle files. Then run a much bigger join and see which one copes better as data outgrows memory.
- **Metadata tables.** Query `SELECT * FROM silver."trips$files" LIMIT 20` and `"trips$snapshots"`. Count files per partition. Write down the average file size; Phase 8 makes this better.
- **Read the plan.** Run `EXPLAIN ANALYZE` with and without a filter on `pickup_ts`. See how many files Trino skips thanks to partition pruning and min/max statistics.
- **DuckDB from the host.** Attach the `lake` catalog from DuckDB's Iceberg extension on your laptop. It fails to read data at first, because Polaris tells clients the endpoint `http://seaweedfs:8333`, which only resolves inside Docker. Fix it with Polaris's `endpoint` and `endpointInternal` split ([docs](https://polaris.apache.org/releases/1.1.0/getting-started/minio)). This is a real-world networking lesson.

## Phase 7: Observability and tuning

Goal: Spark metrics in your existing Prometheus and Grafana stack, a NUMA-pinned cluster shape, and a benchmark log that shows which settings actually help on this machine.

### Build: metrics

`conf/spark/metrics.properties` turns on Spark's built-in Prometheus endpoints:

```properties
*.sink.prometheusServlet.class=org.apache.spark.metrics.sink.PrometheusServlet
*.sink.prometheusServlet.path=/metrics/prometheus
master.sink.prometheusServlet.path=/metrics/master/prometheus
applications.sink.prometheusServlet.path=/metrics/applications/prometheus
```

Add to the defaults template:

```properties
spark.ui.prometheus.enabled           true
spark.metrics.appStatusSource.enabled true
spark.io.compression.codec            zstd
```

Scrape targets for Alloy (all published on the host):

| Target | Path | What you get |
| --- | --- | --- |
| Master, port 8080 | `/metrics/master/prometheus` and `/metrics/applications/prometheus` | Workers alive, cores and memory in use, running apps |
| Workers, ports 8081 and up | `/metrics/prometheus` | Free cores and memory per worker |
| Spark Connect driver, port 4040 | `/metrics/executors/prometheus` | Per-executor GC time, shuffle read and write, spill |
| Containers | cAdvisor, or Alloy's built-in cAdvisor exporter | CPU and memory per container versus its limit |
| Scratch SSD | node exporter and smartctl exporter | Write rate, free space, wear percentage |

A useful first dashboard has six panels: cores in use versus total, executor memory used, shuffle bytes written per minute, disk spill per minute, GC time as a share of task time, and scratch free space. The last two are your early warnings.

### Build: tuning starting points for this box

| Setting | Start at | Why |
| --- | --- | --- |
| Spark's share of the box | About 80 of 112 threads, 640 GB | Leaves room for LLM serving, databases and the OS |
| `spark.executor.cores` | 5 | Enough parallel tasks per JVM without heavy lock contention |
| `spark.executor.memory` | 40g, plus `spark.executor.memoryOverhead` 6g | Heaps above about 64 GB suffer longer GC pauses |
| `spark.sql.shuffle.partitions` | 2 to 3 times total executor cores | Adaptive execution merges small partitions afterwards |
| `spark.sql.adaptive.advisoryPartitionSizeInBytes` | 128m | Target size after merging |
| `spark.sql.files.maxPartitionBytes` | 256m | Fewer, larger read tasks on big Parquet files |
| Garbage collector | G1 (the default) | Try generational ZGC on a Java 21 image for very large heaps |

Spark counts logical CPUs, so a worker with 28 cores on this machine uses 14 physical cores and their SMT siblings.

### Build: a NUMA-pinned shape

With NPS4, each NUMA node has 14 physical cores (28 threads) and a quarter of the RAM. A JVM whose threads and memory stay on one node avoids slower cross-node memory traffic. Executors inherit the worker's NUMA policy, so pinning the worker is enough.

`scripts/gen_numa_workers.py` writes one worker service per node:

```python
#!/usr/bin/env python3
"""Generate compose.numa.yaml: one Spark worker pinned to each NUMA node."""
import sys

nodes = int(sys.argv[1]) if len(sys.argv) > 1 else 4
out = ["services:"]
for n in range(nodes):
    out += [
        f"  spark-worker-n{n}:",
        "    extends: { file: compose.spark.yaml, service: spark-worker }",
        "    profiles: [numa]",
        "    cap_add: [SYS_NICE]    # Docker's seccomp profile blocks NUMA memory binding without it",
        f'    ports: ["{8101 + n}:8081"]',
        "    command: >",
        f"      numactl --cpunodebind={n} --membind={n}",
        "      /opt/spark/bin/spark-class org.apache.spark.deploy.worker.Worker",
        "      --cores ${SPARK_WORKER_CORES} --memory ${SPARK_WORKER_MEMORY} spark://spark-master:7077",
    ]
out += ["networks:", "  lake:", "    external: true"]
open("platform/compose.numa.yaml", "w").write("\n".join(out) + "\n")
```

`shapes/numa.env` uses `SPARK_WORKERS=0` (no unpinned workers), 20 cores and 160g per worker, and the Makefile gets a target that adds the profile:

```makefile
up-numa: net conf
	python3 scripts/gen_numa_workers.py 4
	STACKS="$(CORE) numa" $(DC) --profile numa up -d --build --scale spark-worker=0
```

Check the pinning while a job runs: `numastat -p $(pgrep -f CoarseGrainedExecutorBackend | head -1)` should show memory almost entirely on one node.

### Verify

- Grafana shows cores in use rising and falling as jobs run.
- `make up-numa SHAPE=numa` brings up 4 workers, one per node, in the master UI.
- `docs/benchmarks.md` has a table with at least one job timed under two settings.

### Commit

`feat(ops): Prometheus metrics, tuning defaults, NUMA-pinned worker shape`

### Explore (optional)

Run each experiment on the same job (the six-month silver rebuild works well) and log the results in `docs/benchmarks.md`.

- **Executor size sweep.** Try 2 cores and 16g, 5 cores and 40g, then 10 cores and 90g. Watch GC time in the executor tab.
- **Pinned versus unpinned.** Same total cores, `make up` versus `make up-numa`. The difference shows most on shuffle-heavy joins.
- **Force a spill.** Halve executor memory and run a wide aggregation. Find "Spill (Disk)" in the stage page, and the matching write burst on the scratch SSD in Grafana.
- **Skewed joins.** Join trips to a table where one zone holds most rows. Compare with `spark.sql.adaptive.skewJoin.enabled` set to `false` and `true`.
- **Fake a real network.** Add `cap_add: [NET_ADMIN]` to the workers and run `tc qdisc add dev eth0 root tbf rate 10gbit burst 1mbit latency 50ms` inside each. Now shuffle costs what it would between real machines.
- **Columnar engines.** Check whether Apache DataFusion Comet or Gluten with Velox supports Spark 4.1 yet. If one does, add its jar and compare a TPC-H-style query. This is where your Arrow question pays off.

## Phase 8: Iceberg table maintenance

Goal: keep tables fast and storage bounded. Every write adds files and a snapshot. Without maintenance you get thousands of small files, slow query planning and storage that only grows. On AWS, S3 Tables or Glue automate some of this; here you'll do it yourself and understand it.

### Build

`jobs/maintain/optimize_table.py`:

```python
import argparse, sys
from datetime import datetime, timedelta, timezone
sys.path.insert(0, "/opt/jobs")
from pensieve.lake.spark import get_spark

p = argparse.ArgumentParser()
p.add_argument("--table", required=True)               # e.g. silver.trips
p.add_argument("--keep-days", type=int, default=7)
p.add_argument("--target-mb", type=int, default=512)
a = p.parse_args()

spark = get_spark(f"optimize:{a.table}")
cutoff = (datetime.now(timezone.utc) - timedelta(days=a.keep_days)).strftime("%Y-%m-%d %H:%M:%S")
t = f"'{a.table}'"

# 1. Merge small data files into larger ones.
spark.sql(f"CALL lake.system.rewrite_data_files(table => {t}, "
          f"options => map('target-file-size-bytes', '{a.target_mb * 1024 * 1024}'))").show()
# 2. Merge small manifests so planning reads fewer metadata files.
spark.sql(f"CALL lake.system.rewrite_manifests({t})").show()
# 3. Drop old snapshots (and the files only they reference). Time travel ends here.
spark.sql(f"CALL lake.system.expire_snapshots(table => {t}, "
          f"older_than => TIMESTAMP '{cutoff}', retain_last => 5)").show()
# 4. Delete files no snapshot references, e.g. from failed writes.
spark.sql(f"CALL lake.system.remove_orphan_files(table => {t}, "
          f"older_than => TIMESTAMP '{cutoff}')").show()
```

The order matters. Compaction creates new files and a new snapshot, and expiry then frees the old small files. Orphan cleanup always uses an age cutoff, so it can't delete files a running job is still writing.

### Learn by doing: snapshots, schema and partitions

Run these in `make sql`, one block at a time, and look at the results between steps. Work on a copy (CREATE TABLE lake.silver.trips\_lab AS SELECT \* FROM lake.silver.trips) and swap trips\_lab into the examples, because renaming a column on the real table breaks the jobs that use it.

**Time travel and rollback:**

```sql
SELECT snapshot_id, committed_at, operation FROM lake.silver.trips.snapshots;
SELECT count(*) FROM lake.silver.trips VERSION AS OF <older_snapshot_id>;
CALL lake.system.rollback_to_snapshot('silver.trips', <older_snapshot_id>);
```

**Schema evolution** (metadata only; no data files are rewritten):

```sql
ALTER TABLE lake.silver.trips ADD COLUMN tolls_amount DOUBLE;
ALTER TABLE lake.silver.trips RENAME COLUMN tip_amount TO tip_usd;
```

**Partition evolution** (old files keep the old layout; new writes use the new one):

```sql
ALTER TABLE lake.silver.trips ADD PARTITION FIELD bucket(16, pu_zone_id);
SELECT partition, file_count FROM lake.silver.trips.partitions LIMIT 20;
```

**Write-audit-publish with a branch**: write to a branch, check it, then publish:

```sql
ALTER TABLE lake.silver.trips CREATE BRANCH audit;
SET spark.wap.branch = audit;
-- run the silver job for a new month in this session, or insert test rows
SELECT count(*) FROM lake.silver.trips VERSION AS OF 'audit';
SELECT count(*) FROM lake.silver.trips;                  -- main is unchanged
CALL lake.system.fast_forward('silver.trips', 'main', 'audit');
```

`SET spark.wap.branch` needs the table property `write.wap.enabled=true`. Set it with `ALTER TABLE ... SET TBLPROPERTIES` first.

### Verify

- Before and after `make submit JOB=maintain/optimize_table.py ARGS="--table silver.trips"`, count rows in `lake.silver.trips.files`. The count should drop and the average file size should rise.
- `snapshots` shows only recent entries, plus the ones you chose to keep.

### Commit

`feat(maintenance): compaction, snapshot expiry, orphan cleanup job`

### Explore (optional)

- **Make the small-file problem.** Run 500 single-row inserts into a copy of a table. Time `EXPLAIN` and a simple count before and after compaction. Planning time is the hidden cost.
- **Sort for skipping.** Rewrite with `strategy => 'sort'` and `sort_order => 'zorder(pu_zone_id, pickup_ts)'`. Then compare files scanned for a filter on both columns, in Trino's `EXPLAIN ANALYZE`.
- **Copy-on-write versus merge-on-read.** Set `write.delete.mode` to each value, delete 1% of rows, and compare write time, read time and the delete files that appear.
- **What expiry really frees.** Measure bucket size with `make s3 ARGS="ls --recursive --summarize s3://warehouse/lake/"` before and after expiring snapshots.
- **Lose the catalog on purpose.** Stop Polaris, drop and recreate a test catalog, then bring a table back with `CALL lake.system.register_table(...)` pointing at its last metadata file. That's your disaster-recovery drill.

## Phase 9: Orchestrating the ETL with Airflow 3

Goal: one monthly DAG that downloads data, runs bronze, silver and gold, then maintains the tables, with retries and backfills. Each Spark step runs in its own throwaway driver container from your Spark image, the same way `make submit` does.

Why containers instead of `SparkSubmitOperator`? That operator runs the driver inside the Airflow container. Then Airflow needs Java, the same PySpark version, and the same Python minor version as your workers, or Python tasks fail. Launching the Spark image keeps versions identical by construction. It's also how `KubernetesPodOperator` and EMR steps work.

### Build

`docker/airflow/Dockerfile`:

```dockerfile
ARG AIRFLOW_VERSION
FROM apache/airflow:${AIRFLOW_VERSION}
ARG AIRFLOW_VERSION
COPY requirements.in /tmp/requirements.in       # apache-airflow-providers-docker, requests
# Airflow publishes a constraints file per release that pins every dependency.
# Use the one matching the image's Python version (3.12 here).
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /tmp/requirements.in \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-3.12.txt"
```

`compose.airflow.yaml`:

```yaml
services:
  airflow-db:
    image: postgres:${POSTGRES_TAG}
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: ${AIRFLOW_DB_PASSWORD}
      POSTGRES_DB: airflow
    volumes:
      - ${PG_ROOT}/airflow-db:/var/lib/postgresql/data
    networks: [lake]

  airflow:
    build: { context: ./docker/airflow, args: [AIRFLOW_VERSION] }
    command: standalone          # api-server, scheduler, dag-processor, triggerer in one; dev only
    environment:
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:${AIRFLOW_DB_PASSWORD}@airflow-db/airflow
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
      AIRFLOW__CORE__LOAD_EXAMPLES: "False"
      LAKEHOUSE_REPO: ${PWD}       # host path of this repo, for bind mounts in task containers
      DATA_ROOT: ${DATA_ROOT}
      SOURCE_DB_PASSWORD: ${SOURCE_DB_PASSWORD}
    volumes:
      - ../pipelines/dags:/opt/airflow/dags
      - ${DATA_ROOT}/landing:/data/landing
      - /var/run/docker.sock:/var/run/docker.sock
    group_add: ["${DOCKER_GID}"]  # lets Airflow use the Docker socket
    ports: ["8089:8080"]
    depends_on: [airflow-db]
    networks: [lake]

networks:
  lake:
    external: true
```

Add `DOCKER_GID` to `.env` (find it with `getent group docker | cut -d: -f3`). Mounting the Docker socket gives Airflow control of Docker on the host. That's acceptable on a homelab box on its own network, not on shared infrastructure. Standalone mode prints the admin password in the container log.

`dags/tlc_monthly.py`:

```python
import os
from datetime import datetime, timedelta
from airflow.sdk import DAG, task
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

REPO, DATA = os.environ["LAKEHOUSE_REPO"], os.environ["DATA_ROOT"]
MONTH = "{{ data_interval_start.strftime('%Y-%m') }}"

def spark_job(task_id, job, args="", cores=32):
    return DockerOperator(
        task_id=task_id,
        image="lakehouse/spark:4.1.2",
        command=f"lake-submit --conf spark.cores.max={cores} /opt/jobs/{job} {args}",
        network_mode="lake",
        mounts=[
            Mount(source=f"{REPO}/platform/conf/spark", target="/opt/spark/conf", type="bind", read_only=True),
            Mount(source=f"{REPO}/pipelines/jobs", target="/opt/jobs", type="bind", read_only=True),
            Mount(source=f"{REPO}/libs", target="/opt/libs", type="bind", read_only=True),
            Mount(source=f"{DATA}/landing", target="/data/landing", type="bind"),
            Mount(source=f"{DATA}/spark-events", target="/opt/spark/events", type="bind"),
        ],
        environment={"SOURCE_DB_PASSWORD": os.environ["SOURCE_DB_PASSWORD"]},
        docker_url="unix://var/run/docker.sock",
        auto_remove="success",
        mount_tmp_dir=False,
    )

with DAG(
    dag_id="tlc_monthly",
    start_date=datetime(2024, 1, 1),
    schedule="@monthly",
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
) as dag:

    @task
    def download(month: str):
        import requests, pathlib
        dest = pathlib.Path(f"/data/landing/tlc/yellow/yellow_tripdata_{month}.parquet")
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{dest.name}"
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)

    bronze = spark_job("bronze", "ingest/tlc_bronze.py", f"--month {MONTH}")
    zones = spark_job("zones", "ingest/zones_from_pg.py", cores=8)
    silver = spark_job("silver", "transform/trips_silver.py", f"--month {MONTH}")
    gold = spark_job("gold", "transform/daily_zone_weather.py")
    optimize = spark_job("optimize", "maintain/optimize_table.py", "--table silver.trips", cores=16)

    download(MONTH) >> bronze >> silver
    [silver, zones] >> gold >> optimize
```

The weather API job is left out on purpose; adding it is the first exercise. `command` is a templated field, so `{{ data_interval_start }}` becomes each run's month.

### Verify

- Open `http://pensive:8089`, find `tlc_monthly`, and see the graph. Start the stack first with make up-with S=airflow.
- Backfill a few months: `./dc exec airflow airflow backfill create --dag-id tlc_monthly --from-date 2024-07-01 --to-date 2024-09-01`.
- While it runs, `docker ps` shows short-lived `lakehouse/spark` driver containers, and the Spark master UI shows each step as an application.
- When it finishes, the gold table in Trino includes the new months.

### Commit

`feat(orchestration): Airflow 3 with containerized Spark tasks and monthly DAG`

### Explore (optional)

- **Add the weather step.** Give the weather job `--start` and `--end` from the run's data interval, and make gold wait for it.
- **Retries are only safe when jobs are idempotent.** Kill a silver driver container mid-run. Airflow retries it, and because the job overwrites partitions, row counts stay correct. Then imagine the job used plain `append()` instead.
- **Data-aware scheduling.** Split gold into its own DAG and trigger it with an Airflow Asset that the silver task updates, instead of a fixed schedule.
- **Feel the version-matching problem.** Try `SparkSubmitOperator` with PySpark installed in the Airflow image. Note what breaks when the Python minor versions differ from the Spark image's, and why the container approach avoids it.
- **A nightly maintenance DAG.** Loop over every table in `SHOW TABLES IN lake.silver` and run the optimize job for each, with a pool that limits it to one at a time.

## Phase 10: Hardening and next steps

Goal: turn the working lab into something you'd trust to run unattended, and know the path to Kubernetes and AWS. Pick these in any order.

### Hardening checklist

- [ ] **One principal per service.** Replace the root Polaris credential with `spark_etl` (write to bronze, silver and gold), `trino_ro` (read only) and `airflow` principals, each with the narrowest roles that work.
- [ ] **Secrets out of plain text.** Move `.env` values into Docker secrets, or keep an encrypted file in the repo with `sops` and `age`. Rotate the S3 keys once to prove nothing is hardcoded.
- [ ] **Remove dev shortcuts.** Drop `POLARIS_READINESS_IGNORE_SEVERE_ISSUES` and fix what Polaris then reports. Replace `airflow standalone` with separate scheduler, API server and DAG-processor services.
- [ ] **Network boundaries.** Publish only the UIs you use, keep the stack on your homelab VLAN, and put a reverse proxy with TLS (Caddy or Traefik) in front of the web UIs.
- [ ] **Back up the catalog.** Run a nightly `pg_dump` of `catalog-db`, and take ZFS snapshots of the lakehouse dataset. Restore both once, using the register-table drill from Phase 8.
- [ ] **Pin everything.** Replace `latest` tags with the exact versions in versions.env (Tutorial 0), add image digests, run make versions-check in CI, and keep an upgrade log in `docs/`.
- [ ] **Alerts.** In Alertmanager: a worker missing, scratch disk above 80%, a DAG failing twice in a row, and SSD wear past a threshold.

### Continuous integration

A GitHub Actions workflow on every push, with path filters so each job runs only when its area changes (for example, platform/docker/spark/\*\* rebuilds only the Spark image, dl/\*\* runs only the deep-learning checks, and docs-only commits skip builds):

1. `ruff` on `jobs/` and `dags/`.
2. `pytest` on the pure transform functions, with a local `SparkSession` and tiny fixtures.
3. `./dc config` to validate every compose file.
4. Build the Spark image, so a broken Dockerfile fails fast.
5. Import every DAG file in an Airflow container to catch syntax and import errors.

### Where to go next

| Next step | What changes | What you learn |
| --- | --- | --- |
| Kubernetes (k3s on Proxmox) | The Spark Kubernetes Operator replaces standalone mode; Airflow uses `KubernetesPodOperator` | Scheduling, namespaces, resource quotas, the way most companies run Spark today |
| A second machine | Add workers on another node, joined to the same master | Real network shuffle, data locality, the limits of 2.5 GbE |
| AWS | Swap endpoints and credentials: S3 for SeaweedFS, Glue's Iceberg REST endpoint (with SigV4 signing) for Polaris, EMR Serverless for Spark, Athena for Trino, MWAA for Airflow | Because jobs use only table names, they should run unchanged. Check current AWS docs for the exact catalog settings |
| Streaming | Kafka or Redpanda plus Structured Streaming into bronze | Exactly-once writes, checkpoints, small-file pressure |
| dbt | `dbt-spark` or `dbt-trino` for the silver-to-gold SQL | Tested, documented SQL models, which pairs well with Airflow |

When you finish, you'll have deployed, monitored, tuned and orchestrated every layer of a lakehouse yourself, which is most of what a data platform team does.

## Sources

- [Apache Spark 4.2.0 release notes](https://spark.apache.org/releases/spark-release-4-2-0.html)
- [Spark official Docker image tags](https://hub.docker.com/r/amd64/spark/)
- [Iceberg multi-engine support (Spark runtime versions)](https://iceberg.apache.org/multi-engine-support/)
- [Iceberg issue 16856: Spark 4.1 runtime and Flink classes](https://github.com/apache/iceberg/issues/16856)
- [MinIO community edition maintenance mode](https://dev.to/alexandrev/minio-in-maintenance-mode-what-it-means-for-the-community-edition-oem-users-and-open-source-3o4j)
- [SeaweedFS with DuckDB (mini mode and S3 config)](https://duckdb.org/docs/current/guides/network_cloud_storage/seaweedfs_import)
- [Apache Polaris graduation to a top-level project](https://www.fivetran.com/blog/the-impact-of-apache-polaris-graduating-to-top-level-apache-project)
- [Polaris relational JDBC metastore](https://polaris.apache.org/in-dev/unreleased/metastores/relational-jdbc/)
- [Polaris S3 and S3-compatible storage configuration](https://polaris.apache.org/in-dev/unreleased/configuration/configuring-polaris-for-production/configuring-aws-s3-cloud-storage-specific/)
- [Polaris endpoint versus endpointInternal](https://polaris.apache.org/releases/1.1.0/getting-started/minio)
- [Polaris with Trino](https://polaris.apache.org/guides/trino/)
- [Airflow versions and release notes](https://www.getorchestra.io/guides/airflow-version)

