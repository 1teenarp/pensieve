# Streaming and MLOps on the Lakehouse

Oct 29, 2026 · @Praneet

## Start here

This tutorial picks up where [Build a Spark Lakehouse on One Box](https://claude.ai/artifact/X9BMca4uV6qkeMSY6L3JFX) ends and adds two capabilities to the same repo. **Part A (Phases 11–15)** adds real-time data: Kafka as the publish/subscribe backbone, and Spark Structured Streaming writing live Iceberg tables and alerts. **Part B (Phases 16–21)** adds an ML platform on your two GPUs: tracked experiments, repeatable training jobs, LoRA fine-tuning, a model registry, and serving. That covers most of what SageMaker offers, at lab scale.

Phase numbers continue from tutorial 1, so your Git history keeps telling one story. Each phase has the same four parts: **Build**, **Verify**, **Commit** and an optional **Explore**.

**Before you start**, tutorial 1 should be complete through Phase 9. You need the `lake` network, SeaweedFS, Polaris, the Spark image, `make submit`, Trino and Airflow working, plus `silver.trips` and `silver.zones` loaded.

&#91;embedded content: tutorial 2 architecture · streaming path and ML loop\]

Events flow from the producer through Kafka into Spark, which writes tables and publishes alerts and ETAs back to Kafka. The ML loop reads those tables, trains on GPU 1, registers models in MLflow, and serves them, with predictions written back to the lakehouse.

### New pinned versions (as of Oct 2026)

| Component | Version | Role |
| --- | --- | --- |
| [Apache Kafka](https://kafka.incubator.apache.org/community/downloads/) | 4.3.0, image `apache/kafka:4.3.0` (KRaft only) | Event log, pub/sub |
| Spark Kafka connector | `spark-sql-kafka-0-10_2.13:4.1.2` | Spark reads and writes Kafka |
| [MLflow](https://mlflow.org/docs/latest/ml/tracking/server) | Latest 3.x, `ghcr.io/mlflow/mlflow` | Experiments, model registry |
| [PyTorch](https://discuss.pytorch.org/t/rtx-5050-sm-120-support-timeline-when-in-stable-pytorch/224288/8) | 2.7 or newer, CUDA 12.8 wheels (`cu128`) | Training on Blackwell GPUs |
| Transformers, PEFT, TRL | Latest, pinned in a lockfile | LoRA and QLoRA fine-tuning |
| vLLM | Pin a recent `vllm/vllm-openai` tag | Serving the LLM with LoRA adapters |
| XGBoost, Optuna, Evidently | Latest, pinned in a lockfile | Tabular model, sweeps, drift reports |

Your RTX Pro 5000 cards are Blackwell (compute capability 12.0, [72 GB each](https://gpus.axiomgaming.net/gpu/rtx-pro-5000-blackwell)). Any PyTorch wheel built for CUDA 12.8 or newer supports them; older CUDA builds fail with a "not compatible" error.

### What maps to what on AWS

| On your box | On AWS |
| --- | --- |
| Kafka | Amazon MSK |
| Spark Structured Streaming | EMR or Glue streaming jobs |
| MLflow tracking | SageMaker managed MLflow, or SageMaker Experiments |
| `make train` in a GPU container | SageMaker Training Jobs |
| Optuna sweeps | SageMaker automatic model tuning |
| MLflow model registry | SageMaker Model Registry |
| MLflow serving and vLLM | SageMaker real-time endpoints |
| Spark batch scoring | SageMaker Batch Transform |
| Airflow ML DAG | SageMaker Pipelines |
| Evidently drift reports | SageMaker Model Monitor |
| Feature tables in Iceberg | SageMaker Feature Store (offline store) |

### What this adds to the repo

```text
pensieve/
  platform/
    compose.stream.yaml     # Kafka, Kafka UI, exporter; stream services behind the stream-jobs profile
    compose.ml.yaml         # MLflow, trainer; serving behind the serve profile
    docker/tools/           # producer and consumer image (requirements.lock)
    docker/trainer/         # GPU training image (installs ml/uv.lock)
    docker/mlflow/
  pipelines/
    contracts/              # event schemas (trip_event.v1.json)
    tools/                  # producers, consumers, ask_lake.py
    jobs/stream/            # Structured Streaming jobs
    jobs/ml/                # feature building and batch scoring (Spark)
    dags/ml_*.py            # retraining pipelines
  ml/                       # training code, configs, pyproject.toml + uv.lock
  libs/pensieve/lake/catalog.py   # shared Iceberg access from Python (Phase 17)
  make/stream.mk            # targets for Part A
  make/ml.mk                # targets for Part B
```

Tutorial 0's repo conventions apply throughout: Compose paths are relative to `platform/`, versions live in `versions.env`, and every Compose command goes through `./dc`. Start this tutorial's stacks with `make up-stream` and `make up-ml`. Tag each phase's commit, for example `git tag t2-p13`.

**Prerequisites:** Tutorial 0 (Set up Pensieve: Host, Repo and Working Agreement) and Tutorial 1 tagged through `t1-p10`. Each phase follows Tutorial 0's loop: one branch, PR, journal entry and tag. Hand a phase to the agent with `Read AGENTS.md, then do phase t2-p11 from docs/tutorials/2-streaming-mlops.md.` GPU drivers and the NVIDIA container toolkit are owner-only steps; the agent lists them under "Needs owner action" and waits.

## Part A: Streaming

## Phase 11: Kafka as the publish/subscribe backbone

Goal: a single-node Kafka 4.3 broker in KRaft mode, a web UI, and three topics that show the three ways you'll use it: an event stream, an alert channel, and a compacted "latest state" topic.

The core ideas, in one place:

- A **topic** is a named channel. Producers append records to it; it keeps them for a retention period whether or not anyone reads them.
- A topic is split into **partitions**. Records with the same key always land in the same partition, so they stay in order. Partitions are the unit of parallelism.
- A **consumer group** shares the work: each partition goes to exactly one member. That's queue behavior.
- **Several groups** on one topic each get every record. That's publish/subscribe fan-out.
- A **share group** (production-ready since [Kafka 4.2](https://kafka.apache.org/blog/2026/02/17/apache-kafka-4.2.0-release-announcement/)) lets many consumers take records from the same partition with per-record acknowledgement, like a classic work queue.

### Build

Generate a cluster ID once and add it to `.env` as `KAFKA_CLUSTER_ID`:

```bash
docker run --rm apache/kafka:4.3.0 /opt/kafka/bin/kafka-storage.sh random-uuid
sudo mkdir -p /srv/pensieve/kafka && sudo chown 1000:1000 /srv/pensieve/kafka   # 1000 = appuser in the image
```

Also add `KAFKA_DATA=/srv/pensieve/kafka`. Kafka writes sequentially, but fsyncs and many partitions are kinder to SSD than to the 18 TB hard drive.

`compose.stream.yaml`:

```yaml
services:
  kafka:
    image: apache/kafka:${KAFKA_TAG}
    environment:
      CLUSTER_ID: ${KAFKA_CLUSTER_ID}
      KAFKA_NODE_ID: 1
      KAFKA_PROCESS_ROLES: broker,controller
      KAFKA_LISTENERS: INTERNAL://:9092,EXTERNAL://:29092,CONTROLLER://:9093
      KAFKA_ADVERTISED_LISTENERS: INTERNAL://kafka:9092,EXTERNAL://${HOST_NAME}:29092
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: INTERNAL:PLAINTEXT,EXTERNAL:PLAINTEXT,CONTROLLER:PLAINTEXT
      KAFKA_INTER_BROKER_LISTENER_NAME: INTERNAL
      KAFKA_CONTROLLER_LISTENER_NAMES: CONTROLLER
      KAFKA_CONTROLLER_QUORUM_VOTERS: 1@kafka:9093
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 1
      KAFKA_SHARE_COORDINATOR_STATE_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_SHARE_COORDINATOR_STATE_TOPIC_MIN_ISR: 1
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: "false"
      KAFKA_LOG_DIRS: /var/lib/kafka/data
    volumes:
      - ${KAFKA_DATA}:/var/lib/kafka/data
    ports: ["29092:29092"]          # for clients outside Docker
    networks: [lake]

  kafka-ui:
    image: ghcr.io/kafbat/kafka-ui:${KAFKA_UI_TAG}
    environment:
      KAFKA_CLUSTERS_0_NAME: lake
      KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS: kafka:9092
    ports: ["8095:8080"]
    depends_on: [kafka]
    networks: [lake]

networks:
  lake:
    external: true
```

Two listeners exist because a client must reach the broker at the address the broker advertises. Containers use `kafka:9092`; your laptop uses `pensive:29092`. Getting this wrong is the most common Kafka-in-Docker problem.

Save it as `platform/compose.stream.yaml` and start it with `make up-stream`. Add these targets to `make/stream.mk`:

```makefile
K = $(DC) exec kafka /opt/kafka/bin

# usage: make topic NAME=trips.raw PARTS=12 CONFIG="--config retention.ms=604800000"
topic:
	$(K)/kafka-topics.sh --bootstrap-server kafka:9092 --create --if-not-exists \
	  --topic $(NAME) --partitions $(PARTS) --replication-factor 1 $(CONFIG)

topics:
	make topic NAME=trips.raw PARTS=12 CONFIG="--config retention.ms=604800000"
	make topic NAME=trips.alerts PARTS=3
	make topic NAME=zones.state PARTS=3 CONFIG="--config cleanup.policy=compact"
```

| Topic | Partitions | Policy | Used for |
| --- | --- | --- | --- |
| `trips.raw` | 12 | Delete after 7 days | Every trip event, keyed by pickup zone |
| `trips.alerts` | 3 | Default retention | Surge alerts that any service can subscribe to |
| `zones.state` | 3 | Compacted | Latest record per zone; old values for a key are cleaned up |

### Verify

Open four terminals and show both behaviors on `trips.alerts`:

```bash
# terminal 1 and 2: same group -> each message goes to only one of them
./dc exec kafka /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server kafka:9092 --topic trips.alerts --group ops
# terminal 3: a different group -> gets every message (fan-out)
./dc exec kafka /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server kafka:9092 --topic trips.alerts --group audit
# terminal 4: publish
./dc exec -it kafka /opt/kafka/bin/kafka-console-producer.sh \
  --bootstrap-server kafka:9092 --topic trips.alerts
```

Type a few lines in terminal 4. Terminals 1 and 2 split them; terminal 3 sees all of them. Then open `http://pensive:8095` and find both groups and their offsets.

### Commit

`feat(stream): Kafka 4.3 in KRaft mode, Kafka UI, topic definitions`

### Explore (optional)

- **Ordering is per partition, not per topic.** Produce keyed records with `--property parse.key=true --property key.separator=:` and check in the UI that one key always lands in one partition.
- **Replay.** Reset the `audit` group to the earliest offset with `kafka-consumer-groups.sh --reset-offsets --to-earliest --execute` and read everything again. A queue can't do this; a log can.
- **Compaction.** Write three values for the same key to `zones.state`, force a segment roll by lowering `segment.ms`, and watch older values disappear.
- **Share groups.** Consume `trips.alerts` with `kafka-console-share-consumer.sh` from two terminals. Unlike a consumer group, both can take records from the same partition.
- **Look at the storage.** List `${KAFKA_DATA}`. Each partition is a folder of segment files plus indexes. That's the whole data structure.

## Phase 12: Producers and event contracts

Goal: a producer that replays real taxi trips into `trips.raw` as a live stream, at a speed you choose, with a written contract for what an event looks like. It also injects a little late and duplicate data on purpose, because real streams always have both and Phase 14 has to handle them.

### Build

`contracts/trip_event.v1.json` is the contract. Producers validate against it; consumers rely on it:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "trip_event", "version": 1, "type": "object",
  "required": ["event_id", "pickup_ts", "pu_zone_id", "total_amount"],
  "properties": {
    "event_id":        {"type": "string"},
    "vendor_id":       {"type": ["integer", "null"]},
    "pickup_ts":       {"type": "string", "format": "date-time"},
    "dropoff_ts":      {"type": ["string", "null"], "format": "date-time"},
    "pu_zone_id":      {"type": "integer"},
    "do_zone_id":      {"type": ["integer", "null"]},
    "passenger_count": {"type": ["integer", "null"]},
    "trip_distance":   {"type": ["number", "null"]},
    "fare_amount":     {"type": ["number", "null"]},
    "tip_amount":      {"type": ["number", "null"]},
    "total_amount":    {"type": "number"}
  }
}
```

`docker/tools/Dockerfile`, a small Python image for producers and consumers:

```dockerfile
FROM python:3.12-slim
COPY requirements.lock /tmp/requirements.lock   # requirements.in: confluent-kafka, pyarrow, pandas, jsonschema
RUN pip install --no-cache-dir -r /tmp/requirements.lock
WORKDIR /app
```

`tools/replay_trips.py`:

```python
"""Replay one month of TLC trips into Kafka in event-time order."""
import argparse, json, random, time, uuid
import pandas as pd
from confluent_kafka import Producer
from jsonschema import Draft202012Validator

p = argparse.ArgumentParser()
p.add_argument("--month", required=True)                  # 2024-06
p.add_argument("--speed", type=float, default=60.0)      # 60 = one hour of trips per minute
p.add_argument("--late-pct", type=float, default=2.0)    # held back, sent later
p.add_argument("--dup-pct", type=float, default=0.5)     # sent twice
a = p.parse_args()

validate = Draft202012Validator(json.load(open("/app/contracts/trip_event.v1.json"))).validate
cols = {"VendorID": "vendor_id", "tpep_pickup_datetime": "pickup_ts", "tpep_dropoff_datetime": "dropoff_ts",
        "PULocationID": "pu_zone_id", "DOLocationID": "do_zone_id", "passenger_count": "passenger_count",
        "trip_distance": "trip_distance", "fare_amount": "fare_amount", "tip_amount": "tip_amount",
        "total_amount": "total_amount"}
df = pd.read_parquet(f"/data/landing/tlc/yellow/yellow_tripdata_{a.month}.parquet")
df = df.rename(columns={k: v for k, v in cols.items() if k in df.columns})[list(cols.values())]
df = df.dropna(subset=["pickup_ts", "pu_zone_id"]).sort_values("pickup_ts")

prod = Producer({"bootstrap.servers": "kafka:9092", "enable.idempotence": True, "acks": "all",
                 "compression.type": "zstd", "linger.ms": 20})

def to_event(r):
    e = {k: (None if pd.isna(v) else v) for k, v in r.items()}
    for k in ("vendor_id", "pu_zone_id", "do_zone_id", "passenger_count"):
        e[k] = None if e[k] is None else int(e[k])
    for k in ("pickup_ts", "dropoff_ts"):
        e[k] = None if e[k] is None else e[k].isoformat()
    e["event_id"] = str(uuid.uuid4())
    return e

held, start_wall, start_evt = [], time.time(), df["pickup_ts"].iloc[0]
for r in df.to_dict("records"):
    lag = (r["pickup_ts"] - start_evt).total_seconds() / a.speed - (time.time() - start_wall)
    if lag > 0:
        time.sleep(lag)                                  # pace by event time
    e = to_event(r); validate(e)
    if random.random() * 100 < a.late_pct:
        held.append(e); continue                         # will arrive late
    payload = json.dumps(e).encode()
    prod.produce("trips.raw", key=str(e["pu_zone_id"]), value=payload)
    if random.random() * 100 < a.dup_pct:
        prod.produce("trips.raw", key=str(e["pu_zone_id"]), value=payload)   # duplicate
    if len(held) > 500:                                  # release late events in a burst
        for h in held:
            prod.produce("trips.raw", key=str(h["pu_zone_id"]), value=json.dumps(h).encode())
        held.clear()
    prod.poll(0)
prod.flush()
```

The key is the pickup zone, so all events for one zone stay in order on one partition. The `event_id` makes duplicates detectable downstream.

Add the service and a target:

```yaml
  tools:
    build: ./docker/tools
    profiles: [tools]
    volumes:
      - ../pipelines/tools:/app/tools:ro
      - ../pipelines/contracts:/app/contracts:ro
      - ${DATA_ROOT}/landing:/data/landing:ro
    networks: [lake]
```

```makefile
# usage: make replay MONTH=2024-06 SPEED=600
replay:
	$(DC) run --rm tools python tools/replay_trips.py --month $(MONTH) --speed $(or $(SPEED),60)
```

### Verify

- `make replay MONTH=2024-06 SPEED=600`, then in the Kafka UI watch `trips.raw` message counts climb across all 12 partitions.
- Open a message in the UI and check it matches the contract.
- Stop the producer with Ctrl-C and restart it. Note that it starts from the beginning of the month: replay isn't resumable yet. That's fine for a simulator.

### Commit

`feat(stream): trip event contract and replay producer with late and duplicate events`

### Explore (optional)

- **Throughput.** Set `--speed 100000` and measure events per second. Then change `linger.ms`, batch size and compression, and see which matters most.
- **A schema registry.** Run Apicurio Registry or Karapace, switch the contract to Avro, and register it. Try an incompatible change (a required field removed) and watch the registry refuse it.
- **Idempotence.** Turn `enable.idempotence` off, restart the broker while producing, and look for duplicated offsets. Then turn it back on.
- **A second producer.** Write `tools/publish_zones.py` that publishes `silver.zones` rows to the compacted `zones.state` topic, keyed by zone ID. That's change data capture in miniature.

## Phase 13: Structured Streaming from Kafka into Iceberg

Goal: a streaming job that reads `trips.raw` and appends to `lake.bronze.trip_events` every 30 seconds, survives restarts without losing or duplicating data, and is queryable from Trino while it runs.

Structured Streaming treats a stream as a table that keeps growing. Each **micro-batch** reads new offsets, runs an ordinary batch plan, and commits. Two things make it exactly-once end to end:

1. The **checkpoint** records which Kafka offsets each batch covered, before the batch runs.
2. The **Iceberg sink** tags each commit with the batch ID, so a batch replayed after a crash is recognized and not written twice.

### Build

Add the Kafka connector to `docker/spark/Dockerfile`. Rather than guessing transitive versions, let Spark resolve them once at build time and copy the jars in (put this before `USER spark`):

```dockerfile
RUN /opt/spark/bin/spark-submit --master 'local[1]' \
      --conf spark.jars.ivy=/tmp/ivy \
      --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.2 \
      --class org.apache.spark.examples.SparkPi /opt/spark/examples/jars/spark-examples_2.13-4.1.2.jar 1 && \
    cp /tmp/ivy/jars/*.jar /opt/spark/jars/ && rm -rf /tmp/ivy
```

That pulls `spark-sql-kafka`, `spark-token-provider-kafka`, `kafka-clients` and `commons-pool2` at matching versions. Then add a shared checkpoint folder:

```bash
sudo mkdir -p /lake/lakehouse/checkpoints && sudo chown 185:185 /lake/lakehouse/checkpoints
```

In `compose.spark.yaml`, add `- ${DATA_ROOT}/checkpoints:/data/checkpoints` to both the `x-spark` volume list and the worker's list. Drivers write offsets there and executors write streaming state there, so every container must see the same folder. On AWS this would be an S3 prefix.

`jobs/stream/trips_bronze.py`:

```python
import sys
sys.path.insert(0, "/opt/jobs")
from pyspark.sql import functions as F
from pensieve.lake.spark import get_spark

EVENT = ("event_id STRING, vendor_id INT, pickup_ts TIMESTAMP, dropoff_ts TIMESTAMP, "
         "pu_zone_id INT, do_zone_id INT, passenger_count INT, trip_distance DOUBLE, "
         "fare_amount DOUBLE, tip_amount DOUBLE, total_amount DOUBLE")

spark = get_spark("stream:trips_bronze")
spark.sql("CREATE NAMESPACE IF NOT EXISTS lake.bronze")
spark.sql(f"""CREATE TABLE IF NOT EXISTS lake.bronze.trip_events (
    {EVENT}, kafka_partition INT, kafka_offset BIGINT, kafka_ts TIMESTAMP, raw STRING)
  USING iceberg PARTITIONED BY (days(pickup_ts))""")

raw = (spark.readStream.format("kafka")
       .option("kafka.bootstrap.servers", "kafka:9092")
       .option("subscribe", "trips.raw")
       .option("startingOffsets", "earliest")       # only used on the very first run
       .option("maxOffsetsPerTrigger", 500_000)     # caps each batch, which bounds catch-up
       .load())

events = raw.select(
    F.from_json(F.col("value").cast("string"), EVENT).alias("e"),
    F.col("partition").alias("kafka_partition"), F.col("offset").alias("kafka_offset"),
    F.col("timestamp").alias("kafka_ts"), F.col("value").cast("string").alias("raw"),
).select("e.*", "kafka_partition", "kafka_offset", "kafka_ts", "raw")

q = (events.writeStream.format("iceberg").outputMode("append")
     .trigger(processingTime="30 seconds")
     .option("fanout-enabled", "true")             # don't require rows sorted by partition
     .option("checkpointLocation", "file:///data/checkpoints/trips_bronze")
     .toTable("lake.bronze.trip_events"))
q.awaitTermination()
```

Bronze keeps the raw JSON and Kafka coordinates, and deliberately keeps duplicates. If a later stage gets something wrong, you can rebuild it from bronze without going back to Kafka.

Run it as a capped job (it never ends on its own):

```bash
make submit JOB=stream/trips_bronze.py SUBMIT_OPTS="--conf spark.cores.max=8"
```

### Verify

With the producer running in another terminal:

- The application UI's **Structured Streaming** tab shows one batch every 30 seconds, with input rate and processing rate.
- In Trino, `SELECT count(*) FROM bronze.trip_events` grows between runs.
- **The exactly-once test.** Stop the job with Ctrl-C mid-stream and start it again. Then check that no Kafka offset was written twice:

```sql
SELECT kafka_partition, kafka_offset, count(*) AS n
FROM bronze.trip_events GROUP BY 1, 2 HAVING count(*) > 1;   -- expect no rows
```

### Commit

`feat(stream): Kafka to Iceberg bronze stream with checkpoints`

### Explore (optional)

- **Read the checkpoint.** Look in `${DATA_ROOT}/checkpoints/trips_bronze/offsets` and `commits`. Each batch has a small file listing Kafka offsets per partition. Delete the folder, restart, and see the job reprocess from `earliest`, which is why checkpoints are precious.
- **Find the speed limit.** Raise the producer speed until the processing rate falls behind the input rate. Then raise `spark.cores.max`. Kafka partitions cap parallelism: with 12 partitions, more than 12 cores won't help reads.
- **Trigger choices.** Compare `processingTime="5 seconds"`, `"2 minutes"` and `availableNow=True`. The last processes everything waiting, then stops, so it runs well from Airflow as a cheap "streaming as batch" job.
- **Iceberg as a stream source.** Start a second query with `spark.readStream.format("iceberg").load("lake.bronze.trip_events")`. Chaining tables this way is common in medallion streaming.

## Phase 14: Stateful streaming, late data and alerts

Goal: one streaming app that deduplicates events, counts trips per zone in 5-minute windows, writes the counts to Iceberg, and publishes surge alerts to `trips.alerts` for any subscriber. This is the full pub/sub loop: Kafka in, Spark in the middle, Kafka out.

Three ideas carry this phase:

- **Event time versus processing time.** A trip at 08:01 that arrives at 08:20 still belongs to the 08:00–08:05 window. Windows are defined on `pickup_ts`, not on arrival time.
- **A watermark** says how late data may be. With 15 minutes, Spark keeps each window open until it has seen events 15 minutes past the window's end, then emits it and frees its memory. Anything later is dropped.
- **State** is what Spark remembers between batches: open windows and seen event IDs. It lives in a state store under the checkpoint. Watermarks are what keep it from growing forever.

### Build

First, a batch job builds the "normal" demand that alerts compare against. `jobs/transform/zone_baseline.py`:

```python
import sys
sys.path.insert(0, "/opt/jobs")
from pensieve.lake.spark import get_spark

spark = get_spark("zone_baseline")
spark.sql("""
CREATE OR REPLACE TABLE lake.gold.zone_baseline_5min USING iceberg AS
SELECT pu_zone_id, dayofweek(w.start) AS dow, hour(w.start) AS hr, avg(n) AS avg_trips
FROM (SELECT pu_zone_id, window(pickup_ts, '5 minutes') AS w, count(*) AS n
      FROM lake.silver.trips GROUP BY 1, 2)
GROUP BY 1, 2, 3""")
```

Then `jobs/stream/zone_demand.py`:

```python
import sys
sys.path.insert(0, "/opt/jobs")
from pyspark.sql import functions as F
from pensieve.lake.spark import get_spark

EVENT = ("event_id STRING, pickup_ts TIMESTAMP, pu_zone_id INT, "
         "trip_distance DOUBLE, total_amount DOUBLE")
CKPT = "file:///data/checkpoints/zone_demand"

spark = get_spark("stream:zone_demand")
spark.sql("""CREATE TABLE IF NOT EXISTS lake.gold.zone_demand_5min (
  window_start TIMESTAMP, window_end TIMESTAMP, pu_zone_id INT,
  trips BIGINT, avg_total DOUBLE, avg_miles DOUBLE)
  USING iceberg PARTITIONED BY (days(window_start))""")

events = (spark.readStream.format("kafka")
          .option("kafka.bootstrap.servers", "kafka:9092")
          .option("subscribe", "trips.raw")
          .option("startingOffsets", "latest")
          .load()
          .select(F.from_json(F.col("value").cast("string"), EVENT).alias("e")).select("e.*")
          .withWatermark("pickup_ts", "15 minutes")
          .dropDuplicatesWithinWatermark(["event_id"]))     # removes the producer's duplicates

counts = (events.groupBy(F.window("pickup_ts", "5 minutes").alias("w"), "pu_zone_id")
          .agg(F.count("*").alias("trips"),
               F.round(F.avg("total_amount"), 2).alias("avg_total"),
               F.round(F.avg("trip_distance"), 2).alias("avg_miles"))
          .select(F.col("w.start").alias("window_start"), F.col("w.end").alias("window_end"),
                  "pu_zone_id", "trips", "avg_total", "avg_miles"))

# Query 1: finished windows into Iceberg (exactly-once).
q1 = (counts.writeStream.format("iceberg").outputMode("append")
      .trigger(processingTime="30 seconds")
      .option("checkpointLocation", f"{CKPT}/iceberg")
      .toTable("lake.gold.zone_demand_5min"))

# Query 2: surge alerts to Kafka. A stream-static join adds the baseline.
baseline = spark.table("lake.gold.zone_baseline_5min")
alerts = (counts
    .withColumn("dow", F.dayofweek("window_start")).withColumn("hr", F.hour("window_start"))
    .join(baseline, ["pu_zone_id", "dow", "hr"])
    .where("trips >= 20 AND trips > 3 * avg_trips")
    .select(F.col("pu_zone_id").cast("string").alias("key"),
            F.to_json(F.struct(
                F.concat_ws(":", "pu_zone_id", F.col("window_start").cast("string")).alias("alert_id"),
                "pu_zone_id", "window_start", "trips",
                F.round("avg_trips", 1).alias("normal_trips"))).alias("value")))
q2 = (alerts.writeStream.format("kafka").outputMode("append")
      .option("kafka.bootstrap.servers", "kafka:9092")
      .option("topic", "trips.alerts")
      .option("checkpointLocation", f"{CKPT}/alerts")
      .start())

spark.streams.awaitAnyTermination()
```

Each query has its own checkpoint and its own state. The Iceberg sink is exactly-once. The Kafka sink is at-least-once: after a crash, an alert can be published twice. That's why each alert carries a stable `alert_id` that subscribers can deduplicate on.

A subscriber, `tools/alert_listener.py`:

```python
import json, sys
from confluent_kafka import Consumer

c = Consumer({"bootstrap.servers": "kafka:9092", "group.id": sys.argv[1] if len(sys.argv) > 1 else "dispatch",
              "auto.offset.reset": "latest"})
c.subscribe(["trips.alerts"])
seen = set()
while True:
    m = c.poll(1.0)
    if m is None or m.error():
        continue
    a = json.loads(m.value())
    if a["alert_id"] in seen:
        continue                      # deduplicate at-least-once delivery
    seen.add(a["alert_id"])
    print(f"SURGE zone {a['pu_zone_id']} at {a['window_start']}: {a['trips']} trips (normal {a['normal_trips']})")
```

### Verify

```bash
make submit JOB=transform/zone_baseline.py
make submit JOB=stream/zone_demand.py SUBMIT_OPTS="--conf spark.cores.max=12" &
./dc run --rm tools python tools/alert_listener.py dispatch &
make replay MONTH=2024-06 SPEED=600
```

- After about 20 minutes of event time, rows appear in `gold.zone_demand_5min`. Windows only land once the watermark passes them.
- Alerts print in the listener. Start a second listener with group `audit`: it gets the same alerts.
- In the streaming tab, each query's progress shows the state store row count rising and then levelling off. If it keeps climbing, the watermark isn't working.

### Commit

`feat(stream): windowed zone demand with watermark, dedup and Kafka alerts`

### Explore (optional)

- **How late is too late?** Raise the producer's `--late-pct` and hold events longer. Watch `numRowsDroppedByWatermark` in the query progress JSON. Then widen the watermark and watch state size grow. That's the trade-off between completeness and memory.
- **Output modes.** Switch query 1 to `update` mode with a `foreachBatch` that runs `MERGE INTO`. You'll get early, revised counts instead of waiting for windows to close.
- **State store backend.** Set `spark.sql.streaming.stateStore.providerClass` to the RocksDB provider and compare memory use with many zones and long windows.
- **Real-time mode.** Spark 4.1 added a [Real-time Mode](https://apache.googlesource.com/spark/+/master/docs/streaming/real-time-mode.md) for millisecond latency. In 4.1 it supports stateless queries in `update` mode only. Write a Kafka-to-Kafka filter for fares above $100 with `Trigger.RealTime("5 minutes")` and compare end-to-end latency with a 1-second micro-batch. If your PySpark doesn't expose the trigger yet, write it in `spark-shell` (Scala).
- **Custom state.** Rewrite the surge rule with `transformWithStateInPandas`, keeping a running per-zone average in state instead of the batch baseline.

## Phase 15: Operating streams

Goal: streaming jobs that run as services and restart on their own, dashboards that show whether they keep up, tables that stay healthy under constant small commits, and a set of failure drills you've actually run.

Batch and streaming are operated differently. A batch job is a task Airflow starts and waits for. A streaming job is a service: it should always be running, and the questions are "is it keeping up?" and "what happens when it dies?"

### Build: streams as services

Add to `compose.stream.yaml`:

```yaml
x-stream: &stream
  image: lakehouse/spark:4.1.2
  restart: unless-stopped
  networks: [lake]
  environment:
    SOURCE_DB_PASSWORD: ${SOURCE_DB_PASSWORD}
  volumes:
    - ./conf/spark:/opt/spark/conf:ro
    - ../pipelines/jobs:/opt/jobs:ro
    - ../libs:/opt/libs:ro
    - ${DATA_ROOT}/spark-events:/opt/spark/events
    - ${DATA_ROOT}/checkpoints:/data/checkpoints

services:
  stream-bronze:
    <<: *stream
    profiles: [stream-jobs]
    command: lake-submit --conf spark.cores.max=8 --conf spark.sql.streaming.metricsEnabled=true
             --conf spark.ui.port=4041 /opt/jobs/stream/trips_bronze.py
    ports: ["4041:4041"]

  stream-demand:
    <<: *stream
    profiles: [stream-jobs]
    command: lake-submit --conf spark.cores.max=12 --conf spark.sql.streaming.metricsEnabled=true
             --conf spark.ui.port=4042 /opt/jobs/stream/zone_demand.py
    ports: ["4042:4042"]

  kafka-exporter:
    image: danielqsj/kafka-exporter:${KAFKA_EXPORTER_TAG}
    command: --kafka.server=kafka:9092
    ports: ["9308:9308"]
    networks: [lake]
```

`restart: unless-stopped` plus the checkpoint means a crash costs a few seconds and no data. Each stream gets its own driver UI port so you can watch them side by side. Both sit behind the stream-jobs profile, so make up-stream doesn't start them by accident; start them with ./dc --profile stream-jobs up -d stream-bronze stream-demand.

### Build: watching them

The key question is **lag**: how far behind the newest data the job is. Spark tracks its own Kafka offsets in the checkpoint rather than in a consumer group, so Kafka-side consumer lag tools won't see it. Use both views:

| Signal | Where it comes from | Alert when |
| --- | --- | --- |
| Input rate versus processing rate | Spark streaming metrics on the driver UI (`/metrics/prometheus`) | Processing stays below input for 10 minutes |
| Batch duration | Same | Longer than the trigger interval, so batches queue up |
| Topic end offsets | `kafka-exporter` on port 9308 | Combined with Spark's committed offsets, gives true lag |
| Event-time lag | `now() - max(pickup_ts)` from Trino every minute | Beyond your freshness target |
| State rows and memory | Streaming query progress, `stateOperators` | Growing without levelling off |

Add the ports to your Alloy scrape config next to tutorial 1's Spark targets.

### Build: keeping tables healthy

A 30-second trigger makes about 2,900 commits a day, each with small files. Run the Phase 8 maintenance job on stream tables every hour, from Airflow, while the stream keeps writing. Compaction and appends don't conflict in Iceberg, because appends only add files.

```python
# dags/stream_maintenance.py
from datetime import datetime
from airflow.sdk import DAG
from common.spark import spark_job         # the helper from tutorial 1, moved to dags/common/spark.py

with DAG("stream_maintenance", start_date=datetime(2026, 1, 1), schedule="@hourly",
         catchup=False, max_active_runs=1) as dag:
    for t in ("bronze.trip_events", "gold.zone_demand_5min"):
        spark_job(f"optimize_{t.replace('.', '_')}", "maintain/optimize_table.py",
                  f"--table {t} --keep-days 2 --target-mb 256", cores=8)
```

To import `spark_job` this way, move it from `tlc_monthly.py` into `dags/common/spark.py` and import it from there in both DAGs.

### Verify: the failure drills

Run each drill with the producer going, and write what you saw in `docs/stream-drills.md`.

1. **Driver crash.** `docker kill` the `stream-bronze` container. It restarts, and the offset-duplicate query from Phase 13 still returns no rows.
2. **Broker restart.** Restart Kafka. Both streams stall, retry, and catch up without manual help.
3. **Backlog.** Stop `stream-demand` for 30 minutes, then start it. Watch it catch up in chunks of `maxOffsetsPerTrigger`, with lag shrinking in Grafana.
4. **Code change.** Add a column to `zone_demand.py` and restart. Adding a projection is checkpoint-compatible; changing the grouping keys isn't. Try both and read the error.
5. **Retention runs out.** Lower `trips.raw` retention to 10 minutes, stop a stream for 20, and restart. Spark fails because offsets are gone; `failOnDataLoss=false` skips ahead instead. Decide which you'd want in production, then restore retention.

### Commit

`feat(stream): streams as services, lag metrics, hourly maintenance, failure drills`

### Explore (optional)

- **Backfill from bronze.** Stop `stream-demand`, delete its checkpoint, and rebuild `gold.zone_demand_5min` for one day with a batch job reading `bronze.trip_events`. Same logic, two execution modes.
- **Exactly-once to Kafka.** Replace the alerts Kafka sink with `foreachBatch` and a transactional producer whose transactional ID includes the batch ID. Compare duplicates after a crash.
- **Kafka Connect.** Run Kafka Connect with the Iceberg sink connector and compare it with your Spark bronze job: less code, less control.
- **Second consumer engine.** Read `trips.raw` with Trino's Kafka connector and query the live topic directly with SQL.

## Part B: MLOps

## Phase 16: The ML platform base

Goal: GPUs usable from Docker, an MLflow tracking server that stores runs in Postgres and artifacts in SeaweedFS, and one locked training image that every later phase uses. MLflow plays the part of SageMaker Experiments and the Model Registry.

### Build: GPUs in containers

Install the NVIDIA Container Toolkit from NVIDIA's apt repository, then:

```bash
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi   # both GPUs listed
```

Decide which GPU does what, since you also serve LLMs on this box. This tutorial assumes **GPU 0 for serving** (vLLM) and **GPU 1 for training**. Add `TRAIN_GPU=1` and `SERVE_GPU=0` to `.env`, plus `MODEL_CACHE=/trunk/ai/models` for downloaded model weights. Point it at your existing model library on trunk.

### Build: MLflow

`docker/mlflow/Dockerfile`:

```dockerfile
ARG MLFLOW_TAG
FROM ghcr.io/mlflow/mlflow:${MLFLOW_TAG}
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock
```

`MLFLOW_TAG` comes from `versions.env`, and `requirements.in` lists `psycopg2-binary` and `boto3`. Save the Compose file as `platform/compose.ml.yaml`, start the stack with `make up-ml`, and put this tutorial's ML targets in `make/ml.mk`. `compose.ml.yaml`:

```yaml
services:
  mlflow-db:
    image: postgres:${POSTGRES_TAG}
    environment:
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: ${MLFLOW_DB_PASSWORD}
      POSTGRES_DB: mlflow
    volumes:
      - ${PG_ROOT}/mlflow-db:/var/lib/postgresql/data
    networks: [lake]

  mlflow:
    build: { context: ./docker/mlflow, args: [MLFLOW_TAG] }
    command: >
      mlflow server
      --backend-store-uri postgresql://mlflow:${MLFLOW_DB_PASSWORD}@mlflow-db:5432/mlflow
      --artifacts-destination s3://mlflow
      --serve-artifacts
      --host 0.0.0.0 --port 5000
      --allowed-hosts mlflow,mlflow:5000,${HOST_NAME},${HOST_NAME}:5000,localhost:5000
    environment:
      MLFLOW_S3_ENDPOINT_URL: http://seaweedfs:8333
      AWS_ACCESS_KEY_ID: ${S3_ACCESS_KEY}
      AWS_SECRET_ACCESS_KEY: ${S3_SECRET_KEY}
    ports: ["5000:5000"]
    depends_on: [mlflow-db]
    networks: [lake]

networks:
  lake:
    external: true
```

`--serve-artifacts` makes MLflow a proxy for artifact uploads and downloads, so training code only needs `MLFLOW_TRACKING_URI`, never S3 keys. MLflow 3.5 and later check the `Host` header of every request ([tracking server docs](https://mlflow.org/docs/latest/ml/tracking/server)); `--allowed-hosts` lists the names you'll use. Create the bucket with `make buckets` after adding `mlflow` to the bucket list.

### Build: one locked training image

`ml/pyproject.toml` (torch comes from PyTorch's CUDA 12.8 index, which supports Blackwell):

```toml
[project]
name = "lake-ml"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
  "torch>=2.7", "transformers", "peft", "trl", "datasets", "accelerate", "bitsandbytes",
  "mlflow>=3", "xgboost", "scikit-learn", "pandas", "pyarrow", "pyiceberg[pyarrow]",
  "optuna", "evidently", "trino", "pyyaml", "psycopg2-binary",
]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
```

Run `uv lock` in `ml/` once and commit `uv.lock`. From then on every build installs exactly the same versions, which is half of reproducibility.

`docker/trainer/Dockerfile` (build context is the repo root):

```dockerfile
ARG UV_VERSION
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv git curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /work
COPY ml/pyproject.toml ml/uv.lock ./
RUN uv sync --frozen --no-install-project       # dependencies: a cached layer
ENV PATH=/work/.venv/bin:$PATH
COPY libs/ ./libs/
ENV PYTHONPATH=/work/libs
COPY ml/ ./ml/                                   # code: changes every commit
```

Add the trainer to `compose.ml.yaml`:

```yaml
  trainer:
    build: { context: .., dockerfile: platform/docker/trainer/Dockerfile, args: [UV_VERSION] }
    image: lakehouse/trainer:${GIT_SHA:-dev}
    profiles: [ml]
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
      GIT_SHA: ${GIT_SHA:-dev}
      HF_HOME: /cache/hf
      HF_TOKEN: ${HF_TOKEN:-}
      POLARIS_CREDENTIAL: ${POLARIS_CLIENT_ID}:${POLARIS_CLIENT_SECRET}
      S3_ACCESS_KEY: ${S3_ACCESS_KEY}
      S3_SECRET_KEY: ${S3_SECRET_KEY}
    volumes:
      - ${MODEL_CACHE}:/cache
    shm_size: 16g                  # PyTorch data loaders use shared memory
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["${GPU:-1}"]
              capabilities: [gpu]
    networks: [lake]
```

The image tag is the Git commit, so a run can always name the exact code that produced it. Add Makefile plumbing:

```makefile
GIT_SHA := $(shell git rev-parse --short HEAD)$(shell git diff --quiet || echo -dirty)
GPU ?= $(TRAIN_GPU)
export GIT_SHA GPU

ml-build:
	$(DC) --profile ml build trainer

# usage: make ml-run CMD="python -m ml.check_gpu"
ml-run: ml-build
	$(DC) --profile ml run --rm trainer $(CMD)
```

### Verify

`ml/check_gpu.py`:

```python
import mlflow, torch

print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
mlflow.set_experiment("platform-check")
with mlflow.start_run():
    x = torch.randn(8192, 8192, device="cuda")
    mlflow.log_metric("matmul_sum", float((x @ x).sum()))
```

- `make ml-run CMD="python -m ml.check_gpu"` prints a CUDA 12.8 build and capability `(12, 0)`.
- `http://pensive:5000` shows the `platform-check` experiment with one run, tagged with nothing yet. Phase 18 fixes that.

### Commit

`feat(ml): GPU runtime, MLflow on Postgres and S3, locked trainer image`

### Explore (optional)

- **Where do artifacts go?** Log a file with `mlflow.log_artifact`, then find it with `make s3 ARGS="ls --recursive s3://mlflow/"`. Compare what's in Postgres (runs, params, metrics) and what's in S3 (files).
- **Image hygiene.** Check the trainer image size with `docker images`. Find which layer is biggest with `docker history`, and think about what a slim inference image would leave out.
- **Lock drift.** Delete `uv.lock`, run `uv lock` again a month later, and diff it. That's why you commit the lock.
- **GPU sharing.** Run `nvidia-smi dmon` while the check runs. Then try running it on the serving GPU while vLLM is loaded and see how memory runs out.

## Phase 17: Features from the lakehouse and a first tracked model

Goal: predict trip duration at pickup time. Spark builds a feature table in Iceberg; a GPU XGBoost model trains on it; MLflow records the parameters, metrics, model and the exact table snapshot used. That last part means any model can be traced back to the precise rows it learned from.

### Build: the feature table

Only use what's known at pickup. Fare and total amount are computed from the meter, which runs for the whole trip, so they'd leak the answer. `jobs/ml/build_trip_features.py`:

```python
import argparse, sys
sys.path.insert(0, "/opt/jobs")
from pyspark.sql import functions as F
from pensieve.lake.spark import get_spark

p = argparse.ArgumentParser(); p.add_argument("--month", required=True); a = p.parse_args()
spark = get_spark(f"features:trip_duration:{a.month}")
spark.sql("CREATE NAMESPACE IF NOT EXISTS lake.ml")

weather = (spark.table("lake.bronze.weather_hourly")
           .select(F.date_trunc("hour", "ts").alias("hour"), "temperature_2m", "precipitation"))
zones = spark.table("lake.silver.zones").select("location_id", "borough")

feat = (spark.table("lake.silver.trips")
  .where(F.date_format("pickup_ts", "yyyy-MM") == a.month)
  .withColumn("duration_min", (F.unix_timestamp("dropoff_ts") - F.unix_timestamp("pickup_ts")) / 60)
  .where("duration_min BETWEEN 1 AND 180")
  .withColumn("hour", F.date_trunc("hour", "pickup_ts"))
  .join(weather, "hour", "left")
  .join(zones.withColumnRenamed("location_id", "pu_zone_id").withColumnRenamed("borough", "pu_borough"), "pu_zone_id", "left")
  .join(zones.withColumnRenamed("location_id", "do_zone_id").withColumnRenamed("borough", "do_borough"), "do_zone_id", "left")
  .select("pickup_ts", F.lit(a.month).alias("month"),
          F.hour("pickup_ts").alias("pickup_hour"), F.dayofweek("pickup_ts").alias("pickup_dow"),
          "pu_zone_id", "do_zone_id", "pu_borough", "do_borough",
          "trip_distance", "passenger_count", "temperature_2m", "precipitation", "duration_min"))

if spark.catalog.tableExists("lake.ml.trip_duration_features"):
    feat.writeTo("lake.ml.trip_duration_features").overwritePartitions()
else:
    feat.writeTo("lake.ml.trip_duration_features").partitionedBy(F.col("month")).create()
```

Using a destination zone at pickup time assumes riders enter a destination, as in app-booked rides. That's a modelling choice worth writing down; an exercise removes it.

### Build: reading Iceberg from Python

The trainer reads the table with PyIceberg straight into Arrow, with no Spark involved. `libs/pensieve/lake/catalog.py`:

```python
import os
from pyiceberg.catalog import load_catalog

def catalog():
    return load_catalog("lake", **{
        "type": "rest",
        "uri": "http://polaris:8181/api/catalog",
        "warehouse": "lake",
        "credential": os.environ["POLARIS_CREDENTIAL"],
        "scope": "PRINCIPAL_ROLE:ALL",
        "s3.endpoint": "http://seaweedfs:8333",
        "s3.access-key-id": os.environ["S3_ACCESS_KEY"],
        "s3.secret-access-key": os.environ["S3_SECRET_KEY"],
        "s3.region": "us-east-1",
    })

def read_table(name, row_filter="true", snapshot_id=None):
    """Return (pandas DataFrame, snapshot_id) so the caller can log exactly what it read."""
    t = catalog().load_table(name)
    snap = snapshot_id or t.current_snapshot().snapshot_id
    return t.scan(row_filter=row_filter, snapshot_id=snap).to_pandas(), snap
```

### Build: the training script

`ml/train_tabular.py` takes a config file, so Phase 18 can run it as a job unchanged:

```python
import argparse, os, yaml
import mlflow, numpy as np, xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from pensieve.lake.catalog import read_table

CAT = ["pu_zone_id", "do_zone_id", "pu_borough", "do_borough"]
NUM = ["pickup_hour", "pickup_dow", "trip_distance", "passenger_count", "temperature_2m", "precipitation"]

def prep(df):
    X = df[CAT + NUM].copy()
    for c in CAT:
        X[c] = X[c].astype("category")
    return X, df["duration_min"].to_numpy()

def main(cfg, nested=False):
    mlflow.set_experiment(cfg["experiment"])
    with mlflow.start_run(run_name=cfg.get("run_name"), nested=nested) as run:
        df, snap = read_table(cfg["table"], snapshot_id=cfg.get("snapshot_id"))
        mlflow.set_tags({"git_sha": os.environ.get("GIT_SHA", "dev"), "data.table": cfg["table"],
                         "data.snapshot_id": str(snap), "image": f"lakehouse/trainer:{os.environ.get('GIT_SHA')}"})
        mlflow.log_dict(cfg, "config.yaml")
        split = {k: df[df["month"].isin(v)] for k, v in cfg["split"].items()}    # split by time, never randomly
        (Xtr, ytr), (Xva, yva), (Xte, yte) = (prep(split[k]) for k in ("train", "valid", "test"))

        params = {"device": "cuda", "tree_method": "hist", "objective": "reg:absoluteerror",
                  "enable_categorical": True, **cfg["params"]}
        mlflow.log_params(params)
        model = xgb.XGBRegressor(**params, early_stopping_rounds=50)
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=200)

        baseline = np.full_like(yte, np.median(ytr))                 # always compare with a dumb model
        pred = model.predict(Xte)
        mlflow.log_metrics({"test_mae": mean_absolute_error(yte, pred),
                            "test_rmse": root_mean_squared_error(yte, pred),
                            "baseline_mae": mean_absolute_error(yte, baseline),
                            "best_iteration": model.best_iteration})
        mlflow.xgboost.log_model(model, name="model", input_example=Xte.head(5),
                                 registered_model_name=cfg.get("register_as"))
        print("run", run.info.run_id)
        return mean_absolute_error(yte, pred)

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True)
    main(yaml.safe_load(open(p.parse_args().config)))
```

`ml/configs/trip_duration.yaml`:

```yaml
experiment: trip-duration
run_name: xgb-baseline
table: ml.trip_duration_features
split:
  train: ["2024-01", "2024-02", "2024-03", "2024-04"]
  valid: ["2024-05"]
  test:  ["2024-06"]
params: { n_estimators: 2000, learning_rate: 0.05, max_depth: 8, subsample: 0.8, colsample_bytree: 0.8 }
register_as: trip_duration
```

### Verify

```bash
for m in 2024-01 2024-02 2024-03 2024-04 2024-05 2024-06; do
  make submit JOB=ml/build_trip_features.py ARGS="--month $m"; done
make ml-run CMD="python -m ml.train_tabular --config ml/configs/trip_duration.yaml"
```

- In MLflow, the run shows `test_mae` well below `baseline_mae`. If it doesn't, something is wrong with the features.
- The run's tags include `data.snapshot_id`. In Trino, `SELECT * FROM ml."trip_duration_features$snapshots"` lists that ID.
- **Models** in the MLflow UI shows `trip_duration` version 1.

### Commit

`feat(ml): trip duration features, PyIceberg reader, tracked XGBoost training`

### Explore (optional)

- **Prove the reproducibility.** Append another month of features, then retrain with `snapshot_id` set to the old run's tag. You should get the same row count and nearly the same metrics.
- **Remove the destination.** Drop `do_zone_id` and `do_borough` to model a street hail. How much worse does it get, and for which boroughs?
- **Leakage on purpose.** Add `fare_amount` as a feature and watch `test_mae` collapse. Then explain in one sentence why that model is useless in production.
- **Spark versus Arrow.** Time `read_table` for six months of features, then time the same read with Spark and `toPandas()`. Arrow end to end is a direct answer to your earlier question.
- **Error by segment.** Log a table artifact of MAE per borough and per pickup hour with `mlflow.log_table`. Averages hide where the model fails.

## Phase 18: Training jobs as a service

Goal: a SageMaker-style experience. You submit a config, pick a GPU, and get back a tracked run; you can rerun any past run exactly; and you can sweep hyperparameters across both GPUs in parallel. The pieces already exist. This phase turns them into a small, consistent interface.

A SageMaker training job is four things: a container image, an input data location, hyperparameters, and an output location, all recorded together. Here they are:

| SageMaker training job | Your equivalent |
| --- | --- |
| Training image in ECR | `lakehouse/trainer:<git sha>` |
| Input data channel (S3 prefix) | Iceberg table plus a pinned snapshot ID |
| Hyperparameters | `ml/configs/*.yaml`, logged as `config.yaml` |
| Instance type | `GPU=0` or `GPU=1` |
| Model artifacts in S3 | MLflow model, stored in `s3://mlflow` |
| Training job record | MLflow run with tags |

### Build: one entry point

`ml/train.py` dispatches on a `task` field, so every kind of training is launched the same way:

```python
import argparse, yaml, importlib

TASKS = {"tabular": "ml.train_tabular", "sft": "ml.train_sft"}   # sft arrives in Phase 19

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--set", nargs="*", default=[], help="overrides like params.max_depth=10")
a = p.parse_args()

cfg = yaml.safe_load(open(a.config))
for kv in a.set:                       # command-line overrides, recorded with the run
    key, val = kv.split("=", 1)
    node = cfg
    *parents, leaf = key.split(".")
    for k in parents:
        node = node.setdefault(k, {})
    node[leaf] = yaml.safe_load(val)
importlib.import_module(TASKS[cfg["task"]]).main(cfg)
```

Add `task: tabular` to `trip_duration.yaml`. Then the Makefile targets:

```makefile
# usage: make train CONFIG=ml/configs/trip_duration.yaml GPU=1 SET="params.max_depth=10"
train: ml-build
	$(DC) --profile ml run --rm trainer python -m ml.train --config $(CONFIG) --set $(SET)

# usage: make reproduce RUN_ID=<mlflow run id>
reproduce:
	./scripts/reproduce.sh $(RUN_ID)
```

### Build: rerunning any past run

`scripts/reproduce.sh` checks out the run's commit, rebuilds that image, and replays its logged config against the same data snapshot:

```bash
#!/usr/bin/env bash
set -euo pipefail
RUN_ID=$1
TAGS=$(curl -sf "http://localhost:5000/api/2.0/mlflow/runs/get?run_id=$RUN_ID" | jq '.run.data.tags')
SHA=$(echo "$TAGS" | jq -r '.[] | select(.key=="git_sha") | .value')
SNAP=$(echo "$TAGS" | jq -r '.[] | select(.key=="data.snapshot_id") | .value')
[[ "$SHA" == *-dirty ]] && { echo "run $RUN_ID came from uncommitted code; can't reproduce"; exit 1; }

WORK=$(mktemp -d); git worktree add "$WORK" "$SHA"
mkdir -p "$WORK/ml/configs/replay"
curl -sf "http://localhost:5000/get-artifact?path=config.yaml&run_uuid=$RUN_ID" > "$WORK/ml/configs/replay/$RUN_ID.yaml"
(cd "$WORK" && make train CONFIG=ml/configs/replay/$RUN_ID.yaml \
   SET="snapshot_id=$SNAP run_name=reproduce-$RUN_ID register_as=null")
git worktree remove --force "$WORK"
```

The `-dirty` check is the point: a run from uncommitted code can't be reproduced, so the tag makes that visible instead of silently wrong. If the artifact download route differs in your MLflow version, use `mlflow artifacts download --run-id $RUN_ID --artifact-path config.yaml` inside the trainer container instead.

### Build: sweeps across both GPUs

Optuna coordinates trials through a database, so two workers on two GPUs can share one study. Create an `optuna` database in `mlflow-db` (`CREATE DATABASE optuna;`). `ml/sweep.py`:

```python
import argparse, copy, os, yaml, mlflow, optuna
from ml import train_tabular

p = argparse.ArgumentParser()
p.add_argument("--config", required=True); p.add_argument("--study", required=True)
p.add_argument("--trials", type=int, default=20)
a = p.parse_args()
base = yaml.safe_load(open(a.config))

def objective(trial):
    cfg = copy.deepcopy(base)
    cfg["run_name"] = f"{a.study}-t{trial.number}"
    cfg["register_as"] = None                       # only the winner gets registered, by hand
    cfg["params"].update({
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 50, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    })
    return train_tabular.main(cfg, nested=True)    # logs a nested run under the sweep, returns test MAE

mlflow.set_experiment(base["experiment"])
study = optuna.create_study(study_name=a.study, direction="minimize", load_if_exists=True,
                            storage=f"postgresql://mlflow:{os.environ['MLFLOW_DB_PASSWORD']}@mlflow-db/optuna")
with mlflow.start_run(run_name=f"sweep-{a.study}-{os.environ.get('GPU', '?')}"):
    study.optimize(objective, n_trials=a.trials)
```

Pass `MLFLOW_DB_PASSWORD` and `GPU` into the trainer's environment. Each trial logs a nested run and returns the test MAE that `train_tabular.main` reports. One honest caveat: tuning on the test months means `test_mae` is no longer an unbiased estimate. Hold out one more month for the final check.

```bash
make ml-run GPU=0 CMD="python -m ml.sweep --config ml/configs/trip_duration.yaml --study td-oct" &
make ml-run GPU=1 CMD="python -m ml.sweep --config ml/configs/trip_duration.yaml --study td-oct"
```

### Verify

- `make train CONFIG=ml/configs/trip_duration.yaml SET="params.max_depth=10"` creates a run whose logged config shows `max_depth: 10`.
- Commit, train, then `make reproduce RUN_ID=<that run>`: the new run reads the same snapshot and gets the same `test_mae` to within GPU nondeterminism.
- The sweep shows two parent runs (one per GPU) and nested trials under each, all sharing one Optuna study. In MLflow, compare trials with the parallel-coordinates chart.

### Commit

`feat(ml): train entry point with overrides, reproduce script, two-GPU Optuna sweeps`

### Explore (optional)

- **Find the nondeterminism.** Reproduce the same run three times and compare MAE to six decimals. Then set `seed`, and XGBoost's deterministic histogram option, and try again.
- **A tiny job queue.** Write a script that reads queued configs from a folder and runs each one on whichever GPU is free (check with `nvidia-smi --query-gpu=memory.used --format=csv`). That's the core of a training scheduler.
- **Pruning.** Add Optuna's median pruner and report validation MAE every 200 boosting rounds through a callback. Count how many trials stop early.
- **Cost accounting.** Log GPU-seconds per run as a metric. SageMaker bills exactly this; seeing it changes how you design sweeps.

## Phase 19: LoRA fine-tuning a small LLM on your own lakehouse

Goal: teach a small open model to write correct Trino SQL for *your* tables. You'll build the training set from the lakehouse itself, verify every example by running it, fine-tune with LoRA on one GPU, and measure the result by **execution accuracy**: does the generated SQL return the right rows? Base model versus fine-tuned is the number that matters.

Why this task: it's small enough to train in under an hour on a 72 GB card, it uses everything from tutorial 1, and grading is objective because Trino can check every answer.

**LoRA in one paragraph.** Instead of updating all of a model's weights, LoRA freezes them and trains small low-rank matrices added to chosen layers. A rank-16 adapter for a 4-billion-parameter model is tens of megabytes, trains fast, and can be swapped in and out at serving time. **QLoRA** does the same on top of a 4-bit quantized base model, which lets a model of about 30B parameters fine-tune on one 72 GB GPU.

### Build: a verified dataset

`ml/sql_dataset.py` generates question and SQL pairs from templates, runs each against Trino, and keeps only those that succeed and return rows:

```python
import json, random, trino

conn = trino.dbapi.connect(host="trino", port=8080, user="ml", catalog="lake")
cur = conn.cursor()

def run(sql):
    cur.execute(sql); return cur.fetchall()

def schema_text(tables):
    out = []
    for t in tables:
        cols = run(f"DESCRIBE {t}")
        out.append(f"{t}(" + ", ".join(f"{c[0]} {c[1]}" for c in cols) + ")")
    return "\n".join(out)

TABLES = ["gold.daily_zone_weather", "silver.trips", "silver.zones"]
SYSTEM = ("You write Trino SQL for the catalog 'lake'. Return only SQL.\nTables:\n" + schema_text(TABLES))
boroughs = [r[0] for r in run("SELECT DISTINCT borough FROM silver.zones WHERE borough IS NOT NULL")]
months = [f"2024-{m:02d}" for m in range(1, 7)]

TEMPLATES = {
  "trips_by_borough_month": ("How many trips started in {b} in {m}?",
      "SELECT sum(trips) FROM gold.daily_zone_weather WHERE borough = '{b}' AND date_format(day, '%Y-%m') = '{m}'"),
  "rainiest_days": ("What were the 5 days with the most rain, and how many trips happened on each?",
      "SELECT day, max(precip_mm) AS rain, sum(trips) AS trips FROM gold.daily_zone_weather GROUP BY day ORDER BY rain DESC LIMIT 5"),
  "tip_ratio_rain": ("In {b}, is the average tip ratio higher on days with more than 5 mm of rain?",
      "SELECT precip_mm > 5 AS rainy, avg(avg_tip_ratio) FROM gold.daily_zone_weather WHERE borough = '{b}' GROUP BY 1"),
  # ...write 30-60 of these, covering joins, windows, top-N, date filters and NULL handling
}

rows = []
for name, (q, s) in TEMPLATES.items():
    for b in boroughs:
        for m in months:
            question, sql = q.format(b=b, m=m), s.format(b=b, m=m)
            try:
                if run(sql):
                    rows.append({"template": name, "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": sql}]})
            except Exception:
                pass                                   # broken template: fix it, don't train on it

rows = list({json.dumps(r, sort_keys=True): r for r in rows}.values())   # templates without {b}/{m} repeat
names = sorted(TEMPLATES); random.Random(7).shuffle(names)
held_out = set(names[: max(1, len(names) // 5)])          # whole templates the model never sees
for split, keep in (("train", lambda r: r["template"] not in held_out), ("eval", lambda r: r["template"] in held_out)):
    with open(f"/cache/datasets/lake_sql_{split}.jsonl", "w") as f:
        for r in filter(keep, rows):
            f.write(json.dumps(r) + "\n")
```

Holding out whole templates, not random rows, is what makes the evaluation honest. Otherwise the model can memorize near-copies of its test questions. For a richer dataset, also paraphrase each question a few ways, with a larger model you serve in vLLM, and keep only paraphrases whose SQL still validates. Create `/cache/datasets` (that is, `${MODEL_CACHE}/datasets`) before the first run.

### Build: fine-tuning

`ml/train_sft.py`:

```python
import os, mlflow, torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

def main(cfg, nested=False):
    mlflow.set_experiment(cfg["experiment"])
    ds = load_dataset("json", data_files={"train": cfg["train_file"], "eval": cfg["eval_file"]})
    model_kwargs = {"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa"}
    if cfg.get("qlora"):
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

    args = SFTConfig(
        output_dir=f"/cache/runs/{cfg['run_name']}", run_name=cfg["run_name"], report_to="mlflow",
        num_train_epochs=cfg["epochs"], learning_rate=cfg["lr"],
        per_device_train_batch_size=cfg["batch_size"], gradient_accumulation_steps=cfg["grad_accum"],
        bf16=True, gradient_checkpointing=True, max_length=cfg["max_length"],
        logging_steps=10, eval_strategy="steps", eval_steps=100, save_strategy="no",
        assistant_only_loss=True,               # learn the SQL, not the long schema prompt
        model_init_kwargs=model_kwargs)
    lora = LoraConfig(r=cfg["lora_r"], lora_alpha=2 * cfg["lora_r"], lora_dropout=0.05,
                      target_modules="all-linear", task_type="CAUSAL_LM")

    with mlflow.start_run(run_name=cfg["run_name"], nested=nested) as run:
        mlflow.set_tags({"git_sha": os.environ.get("GIT_SHA", "dev"), "base_model": cfg["base_model"]})
        mlflow.log_dict(cfg, "config.yaml")
        trainer = SFTTrainer(model=cfg["base_model"], args=args, peft_config=lora,
                             train_dataset=ds["train"], eval_dataset=ds["eval"])
        trainer.train()
        out = f"/cache/adapters/{cfg['run_name']}"
        trainer.save_model(out)                              # saves only the adapter
        AutoTokenizer.from_pretrained(cfg["base_model"]).save_pretrained(out)
        mlflow.log_artifacts(out, artifact_path="adapter")
        print("run", run.info.run_id)
```

The dataset rows include a `template` column; TRL ignores columns it doesn't use, or you can drop it with `ds.remove_columns("template")`.

`ml/configs/lake_sql_lora.yaml`:

```yaml
task: sft
experiment: lake-sql
run_name: qwen3-4b-lora-r16
base_model: Qwen/Qwen3-4B-Instruct-2507   # or the current small instruct model you prefer
train_file: /cache/datasets/lake_sql_train.jsonl
eval_file: /cache/datasets/lake_sql_eval.jsonl
epochs: 2
lr: 2.0e-4
batch_size: 8
grad_accum: 2
max_length: 2048
lora_r: 16
qlora: false
```

TRL's option names move between releases (for example `max_seq_length` became `max_length`, and masking the prompt out of the loss has had different switches). Your lockfile pins one version, so check its docs once if a name is rejected. `assistant_only_loss` also needs a chat template that marks assistant turns; if yours doesn't, train on full sequences first and compare.

### Build: execution-accuracy evaluation

`ml/eval_sql.py` generates SQL for every held-out question, runs it, and compares rows with the reference SQL:

```python
import argparse, json, mlflow, torch, trino
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

p = argparse.ArgumentParser()
p.add_argument("--base", required=True); p.add_argument("--adapter")
p.add_argument("--eval-file", default="/cache/datasets/lake_sql_eval.jsonl")
a = p.parse_args()

tok = AutoTokenizer.from_pretrained(a.base)
model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16, device_map="cuda")
if a.adapter:
    model = PeftModel.from_pretrained(model, a.adapter)
cur = trino.dbapi.connect(host="trino", port=8080, user="ml", catalog="lake").cursor()

def rows(sql):
    cur.execute(sql); return sorted(map(tuple, cur.fetchall()), key=repr)

ok = bad_sql = 0
examples = [json.loads(l) for l in open(a.eval_file)]
for ex in examples:
    prompt = tok.apply_chat_template(ex["messages"][:2], add_generation_prompt=True, return_tensors="pt").to("cuda")
    out = model.generate(prompt, max_new_tokens=256, do_sample=False)
    sql = tok.decode(out[0, prompt.shape[1]:], skip_special_tokens=True).strip().strip("`").removeprefix("sql").strip().rstrip(";")
    try:
        ok += rows(sql) == rows(ex["messages"][2]["content"])
    except Exception:
        bad_sql += 1

mlflow.set_experiment("lake-sql")
with mlflow.start_run(run_name=f"eval-{'lora' if a.adapter else 'base'}"):
    mlflow.log_params({"base": a.base, "adapter": a.adapter or "none"})
    mlflow.log_metrics({"exec_accuracy": ok / len(examples), "invalid_sql_rate": bad_sql / len(examples)})
```

If your base model emits a reasoning block before the answer, strip it before running the SQL, or pick an instruct variant without one.

### Verify

```bash
make ml-run CMD="python -m ml.sql_dataset"
make ml-run CMD="python -m ml.eval_sql --base Qwen/Qwen3-4B-Instruct-2507"                      # baseline first
make train CONFIG=ml/configs/lake_sql_lora.yaml
make ml-run CMD="python -m ml.eval_sql --base Qwen/Qwen3-4B-Instruct-2507 --adapter /cache/adapters/qwen3-4b-lora-r16"
```

- Training and eval loss curves appear live in MLflow while the job runs.
- The fine-tuned `exec_accuracy` beats the base model's on templates the model never saw. If it only wins on training templates, it memorized; add template variety.
- `ls ${MODEL_CACHE}/adapters/qwen3-4b-lora-r16` shows an adapter of tens of megabytes, not gigabytes.

### Commit

`feat(ml): verified text-to-SQL dataset, LoRA SFT task, execution-accuracy eval`

### Explore (optional)

- **Rank sweep.** Train with `lora_r` of 4, 16 and 64 through the Phase 18 sweep tooling. Plot accuracy against adapter size. Past a point, rank stops helping.
- **QLoRA on a bigger model.** Set `qlora: true` with a 14B base. Compare memory in `nvidia-smi`, speed, and accuracy against the 4B LoRA.
- **Two GPUs.** Launch the same job with `accelerate launch --num_processes 2` and both GPUs reserved. Measure the speedup and note what data parallelism actually splits.
- **Prompt versus fine-tune.** Give the base model five worked examples in its prompt (few-shot) and compare with the adapter. Sometimes the cheap fix is enough.
- **Preference tuning.** Collect pairs where one SQL runs correctly and another doesn't, and train with TRL's DPO trainer on top of your SFT adapter.

## Phase 20: Model registry and serving

Goal: promote models by name instead of by file, then use them four ways: a REST endpoint for the tabular model, Spark batch scoring into Iceberg, scoring inside the stream, and vLLM serving the base LLM with your LoRA adapter on top.

### Build: aliases in the registry

The registry keeps numbered versions of `trip_duration`. **Aliases** are movable labels on them. Everything downstream asks for `models:/trip_duration@champion`, so promoting or rolling back is one call and no redeploy of code:

```python
# ml/promote.py  — usage: python -m ml.promote trip_duration 3 champion
import sys
from mlflow import MlflowClient
name, version, alias = sys.argv[1], sys.argv[2], sys.argv[3]
MlflowClient().set_registered_model_alias(name, alias, version)
print(f"{name}@{alias} -> v{version}")
```

Use two aliases: `champion` (serving) and `challenger` (being evaluated). Phase 21 automates the switch.

### Build: a REST endpoint

Add to `compose.ml.yaml`, reusing the trainer image without a GPU:

```yaml
  trip-eta-api:
    image: lakehouse/trainer:${GIT_SHA:-dev}
    profiles: [serve]
    command: mlflow models serve -m models:/trip_duration@champion --host 0.0.0.0 --port 8000 --env-manager local
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
    ports: ["8000:8000"]
    networks: [lake]
```

```bash
curl -s localhost:8000/invocations -H 'Content-Type: application/json' -d '{
  "dataframe_split": {
    "columns": ["pu_zone_id","do_zone_id","pu_borough","do_borough","pickup_hour","pickup_dow",
                "trip_distance","passenger_count","temperature_2m","precipitation"],
    "data": [[132, 236, "Queens", "Manhattan", 17, 6, 17.4, 1, 21.0, 0.0]]}}'
```

Both serving services sit behind the serve profile, so make up-ml doesn't start them or claim a GPU; start them with ./dc --profile serve up -d trip-eta-api vllm. The alias is resolved when the server starts. After promoting a new champion, restart the container. That's the moment SageMaker would do a blue/green endpoint update; the exercise below builds one.

### Build: batch scoring with Spark

Add `mlflow-skinny` and `xgboost` to the Spark image's `requirements.in`, run `make lock`, and add `MLFLOW_TRACKING_URI: http://mlflow:5000` to the `x-spark` environment. `pipelines/jobs/ml/score_trips.py`:

```python
import argparse, sys
sys.path.insert(0, "/opt/jobs")
import mlflow
from mlflow import MlflowClient
from pyspark.sql import functions as F
from pensieve.lake.spark import get_spark

p = argparse.ArgumentParser(); p.add_argument("--month", required=True); a = p.parse_args()
spark = get_spark(f"score:trip_duration:{a.month}")
version = MlflowClient().get_model_version_by_alias("trip_duration", "champion").version
predict = mlflow.pyfunc.spark_udf(spark, f"models:/trip_duration/{version}", env_manager="local")

COLS = ["pu_zone_id", "do_zone_id", "pu_borough", "do_borough", "pickup_hour", "pickup_dow",
        "trip_distance", "passenger_count", "temperature_2m", "precipitation"]
feat = spark.table("lake.ml.trip_duration_features").where(F.col("month") == a.month)
out = (feat.withColumn("pred_duration_min", predict(F.struct(*COLS)))
           .withColumn("model_version", F.lit(int(version)))
           .select("pickup_ts", "month", *COLS, "duration_min", "pred_duration_min", "model_version"))
if spark.catalog.tableExists("lake.ml.trip_duration_predictions"):
    out.writeTo("lake.ml.trip_duration_predictions").overwritePartitions()
else:
    out.writeTo("lake.ml.trip_duration_predictions").partitionedBy(F.col("month")).create()
```

Resolving the alias to a version first, and storing it per row, means every prediction says which model made it. The Spark image runs a different Python version than the trainer; XGBoost models load fine across that gap, but MLflow will warn. Treat the warning as a reminder that environments are part of the model.

Categorical columns must arrive as the same types the model saw in training. If predictions look wrong, cast `pu_borough` and `do_borough` the same way `prep()` does, inside a pandas UDF.

### Build: scoring in the stream

The same UDF works on a streaming DataFrame. A short job joins trip events with the latest weather and zones, scores them, and publishes ETAs to a `trips.eta` topic (create it with `make topic`):

```python
# inside jobs/stream/trip_eta.py, after parsing events as in Phase 13
scored = enriched.withColumn("eta_min", predict(F.struct(*COLS)))
(scored.select(F.col("event_id").alias("key"),
               F.to_json(F.struct("event_id", "pu_zone_id", "eta_min")).alias("value"))
   .writeStream.format("kafka").option("kafka.bootstrap.servers", "kafka:9092")
   .option("topic", "trips.eta").option("checkpointLocation", "file:///data/checkpoints/trip_eta")
   .start().awaitTermination())
```

Building `enriched` (hour, weekday, borough lookups, and the most recent hourly weather) is the exercise. It's the feature-parity problem in miniature: online features must be computed exactly like the training features.

### Build: vLLM with your LoRA adapter

```yaml
  vllm:
    image: vllm/vllm-openai:${VLLM_TAG}
    profiles: [serve]
    command: >
      --model Qwen/Qwen3-4B-Instruct-2507
      --enable-lora --max-lora-rank 64
      --lora-modules lake-sql=/cache/adapters/qwen3-4b-lora-r16
      --gpu-memory-utilization 0.5 --max-model-len 8192
    environment:
      HF_HOME: /cache/hf
    volumes:
      - ${MODEL_CACHE}:/cache
    ports: ["8001:8000"]
    ipc: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["${SERVE_GPU:-0}"]
              capabilities: [gpu]
    networks: [lake]
```

Requests with `"model": "lake-sql"` use the adapter; requests for the base model name don't. One base model can serve many adapters at once. `tools/ask_lake.py` closes the loop:

```python
import json, sys, urllib.request, trino

SYSTEM = open("/app/contracts/lake_sql_system_prompt.txt").read()   # save SYSTEM from Phase 19 here
q = " ".join(sys.argv[1:])
body = {"model": "lake-sql", "temperature": 0,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]}
req = urllib.request.Request("http://vllm:8000/v1/chat/completions", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
sql = json.load(urllib.request.urlopen(req))["choices"][0]["message"]["content"].strip().strip("`")
print(sql)
cur = trino.dbapi.connect(host="trino", port=8080, user="ask", catalog="lake").cursor()
cur.execute(sql.removeprefix("sql").strip().rstrip(";"))
for row in cur.fetchall()[:20]:
    print(row)
```

Add `trino` to the tools image's requirements.in, re-lock with make lock, and rebuild.

### Verify

- `python -m ml.promote trip_duration 1 champion`, then the `curl` above returns a duration in minutes.
- `make submit JOB=ml/score_trips.py ARGS="--month 2024-06"`, then in Trino: `SELECT avg(abs(pred_duration_min - duration_min)) FROM ml.trip_duration_predictions WHERE month = '2024-06'` is close to the run's `test_mae`.
- `./dc run --rm tools python tools/ask_lake.py "Which borough had the most trips on rainy days?"` prints SQL and rows.

### Commit

`feat(ml): registry aliases, REST endpoint, Spark and streaming scoring, vLLM with LoRA`

### Explore (optional)

- **Blue/green.** Run two `trip-eta-api` containers, one on `champion` and one on `challenger`, behind a small Caddy or Traefik proxy that splits traffic 90/10. Log which version answered each request.
- **Load test.** Hit the REST endpoint with `hey` or `locust`. Then compare with Spark batch scoring on the same rows per second. Choosing between real-time and batch inference is mostly this number and the latency you need.
- **Hot-swap adapters.** Start vLLM with runtime LoRA updating enabled and load a newly trained adapter through its API without restarting the server.
- **Merge versus keep separate.** Merge the adapter into the base weights with PEFT's `merge_and_unload`, serve the merged model, and compare latency with the LoRA path. Merged is a bit faster; separate is far more flexible.

## Phase 21: ML pipelines and monitoring

Goal: close the loop. New data lands, features rebuild, a challenger trains, it's promoted only if it beats the champion, predictions are scored, and a monitor tells you when the live model drifts. This is the job SageMaker Pipelines and Model Monitor do.

### Build: the retraining pipeline

GPU training tasks run like Spark tasks did in tutorial 1: a `DockerOperator` launching the trainer image, with a GPU attached. Add to `dags/common/spark.py` (or a new `dags/common/ml.py`):

```python
import os
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import DeviceRequest, Mount

def trainer_task(task_id, cmd, gpu=None):
    return DockerOperator(
        task_id=task_id, image=f"lakehouse/trainer:{os.environ.get('TRAINER_TAG', 'dev')}",
        command=cmd, network_mode="lake", auto_remove="success", mount_tmp_dir=False,
        docker_url="unix://var/run/docker.sock", shm_size=16 * 2**30,
        device_requests=[DeviceRequest(device_ids=[gpu], capabilities=[["gpu"]])] if gpu else None,
        mounts=[Mount(source=os.environ["MODEL_CACHE"], target="/cache", type="bind")],
        environment={k: os.environ[k] for k in ("POLARIS_CREDENTIAL", "S3_ACCESS_KEY", "S3_SECRET_KEY")}
                    | {"MLFLOW_TRACKING_URI": "http://mlflow:5000", "GIT_SHA": os.environ.get("TRAINER_TAG", "dev")})
```

Pass `TRAINER_TAG`, `MODEL_CACHE` and `POLARIS_CREDENTIAL` into the Airflow container. Pin `TRAINER_TAG` to a commit you've reviewed; the pipeline trains with released code, not whatever is checked out.

`ml/compare.py` decides promotion with a rule you can read:

```python
"""Promote the challenger if it beats the champion by at least MIN_GAIN on the newest month."""
import sys, numpy as np, mlflow
from mlflow import MlflowClient
from pensieve.lake.catalog import read_table
from ml.train_tabular import prep

MIN_GAIN = 0.02                                       # 2% lower MAE, or no change
month = sys.argv[1]
c = MlflowClient()
df, _ = read_table("ml.trip_duration_features", row_filter=f"month == '{month}'")
X, y = prep(df)

def mae(alias):
    v = c.get_model_version_by_alias("trip_duration", alias)
    pred = mlflow.pyfunc.load_model(f"models:/trip_duration/{v.version}").predict(X)
    return float(np.mean(np.abs(pred - y))), v.version

champ, cv = mae("champion")
chall, nv = mae("challenger")
print(f"champion v{cv}: {champ:.3f}  challenger v{nv}: {chall:.3f}")
if chall < champ * (1 - MIN_GAIN):
    c.set_registered_model_alias("trip_duration", "champion", nv)
    print("promoted")
```

The training run must set the `challenger` alias on its new version. Add `set_alias: challenger` to the config and one line in `train_tabular.py` that applies it after registering. Both models are scored on a month neither trained on.

`dags/ml_trip_duration.py`:

```python
from datetime import datetime
from airflow.sdk import DAG
from common.spark import spark_job
from common.ml import trainer_task

MONTH = "{{ data_interval_start.strftime('%Y-%m') }}"

with DAG("ml_trip_duration", start_date=datetime(2024, 7, 1), schedule="@monthly",
         catchup=False, max_active_runs=1) as dag:
    features = spark_job("features", "ml/build_trip_features.py", f"--month {MONTH}")
    train = trainer_task("train_challenger",
        "python -m ml.train --config ml/configs/trip_duration.yaml --set run_name=monthly-" + MONTH
        + " set_alias=challenger", gpu="1")
    compare = trainer_task("compare_and_promote", f"python -m ml.compare {MONTH}")
    score = spark_job("score", "ml/score_trips.py", f"--month {MONTH}")
    features >> train >> compare >> score
```

The training config's `split` should roll forward with each run, for example "train on the last 4 months, validate on the one before this month". Make the split computed from a `--set as_of=` value rather than hardcoded. Better still, trigger this DAG from an Airflow Asset that the `tlc_monthly` silver task updates, so retraining follows the data instead of the calendar.

### Build: monitoring

Two things can go wrong after deployment: the **inputs** shift (data drift), or the model gets **worse** at its job. Trip durations become known when trips end, so you can measure the second directly.

`ml/monitor.py` runs after each scoring pass:

```python
import sys, mlflow, pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from pensieve.lake.catalog import read_table, catalog

month, ref_month = sys.argv[1], sys.argv[2]
cols = ["pickup_hour", "pickup_dow", "trip_distance", "passenger_count", "temperature_2m", "precipitation"]
cur, _ = read_table("ml.trip_duration_predictions", row_filter=f"month == '{month}'")
ref, _ = read_table("ml.trip_duration_features", row_filter=f"month == '{ref_month}'")

result = Report([DataDriftPreset()]).run(reference_data=ref[cols], current_data=cur[cols])
mae = float((cur["pred_duration_min"] - cur["duration_min"]).abs().mean())

mlflow.set_experiment("trip-duration-monitoring")
with mlflow.start_run(run_name=f"monitor-{month}"):
    mlflow.log_metric("live_mae", mae)
    result.save_html("/tmp/drift.html"); mlflow.log_artifact("/tmp/drift.html")

row = pd.DataFrame([{"month": month, "live_mae": mae, "model_version": int(cur["model_version"].iloc[0])}])
import pyarrow as pa
t = catalog().load_table("ml.model_monitoring")       # create once with a matching schema
t.append(pa.Table.from_pandas(row, preserve_index=False))
```

Evidently's API has changed between major versions; match the imports to the version your lockfile pins. Add `monitor` as a final task in the DAG.

With `ml.model_monitoring` in Iceberg, Grafana can chart `live_mae` per month through Trino, next to the streaming panels from Phase 15. Alert when live MAE is more than 20% above the training run's `test_mae`.

### Verify

- Backfill the DAG for 2024-07 and 2024-08 (download and process those months with `tlc_monthly` first). Each run creates a challenger; `compare_and_promote` prints both scores and promotes only when the margin is met.
- MLflow's model page shows the alias history: which version was champion when.
- The monitoring experiment holds a drift report per month. Open one and find the feature that moved most.

### Commit

`feat(ml): monthly retrain pipeline with gated promotion, drift and live-accuracy monitoring`

### Explore (optional)

- **Shadow deployment.** Score every month with both champion and challenger, store both, and compare live MAE for three months before promoting anything.
- **Retrain the LLM too.** Add a DAG that regenerates the SQL dataset when tables change schema, retrains the adapter, reruns execution accuracy, and hot-loads the adapter into vLLM only if accuracy didn't drop.
- **Model cards.** Have the training task write a short markdown model card (data snapshot, metrics, known weak segments) as an MLflow artifact. Make promotion fail if it's missing.

## Reproducibility checklist

Every result in this tutorial should be traceable from one MLflow run ID:

- [ ] **Code**: `git_sha` tag, never `-dirty` for anything promoted.
- [ ] **Environment**: trainer image tagged with that commit, built from a committed `uv.lock`.
- [ ] **Data**: Iceberg `data.snapshot_id` tag; snapshots kept at least as long as models that use them (set expiry in Phase 8 accordingly).
- [ ] **Config**: `config.yaml` artifact, including command-line overrides.
- [ ] **Streams**: checkpoints backed up with the catalog; producer contract versioned in `contracts/`.
- [ ] **One command**: `make reproduce RUN_ID=...` works for at least one promoted model.

## Where to go next

| Next step | What you'd learn |
| --- | --- |
| Ray (Train and Tune) on this box | Distributed training and tuning without hand-written GPU scheduling |
| A feature store (Feast) over your Iceberg tables | Serving the same features online and offline, solving the parity problem from Phase 20 |
| Kafka on k3s with Strimzi, Spark via its Kubernetes operator | Running streams and training the way most platform teams do |
| LLM evaluation with MLflow's GenAI tools | Judges, traces and regression tests for prompts and adapters |
| SageMaker for real | Replace one piece at a time: S3 first, then MLflow on SageMaker, then training jobs |

## Sources

- [Apache Kafka downloads and Docker images](https://kafka.incubator.apache.org/community/downloads/)
- [Apache Kafka 4.2 release (share groups production-ready)](https://kafka.apache.org/blog/2026/02/17/apache-kafka-4.2.0-release-announcement/)
- [Spark Real-time Mode documentation](https://apache.googlesource.com/spark/+/master/docs/streaming/real-time-mode.md)
- [MLflow tracking server and security middleware](https://mlflow.org/docs/latest/ml/tracking/server)
- [PyTorch Blackwell (sm\_120) support with CUDA 12.8](https://discuss.pytorch.org/t/rtx-5050-sm-120-support-timeline-when-in-stable-pytorch/224288/8)
- [RTX PRO 5000 Blackwell specifications](https://gpus.axiomgaming.net/gpu/rtx-pro-5000-blackwell)

