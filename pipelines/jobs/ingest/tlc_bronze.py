import argparse
import sys

sys.path.insert(0, "/opt/jobs")
from pensieve.lake.spark import get_spark
from pyspark.sql import functions as F

# Raw as received, cast to one canonical schema: the monthly files' column names
# change case and some types drift, so every file is mapped onto these names.
CANONICAL = {
    "vendorid": "int",
    "tpep_pickup_datetime": "timestamp",
    "tpep_dropoff_datetime": "timestamp",
    "passenger_count": "double",
    "trip_distance": "double",
    "ratecodeid": "double",
    "store_and_fwd_flag": "string",
    "pulocationid": "int",
    "dolocationid": "int",
    "payment_type": "bigint",
    "fare_amount": "double",
    "extra": "double",
    "mta_tax": "double",
    "tip_amount": "double",
    "tolls_amount": "double",
    "improvement_surcharge": "double",
    "total_amount": "double",
    "congestion_surcharge": "double",
    "airport_fee": "double",
}

p = argparse.ArgumentParser()
p.add_argument("--month", required=True)  # e.g. 2024-01
p.add_argument("--landing", default="file:///data/landing/tlc/yellow")
a = p.parse_args()

spark = get_spark(f"tlc_bronze:{a.month}")
raw = spark.read.parquet(f"{a.landing}/yellow_tripdata_{a.month}.parquet")
actual = {c.lower(): c for c in raw.columns}
df = raw.select(
    *[(F.col(actual[n]) if n in actual else F.lit(None)).cast(t).alias(n)
      for n, t in CANONICAL.items()],
    F.col("_metadata.file_path").alias("_source_file"),
).withColumn("_ingest_month", F.lit(a.month)).withColumn("_ingested_at", F.current_timestamp())

spark.sql("CREATE NAMESPACE IF NOT EXISTS lake.bronze")
if spark.catalog.tableExists("lake.bronze.tlc_yellow"):
    df.writeTo("lake.bronze.tlc_yellow").overwritePartitions()  # rerun = same result
else:
    df.writeTo("lake.bronze.tlc_yellow").partitionedBy(F.col("_ingest_month")).create()
spark.stop()
