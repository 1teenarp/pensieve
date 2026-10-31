import argparse
import sys

sys.path.insert(0, "/opt/jobs")
from pensieve.lake.spark import get_spark
from pyspark.sql import functions as F

p = argparse.ArgumentParser()
p.add_argument("--month", required=True)
a = p.parse_args()
spark = get_spark(f"trips_silver:{a.month}")

spark.sql("CREATE NAMESPACE IF NOT EXISTS lake.silver")  # the tutorial omits this; without it the first run fails (see journal)
spark.sql("""CREATE TABLE IF NOT EXISTS lake.silver.trips (
  vendor_id INT, pickup_ts TIMESTAMP, dropoff_ts TIMESTAMP, passenger_count INT,
  trip_distance DOUBLE, pu_zone_id INT, do_zone_id INT, payment_type INT,
  fare_amount DOUBLE, tip_amount DOUBLE, total_amount DOUBLE)
  USING iceberg PARTITIONED BY (days(pickup_ts))""")

trips = (spark.table("lake.bronze.tlc_yellow").where(F.col("_ingest_month") == a.month)
         .where(F.date_format("tpep_pickup_datetime", "yyyy-MM") == a.month)  # drop mis-dated rows
         .where("trip_distance > 0 AND total_amount >= 0 AND tpep_dropoff_datetime > tpep_pickup_datetime")
         .select(F.col("vendorid").alias("vendor_id"), F.col("tpep_pickup_datetime").alias("pickup_ts"),
                 F.col("tpep_dropoff_datetime").alias("dropoff_ts"), F.col("passenger_count").cast("int"),
                 "trip_distance", F.col("pulocationid").alias("pu_zone_id"),
                 F.col("dolocationid").alias("do_zone_id"), F.col("payment_type").cast("int"),
                 "fare_amount", "tip_amount", "total_amount")
         .dropDuplicates())
trips.writeTo("lake.silver.trips").overwritePartitions()  # replaces that month's days only
spark.stop()
