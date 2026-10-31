import os
import sys

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
spark.stop()
