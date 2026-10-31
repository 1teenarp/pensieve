import argparse
import sys

import pandas as pd
import requests

sys.path.insert(0, "/opt/jobs")
from pensieve.lake.spark import get_spark
from pyspark.sql import functions as F

p = argparse.ArgumentParser()
p.add_argument("--start", required=True)  # 2024-01-01
p.add_argument("--end", required=True)    # 2024-12-31
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
spark.stop()
