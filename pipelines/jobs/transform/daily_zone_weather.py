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
spark.stop()
