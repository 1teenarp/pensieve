import argparse
import sys

sys.path.insert(0, "/opt/jobs")
from pensieve.lake.spark import get_spark

p = argparse.ArgumentParser()
p.add_argument("--table", required=True)
args = p.parse_args()

spark = get_spark(f"row_count:{args.table}")
print(args.table, spark.table(args.table).count())
spark.stop()
