import argparse
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, "/opt/jobs")
from pensieve.lake.spark import get_spark

p = argparse.ArgumentParser()
p.add_argument("--table", required=True)  # e.g. silver.trips
p.add_argument("--keep-days", type=int, default=7)
p.add_argument("--target-mb", type=int, default=512)
a = p.parse_args()

spark = get_spark(f"optimize:{a.table}")
cutoff = (datetime.now(timezone.utc) - timedelta(days=a.keep_days)).strftime("%Y-%m-%d %H:%M:%S")  # noqa: UP017  # datetime.UTC needs Python 3.11; the image has 3.10
t = f"'{a.table}'"
print(f"optimize {a.table}: keep {a.keep_days} days (cutoff {cutoff}), target {a.target_mb} MB")

# 1. Merge small data files into larger ones.
spark.sql(f"CALL lake.system.rewrite_data_files(table => {t}, "
          f"options => map('target-file-size-bytes', '{a.target_mb * 1024 * 1024}'))").show()
# 2. Merge small manifests so planning reads fewer metadata files.
spark.sql(f"CALL lake.system.rewrite_manifests({t})").show()
# 3. Drop old snapshots (and the files only they reference). Time travel ends here.
spark.sql(f"CALL lake.system.expire_snapshots(table => {t}, "
          f"older_than => TIMESTAMP '{cutoff}', retain_last => 5)").show()
# 4. Delete files no snapshot references, e.g. from failed writes.
# prefix_listing: list via the table's FileIO (S3FileIO) instead of Hadoop's FileSystem,
# which has no driver for the s3:// scheme (UnsupportedFileSystemException, see journal).
spark.sql(f"CALL lake.system.remove_orphan_files(table => {t}, "
          f"older_than => TIMESTAMP '{cutoff}', prefix_listing => true)").show()
spark.stop()
