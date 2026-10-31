from pyspark.sql import SparkSession


def get_spark(app_name: str) -> SparkSession:
    # Cluster, catalog and S3 settings come from spark-defaults.conf.
    return SparkSession.builder.appName(app_name).getOrCreate()
