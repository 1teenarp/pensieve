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
# requirements.in: pyarrow, pandas, requests — regenerate the lockfile with `make lock`
COPY requirements.lock /tmp/requirements.lock
# newer pip needs the flag, older pip rejects it
RUN pip3 install --no-cache-dir -r /tmp/requirements.lock || \
    pip3 install --no-cache-dir --break-system-packages -r /tmp/requirements.lock
COPY lake-submit /opt/spark/bin/lake-submit
RUN chmod +x /opt/spark/bin/lake-submit && mkdir -p /opt/spark/events /scratch && \
    chown -R spark:spark /opt/spark/events /scratch
# libs/ is mounted at /opt/libs, so `import pensieve` works
ENV PATH=/opt/spark/bin:$PATH PYTHONPATH=/opt/libs
USER spark
