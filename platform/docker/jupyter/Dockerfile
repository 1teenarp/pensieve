ARG JUPYTER_TAG
FROM quay.io/jupyter/minimal-notebook:${JUPYTER_TAG}
# Match the client version to the server (4.1.2).
COPY requirements.lock /tmp/requirements.lock
# requirements.in: pyspark-client==4.1.2, pandas, pyarrow, grpcio, grpcio-status, protobuf, googleapis-common-protos
RUN pip install --no-cache-dir -r /tmp/requirements.lock
