#!/usr/bin/env python3
"""Generate compose.numa.yaml: one Spark worker pinned to each NUMA node.

Executors inherit the worker's NUMA policy, so pinning the worker is enough.
Run from the repo root (the Makefile target does): the file is a generated
artifact and gitignored; `make up-numa` regenerates it on every run.
"""
import sys

nodes = int(sys.argv[1]) if len(sys.argv) > 1 else 4
out = ["services:"]
for n in range(nodes):
    out += [
        f"  spark-worker-n{n}:",
        "    extends: { file: compose.spark.yaml, service: spark-worker }",
        "    profiles: [numa]",
        "    cap_add: [SYS_NICE]    # Docker's seccomp profile blocks NUMA memory binding without it",
        f'    ports: ["{8101 + n}:8081"]',
        "    command: >",
        f"      numactl --cpunodebind={n} --membind={n}",
        "      /opt/spark/bin/spark-class org.apache.spark.deploy.worker.Worker",
        "      --cores ${SPARK_WORKER_CORES} --memory ${SPARK_WORKER_MEMORY} spark://spark-master:7077",
    ]
out += ["networks:", "  lake:", "    external: true"]
with open("platform/compose.numa.yaml", "w") as f:
    f.write("\n".join(out) + "\n")
