# conf is a recipe-less target in make/core.mk; each stack's .mk adds its rendered files here.
SPARK_DEFAULTS := platform/conf/spark/spark-defaults.conf

# make tracks file mtimes, not variables: switching SHAPE alone would leave the previous
# shape's render in place. The stamp records the shape the conf was rendered from;
# on a mismatch the target is marked phony, forcing exactly one re-render.
-include $(SPARK_DEFAULTS).shape
ifneq ($(SPARK_DEFAULTS_SHAPE),$(SHAPE))
.PHONY: $(SPARK_DEFAULTS)
endif

conf: $(SPARK_DEFAULTS)

$(SPARK_DEFAULTS): platform/conf/spark/spark-defaults.conf.template $(wildcard .env) shapes/$(SHAPE).env
	envsubst '$${EXECUTOR_CORES} $${EXECUTOR_MEMORY} $${EXECUTOR_MEMORY_OVERHEAD} $${SPARK_SHUFFLE_PARTITIONS} $${APP_MAX_CORES} $${HOST_NAME} $${POLARIS_CLIENT_ID} $${POLARIS_CLIENT_SECRET} $${S3_ACCESS_KEY} $${S3_SECRET_KEY}' < $< > $@
	printf 'SPARK_DEFAULTS_SHAPE=%s\n' '$(SHAPE)' > $@.shape

# usage: make submit JOB=ingest/tlc_bronze.py ARGS="--month 2024-01"
submit:
	$(DC) run --rm spark-client lake-submit $(SUBMIT_OPTS) /opt/jobs/$(JOB) $(ARGS)

# NUMA-pinned workers: regenerates platform/compose.numa.yaml (gitignored), then
# brings up the core plus the `numa` profile with the unpinned workers scaled to 0.
up-numa: net conf
	python3 scripts/gen_numa_workers.py 4
	STACKS="$(CORE) numa" $(DC) --profile numa up -d --build --scale spark-worker=0

# Remove the NUMA-pinned workers again. `down` on the numa file alone fails
# ("depends on undefined service spark-master"): the base spark-worker's
# depends_on is inherited through `extends`, so that file is not a valid
# project by itself. `rm` of just the n-services avoids taking the core down.
down-numa:
	STACKS="$(CORE) numa" $(DC) --profile numa rm -sf spark-worker-n0 spark-worker-n1 spark-worker-n2 spark-worker-n3

# the driver IP must be read inside the container: a container hostname is often not
# resolvable by other containers, and the host cannot see the container's IP
sql:
	$(DC) run --rm spark-client sh -c 'spark-sql --conf spark.driver.host=$$(hostname -i)'
