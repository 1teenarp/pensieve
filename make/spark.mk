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
	envsubst '$${EXECUTOR_CORES} $${EXECUTOR_MEMORY} $${APP_MAX_CORES} $${HOST_NAME} $${POLARIS_CLIENT_ID} $${POLARIS_CLIENT_SECRET} $${S3_ACCESS_KEY} $${S3_SECRET_KEY}' < $< > $@
	printf 'SPARK_DEFAULTS_SHAPE=%s\n' '$(SHAPE)' > $@.shape

# usage: make submit JOB=ingest/tlc_bronze.py ARGS="--month 2024-01"
submit:
	$(DC) run --rm spark-client lake-submit $(SUBMIT_OPTS) /opt/jobs/$(JOB) $(ARGS)

# the driver IP must be read inside the container: a container hostname is often not
# resolvable by other containers, and the host cannot see the container's IP
sql:
	$(DC) run --rm spark-client sh -c 'spark-sql --conf spark.driver.host=$$(hostname -i)'
