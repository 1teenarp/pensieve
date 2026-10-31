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
	envsubst '$${EXECUTOR_CORES} $${EXECUTOR_MEMORY} $${APP_MAX_CORES} $${HOST_NAME}' < $< > $@
	printf 'SPARK_DEFAULTS_SHAPE=%s\n' '$(SHAPE)' > $@.shape

# usage: make submit JOB=ingest/tlc_bronze.py ARGS="--month 2024-01"
submit:
	$(DC) run --rm spark-client lake-submit $(SUBMIT_OPTS) /opt/jobs/$(JOB) $(ARGS)

# placeholder until Phase 3 moves the driver-IP lookup inside the container
sql:
	$(DC) run --rm spark-client /opt/spark/bin/spark-sql --conf spark.driver.host=$$(hostname -i)
