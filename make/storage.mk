# conf is a recipe-less target in make/core.mk; each stack's .mk adds its rendered files here.
S3_JSON := platform/conf/seaweedfs/s3.json

conf: $(S3_JSON)

$(S3_JSON): platform/conf/seaweedfs/s3.json.template $(wildcard .env)
	envsubst '$${S3_ACCESS_KEY} $${S3_SECRET_KEY}' < $< > $@

buckets:
	for b in $(WAREHOUSE_BUCKET) landing; do \
	  echo "s3.bucket.create -name $$b" | $(DC) exec -T seaweedfs weed shell -master=localhost:9333; \
	done

# usage: make s3 ARGS="ls s3://warehouse/ --recursive"
s3:
	docker run --rm --network lake \
	  -e AWS_ACCESS_KEY_ID=$(S3_ACCESS_KEY) -e AWS_SECRET_ACCESS_KEY=$(S3_SECRET_KEY) \
	  -e AWS_DEFAULT_REGION=us-east-1 -v $(PWD):/work -w /work \
	  amazon/aws-cli:$(AWS_CLI_TAG) --endpoint-url http://seaweedfs:8333 s3 $(ARGS)
