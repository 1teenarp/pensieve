SHAPE ?= small
include versions.env
-include .env
include shapes/$(SHAPE).env
export

DC     := ./dc
CORE   := storage catalog spark query sources
STACKS ?= $(CORE)
SCALE   = $(if $(filter spark,$(STACKS)),--scale spark-worker=$(SPARK_WORKERS))

net:
	docker network inspect lake >/dev/null 2>&1 || docker network create lake

up: net conf                  # starts only the selected stacks (default: the core)
	STACKS="$(STACKS)" $(DC) up -d --build $(SCALE)

up-with:                      # usage: make up-with S="notebook airflow"
	$(MAKE) up STACKS="$(CORE) $(S)"

up-stream: ; $(MAKE) up-with S=stream        # tutorial 2
up-ml:     ; $(MAKE) up-with S=ml            # tutorial 2
up-dl:     ; $(MAKE) up-with S="ml dl"       # tutorial 3

down:                         # stops the selected stacks
	STACKS="$(STACKS)" $(DC) down
down-all:
	$(DC) down
ps:
	$(DC) ps
logs:
	$(DC) logs -f $(S)
versions-check:               # fails while any version is still a placeholder
	@! grep -n '<' versions.env

-include make/*.mk
