# conf is a recipe-less target in make/core.mk; this stack renders nothing.
# polaris-bootstrap runs a one-time setup (`profiles: [setup]`); `compose run` enables
# the target service's profile automatically and starts its healthy-gated dependency.

catalog-bootstrap:
	$(DC) run --rm polaris-bootstrap

catalog-create:                  # waits for Polaris health, then creates the `lake` catalog
	until curl -sf localhost:8182/q/health >/dev/null; do sleep 2; done
	./scripts/create-catalog.sh
