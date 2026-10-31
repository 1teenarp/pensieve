# Trino for interactive SQL on the lake catalog (query stack).
# conf is a recipe-less target in make/core.mk; this stack renders nothing.

trino:
	$(DC) exec trino trino --catalog lake
