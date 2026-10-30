conf:
	@echo "nothing to render yet"

lock:                         # regenerate every image's requirements.lock
	for d in platform/docker/*/; do \
	  [ -f $$d/requirements.in ] && uv pip compile $$d/requirements.in -o $$d/requirements.lock \
	    --python-version $$(cat $$d/python-version); \
	done
