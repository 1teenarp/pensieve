conf:                         # no recipe here; each stack's .mk adds its rendered files as prerequisites
.PHONY: conf

lock:                         # regenerate every image's requirements.lock
	for d in platform/docker/*/; do \
	  [ -f $$d/requirements.in ] && uv pip compile $$d/requirements.in -o $$d/requirements.lock \
	    --python-version $$(cat $$d/python-version); \
	done
