UV ?= uv
CONFIG ?= config/parameters.yaml

.PHONY: help sync check test show-config prepare pipeline baseline unlearn evaluate interface-local modal-prepare modal-deploy

help:
	@printf '%s\n' 'Targets: sync check test show-config prepare pipeline baseline unlearn evaluate interface-local modal-prepare modal-deploy'

sync:
	$(UV) sync --locked

check:
	$(UV) run --locked ruff check .

test:
	$(UV) run --locked pytest -q

show-config:
	$(UV) run --locked python scripts/run_pipeline.py --config $(CONFIG) --show-config

prepare:
	$(UV) run --locked python scripts/prepare_data.py --config $(CONFIG)

pipeline:
	$(UV) run --locked python scripts/run_pipeline.py --config $(CONFIG)

baseline:
	$(UV) run --locked python scripts/train_canonical_baseline.py

unlearn:
	$(UV) run --locked python scripts/run_unlearning.py --config $(CONFIG)

evaluate:
	$(UV) run --locked python scripts/evaluate_attacks.py --config $(CONFIG)

interface-local:
	$(UV) run --locked python scripts/run_interface.py --offline --device cpu --output-root outputs

modal-prepare:
	$(UV) run --locked --group modal modal run worker/modal_app.py::prepare_assets

modal-deploy:
	$(UV) run --locked --group modal modal deploy worker/modal_app.py
