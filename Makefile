PYTHON ?= python
CONFIG ?= config/parameters.yaml

.PHONY: prepare train unlearn eval pipeline

prepare:
	$(PYTHON) scripts/prepare_data.py --config $(CONFIG)

train:
	$(PYTHON) scripts/train_vlm.py --config $(CONFIG)

unlearn:
	$(PYTHON) scripts/run_unlearning.py --config $(CONFIG)

eval:
	$(PYTHON) scripts/evaluate_attacks.py --config $(CONFIG)

pipeline:
	$(PYTHON) scripts/run_pipeline.py --config $(CONFIG)
