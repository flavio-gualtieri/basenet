# Reproduce everything from scratch:
#
#   make install     # editable install + test deps
#   make data        # ingest -> pa -> features -> matchups -> validate
#   make train       # one run (seed 42) -> runs/summary.json, runs/training_curve.png
#   make seeds       # 5 seeds -> runs/seeds/seeds_summary.json (the README numbers)
#   make test        # unit + leakage tests on a synthetic pipeline (no network)
#
# `make data` is the only target that touches the network (Baseball Savant via
# pybaseball, MLB StatsAPI), and only in `ingest`; every later stage is a pure
# function of data/raw/. ingest is idempotent: seasons already on disk are
# skipped, so an interrupted download can just be re-run.
#
# Override on the command line, e.g. `make data SEASONS="2023 2024 2025"` or
# `make train PYTHON=~/miniforge3/envs/basenet-env/bin/python`.

PYTHON  ?= python
SEASONS ?= 2021 2022 2023 2024 2025

.PHONY: install data ingest pa features matchups validate train seeds test clean-runs

install:
	$(PYTHON) -m pip install -e ".[dev]"

data: ingest pa features matchups validate

ingest:
	$(PYTHON) -m basenet.data.ingest --seasons $(SEASONS)

pa:
	$(PYTHON) -m basenet.data.pa --seasons $(SEASONS)

features:
	$(PYTHON) -m basenet.data.features --seasons $(SEASONS)

matchups:
	$(PYTHON) -m basenet.data.matchups --seasons $(SEASONS)

validate:
	$(PYTHON) -m basenet.data.validate --seasons $(SEASONS)

train:
	$(PYTHON) -m basenet.train

seeds:
	$(PYTHON) -m basenet.seeds

test:
	$(PYTHON) -m pytest -q

clean-runs:
	rm -rf runs/
