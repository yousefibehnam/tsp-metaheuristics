PYTHON ?= python3
VENV   ?= .venv
BIN     = $(VENV)/bin
export PYTHONUNBUFFERED = 1  # progress lines appear immediately, even when output goes to a log

.PHONY: install test lint data experiment clean

install:          ## create a virtual environment and install the package
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install -q -r requirements.txt
	$(BIN)/pip install -q -e . --no-deps

test:
	$(BIN)/pytest

lint:
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests
	$(BIN)/shellcheck scripts/*.sh

data:             ## download the four TSPLIB instances and their optimal tours (SHA-256 verified)
	scripts/download_data.sh

experiment: data  ## 10 seeds x 4 instances x all algorithms -> results/ (log in data/logs/)
	mkdir -p data/logs
	$(BIN)/tspmh experiment --out results 2>&1 | tee data/logs/experiment.log

clean:
	rm -rf data/logs .pytest_cache .ruff_cache
