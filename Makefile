.PHONY: install generate pipeline reproduce test format lint

SEED ?= 42
ROWS ?= 600
INPUT ?= data/raw/lab_measurements.csv
K ?= 50

install:
	poetry install

generate:
	poetry run python -m labsentinel.generator --seed $(SEED) --rows $(ROWS) --out $(INPUT)

pipeline:
	poetry run python -m labsentinel.pipeline --input $(INPUT) --seed $(SEED) --k $(K)

reproduce: generate pipeline

test:
	poetry run pytest

format:
	poetry run black src tests

lint:
	poetry run ruff check src tests