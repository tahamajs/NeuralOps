.PHONY: lint test coverage build clean

PYTHON ?= python3
PYTEST ?= pytest
RUFF ?= ruff
MYPY ?= mypy

lint:
	$(RUFF) check neuralops/ tests/
	$(RUFF) format --check neuralops/ tests/
	$(MYPY) --strict neuralops/benchmarks/config_v2/ neuralops/benchmarks/suites_v2/ neuralops/benchmarks/cli_v2/ neuralops/benchmarks/results_v2/ 2>&1 || true

test:
	$(PYTEST) tests/unit tests/contract -v

coverage:
	$(PYTEST) --cov=neuralops --cov-config=.coveragerc tests/unit tests/contract --cov-report=term-missing --cov-report=html

build:
	$(PYTHON) -m build

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage htmlcov
