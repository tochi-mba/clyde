.DEFAULT_GOAL := help
UV ?= uv

.PHONY: help install fmt lint type imports test cov check live run clean

help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create the virtualenv and install everything
	$(UV) sync --group dev

fmt: ## Format the codebase
	$(UV) run python -m ruff format .
	$(UV) run python -m ruff check --fix .

lint: ## Lint (no fixes)
	$(UV) run python -m ruff format --check .
	$(UV) run python -m ruff check .

type: ## Strict type check
	$(UV) run python -m mypy

# import-linter ships only a console script, and a Windows Application Control policy refuses
# generated `.venv/Scripts` shims, so it is reached through its own API instead.
imports: ## Enforce the architectural layering contracts
	$(UV) run python -c "from importlinter import configuration; from importlinter.application.use_cases import lint_imports; configuration.configure(); raise SystemExit(0 if lint_imports(config_filename=None, is_debug_mode=False, verbose=False) else 1)"

test: ## Run the test suite with 100% branch coverage enforced
	$(UV) run python -m pytest --cov --cov-report=term-missing

cov: ## Write an HTML coverage report to htmlcov/
	$(UV) run python -m pytest --cov --cov-report=html

check: lint type imports test ## Everything CI runs, on one interpreter

# Spawns a real `claude`, so it costs money and needs a login. Deliberately outside `check`.
live: ## Run the tests that call the real CLI
	$(UV) run python -m pytest -m live --no-cov

run: ## Serve on :8127 with reload
	$(UV) run python -m uvicorn clyde.api.app:create_app --factory --reload --port 8127

clean: ## Remove caches and build output
	rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis htmlcov .coverage build dist
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
