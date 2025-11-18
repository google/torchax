# TorchAx Development Makefile
# 
# This Makefile provides convenient commands for development.
# All commands use uv for fast, reliable dependency management.

.PHONY: help install dev test test-all lint format clean build docs

# Default target
help:
	@echo "TorchAx Development Commands"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install          Install package with CPU backend"
	@echo "  make dev              Install package with all dev dependencies"
	@echo "  make install-cuda     Install with CUDA backend"
	@echo "  make install-tpu      Install with TPU backend"
	@echo ""
	@echo "Development:"
	@echo "  make lint             Run linters (ruff check)"
	@echo "  make format           Format code with ruff"
	@echo "  make test             Run unit tests (file-by-file like CI)"
	@echo "  make test-fast        Run unit tests (parallel, faster)"
	@echo "  make test-all         Run all tests (unit + distributed + tutorials)"
	@echo "  make test-gemma       Run gemma tests"
	@echo ""
	@echo "Cleaning:"
	@echo "  make clean            Clean build artifacts and caches"
	@echo "  make clean-all        Deep clean including Python caches"
	@echo ""
	@echo "Building:"
	@echo "  make build            Build package"
	@echo "  make docs             Build documentation"

# === Installation ===

install:
	uv pip install -e ".[cpu]"

dev:
	uv pip install -e ".[cpu,dev,test,docs]"

install-cuda:
	uv pip install -e ".[cuda,dev,test]"

install-tpu:
	uv pip install -e ".[tpu,dev,test]"

# === Linting & Formatting ===

lint:
	@echo "Running ruff check..."
	@uv run ruff check torchax test test_dist
	@echo "✓ Linting passed!"

format:
	@echo "Formatting code with ruff..."
	@uv run ruff check torchax test test_dist --fix
	@uv run ruff format torchax test test_dist
	@echo "✓ Code formatted!"

# === Testing ===

test:
	@echo "Running unit tests..."
	@export JAX_PLATFORMS=cpu && \
	find ./test -name "test_*.py" -type f | while IFS= read -r test_file; do \
		echo "Running tests in $$test_file"; \
		uv run --frozen pytest "$$test_file" -v --tb=short || exit 1; \
	done
	@echo "✓ Unit tests completed!"

test-fast:
	@echo "Running unit tests (parallel)..."
	@JAX_PLATFORMS=cpu uv run --frozen pytest test/ -v --tb=short -n auto

test-all:
	@echo "Running unit tests..."
	@$(MAKE) test
	@echo ""
	@echo "Running distributed tests..."
	@XLA_FLAGS=--xla_force_host_platform_device_count=4 uv run --frozen pytest test_dist/ -n 0
	@echo ""
	@echo "Running tutorial tests..."
	@export JAX_PLATFORMS=cpu && \
	for file in $$(find docs/docs/tutorials -name '*.py' 2>/dev/null || true); do \
		if [ -f "$$file" ]; then \
			echo "Testing $$file"; \
			uv run --frozen python "$$file" || exit 1; \
		fi \
	done
	@echo "✓ All tests completed!"

test-coverage:
	@echo "Running tests with coverage..."
	@JAX_PLATFORMS=cpu uv run --frozen pytest test/ --cov=torchax --cov-report=html --cov-report=term

# === Cleaning ===

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf build/ dist/ *.egg-info
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Clean complete!"

clean-all: clean
	@echo "Deep cleaning..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@rm -rf .coverage htmlcov/
	@echo "✓ Deep clean complete!"

# === Building ===

build: clean
	@echo "Building package..."
	@uv build
	@echo "✓ Build complete!"

docs:
	@echo "Building documentation..."
	@cd docs && uv run mkdocs build
	@echo "✓ Documentation built! Open docs/site/index.html"

docs-serve:
	@echo "Serving documentation at http://127.0.0.1:8000"
	@cd docs && uv run mkdocs serve

# === CI Simulation ===

ci: lint test
	@echo "✓ CI checks passed!"

# === Utilities ===

check-env:
	@echo "Python: $$(python --version)"
	@echo "uv: $$(uv --version)"
	@python -c "import torch; print(f'PyTorch: {torch.__version__}')"
	@python -c "import jax; print(f'JAX: {jax.__version__}')"
	@python -c "import torchax; print(f'TorchAx: {torchax.__version__}')"

