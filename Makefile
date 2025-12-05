# TorchAx Development Makefile

# Colors for output
ORANGE := \033[0;33m
GREEN := \033[0;32m
RESET := \033[0m

# UV binary - use system uv if available, otherwise use local
UV := $(shell command -v uv 2>/dev/null || echo "./.local/bin/uv")

# Prefer managed Python from uv
export UV_PYTHON_PREFERENCE := only-managed

.PHONY: help install install-test lint lint-check format test test-fast test-all clean clean-all build docs

# Default target
help:
	@echo "$(ORANGE)TorchAx Development Commands$(RESET)"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install-uv       Install uv locally (if not in system)"
	@echo "  make install          Install for development (flexible versions)"
	@echo "  make install-test     Install with pinned test versions (like CI)"
	@echo "  make install-cuda     Install with CUDA backend"
	@echo "  make install-tpu      Install with TPU backend"
	@echo ""
	@echo "Development:"
	@echo "  make lint             Auto-fix and format code"
	@echo "  make lint-check       Check code without modifying"
	@echo "  make format           Format code only"
	@echo "  make test             Run unit tests (file-by-file like CI)"
	@echo "  make test-fast        Run unit tests (parallel)"
	@echo "  make test-all         Run all tests"
	@echo ""
	@echo "Cleaning:"
	@echo "  make clean            Clean build artifacts"
	@echo "  make clean-all        Deep clean"
	@echo ""
	@echo "Building:"
	@echo "  make build            Build package"
	@echo "  make docs             Build documentation"

# === Ensure uv is available ===

.local/bin/uv:
	@echo "$(ORANGE)Installing uv locally to .local/bin/...$(RESET)"
	@mkdir -p .local/bin
	@curl -LsSf https://astral.sh/uv/install.sh | sh
	@if [ -f ~/.local/bin/uv ]; then \
		cp ~/.local/bin/uv .local/bin/uv; \
		echo "$(GREEN)✓ uv installed to .local/bin/uv$(RESET)"; \
	elif [ -f ~/.cargo/bin/uv ]; then \
		cp ~/.cargo/bin/uv .local/bin/uv; \
		echo "$(GREEN)✓ uv installed to .local/bin/uv$(RESET)"; \
	else \
		echo "Error: uv installation failed - check if it's in your PATH"; \
		exit 1; \
	fi

install-uv: .local/bin/uv
	@echo "$(GREEN)✓ uv is ready at .local/bin/uv$(RESET)"

# === Installation ===

install:
	@echo "$(ORANGE)Installing for development (flexible versions)...$(RESET)"
	@$(UV) pip install -e ".[cpu,dev,docs]"
	@echo "$(GREEN)✓ Installation complete!$(RESET)"

install-test:
	@echo "$(ORANGE)Installing with pinned test versions (like CI)...$(RESET)"
	@$(UV) pip install -e ".[test,dev,docs]" --extra-index-url https://download.pytorch.org/whl/cpu
	@echo "$(GREEN)✓ Installation complete!$(RESET)"

install-cuda:
	@$(UV) pip install -e ".[cuda,dev,test]"

install-tpu:
	@$(UV) pip install -e ".[tpu,dev,test]"

# === Linting & Formatting ===

lint:
	@echo "$(ORANGE)1. ==== Ruff format ====$(RESET)"
	@$(UV) tool run ruff format torchax test test_dist
	@echo "$(ORANGE)2. ==== Ruff check & fix ====$(RESET)"
	@$(UV) tool run ruff check torchax test test_dist --fix
	@echo "$(GREEN)✓ Code formatted and linted!$(RESET)"

lint-check:
	@echo "$(ORANGE)1. ==== Ruff format check ====$(RESET)"
	@$(UV) tool run ruff format --check torchax test test_dist
	@echo "$(ORANGE)2. ==== Ruff check ====$(RESET)"
	@$(UV) tool run ruff check torchax test test_dist
	@echo "$(GREEN)✓ Linting passed!$(RESET)"

format:
	@echo "$(ORANGE)Formatting code with ruff...$(RESET)"
	@$(UV) tool run ruff format torchax test test_dist
	@echo "$(GREEN)✓ Code formatted!$(RESET)"

# === Testing ===

test:
	@echo "$(ORANGE)Running unit tests...$(RESET)"
	@export JAX_PLATFORMS=cpu && \
	find ./test -name "test_*.py" -type f | while IFS= read -r test_file; do \
		echo "Running tests for $$test_file"; \
		pytest "$$test_file" || exit 1; \
	done
	@echo "$(GREEN)✓ Unit tests completed!$(RESET)"

test-fast:
	@echo "$(ORANGE)Running unit tests (parallel)...$(RESET)"
	@JAX_PLATFORMS=cpu pytest test/ -v --tb=short -n auto
	@echo "$(GREEN)✓ Tests completed!$(RESET)"

test-all:
	@echo "$(ORANGE)Running all tests...$(RESET)"
	@$(MAKE) test
	@echo ""
	@echo "$(ORANGE)Running distributed tests...$(RESET)"
	@XLA_FLAGS=--xla_force_host_platform_device_count=4 pytest test_dist/ -n 0
	@echo ""
	@echo "$(ORANGE)Running tutorial tests...$(RESET)"
	@export JAX_PLATFORMS=cpu && \
	for file in $$(find docs/docs/tutorials -name '*.py' 2>/dev/null || true); do \
		if [ -f "$$file" ]; then \
			echo "Testing $$file"; \
			python "$$file" || exit 1; \
		fi \
	done
	@echo "$(GREEN)✓ All tests completed!$(RESET)"

test-coverage:
	@echo "$(ORANGE)Running tests with coverage...$(RESET)"
	@JAX_PLATFORMS=cpu pytest test/ --cov=torchax --cov-report=html --cov-report=term

# === Cleaning ===

clean:
	@echo "$(ORANGE)Cleaning build artifacts...$(RESET)"
	@rm -rf build/ dist/ *.egg-info
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Clean complete!$(RESET)"

clean-all: clean
	@echo "$(ORANGE)Deep cleaning...$(RESET)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@rm -rf .coverage htmlcov/
	@echo "$(GREEN)✓ Deep clean complete!$(RESET)"

# === Building ===

build: clean
	@echo "$(ORANGE)Building package...$(RESET)"
	@$(UV) build
	@echo "$(GREEN)✓ Build complete!$(RESET)"

docs:
	@echo "$(ORANGE)Building documentation...$(RESET)"
	@cd docs && $(UV) run mkdocs build
	@echo "$(GREEN)✓ Documentation built!$(RESET)"

docs-serve:
	@echo "$(ORANGE)Serving documentation at http://127.0.0.1:8000$(RESET)"
	@cd docs && $(UV) run mkdocs serve

# === CI Simulation ===

ci: lint-check test
	@echo "$(GREEN)✓ CI checks passed!$(RESET)"

# === Utilities ===

check-env:
	@echo "Python: $$(python --version)"
	@echo "uv: $$($(UV) --version 2>/dev/null || echo 'not available')"
	@python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || echo "PyTorch: not installed"
	@python -c "import jax; print(f'JAX: {jax.__version__}')" 2>/dev/null || echo "JAX: not installed"
	@python -c "import torchax; print(f'TorchAx: {torchax.__version__}')" 2>/dev/null || echo "TorchAx: not installed"
