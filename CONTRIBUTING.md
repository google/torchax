# Contributing

We appreciate all contributions. If you are planning to contribute a bug fix for an open issue, please comment on the thread and we're happy to provide any guidance. You are very welcome to pick issues from `good first issue` and `help wanted` labels.

If you plan to contribute new features, utility functions or extensions to the core, please first open an issue and discuss the feature with us. Sending a PR without discussion might end up resulting in a rejected PR, because we might be taking the core in a different direction than you might be aware of.

## Developer Setup

### Prerequisites

- Python 3.11 or higher
- Git

### Quick Start (Recommended)

The Makefile handles everything, including uv installation:

```bash
# Clone the repository
git clone https://github.com/google/torchax.git
cd torchax

# Install uv locally (if not already in system)
make install-uv

# Create a virtual environment
uv venv --python 3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with all development dependencies
make install

# Verify setup
make check-env

# Run tests
make test
```

### Alternative: System-wide uv

If you prefer uv in your PATH:

```bash
# Install uv system-wide
curl -LsSf https://astral.sh/uv/install.sh | sh

# Then follow the same steps above
uv venv --python 3.11
source .venv/bin/activate
make install
```

### Without uv (Traditional)

```bash
# Create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package in editable mode with dependencies
pip install -e ".[cpu,dev,docs]"
```

### Mac Setup (M1/M2/M3)

Development works great on Apple Silicon Macs:

```bash
# Recommended: Using uv
make install-uv           # Install uv if needed
uv venv --python 3.11
source .venv/bin/activate
make install

# Alternative: Using conda
conda create --name torchax python=3.11
conda activate torchax
pip install -e ".[cpu,dev,docs]"
```

### Hardware-Specific Installation

```bash
# CPU (default - flexible versions for development)
make install

# CPU with pinned test versions (exactly like CI)
make install-test

# CUDA
make install-cuda

# TPU (requires additional setup)
make install-tpu
```

## Development Workflow

### Using Make Commands

We provide a Makefile for common development tasks:

```bash
# See all available commands
make help

# Install for development
make dev

# Check for issues
make lint

# Auto-fix issues and format
make format

# Format code
make format

# Run tests
make test

# Run all tests (including distributed)
make test-all

# Clean build artifacts
make clean
```

## Testing

```bash
# Run unit tests
make test

# Run specific test file
JAX_PLATFORMS=cpu pytest test/test_ops.py

# Run tests with verbose output
JAX_PLATFORMS=cpu pytest test/ -v

# Run distributed tests
make test-all
```

## Project Structure

```
torchax/
├── torchax/           # Main package
│   ├── ops/          # Operator implementations
│   ├── tensor.py     # Core tensor functionality
│   └── ...
├── test/             # Unit tests
├── test_dist/        # Distributed tests
├── examples/         # Example scripts
├── docs/             # Documentation
├── pyproject.toml    # Project configuration
└── Makefile          # Development commands
```

## VSCode Setup

Recommended extensions:

- **Python** (ms-python.python)
- **Ruff** (charliermarsh.ruff)
- **Python Debugger** (ms-python.debugpy)

### Settings

Add to `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll": "explicit",
      "source.organizeImports": "explicit"
    }
  },
  "ruff.lint.args": ["--config=pyproject.toml"],
  "ruff.format.args": ["--config=pyproject.toml"]
}
```

## Common Issues

### ImportError after code changes

If you get import errors or operators not found:

```bash
# Reinstall the package
pip install -e .
```

### JAX backend issues

Set the JAX platform explicitly:

```bash
export JAX_PLATFORMS=cpu  # or cuda, tpu
```

### If tests fail locally but passing in CI

Make sure you have the latest dependencies:

```bash
make clean
make install-test  # Use exact CI versions
```

## Documentation

Build and serve documentation locally:

```bash
make docs-serve
```

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/google/torchax/issues)
- **Discussions**: [GitHub Discussions](https://github.com/google/torchax/discussions)
