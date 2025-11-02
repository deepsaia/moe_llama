# Running Tests

## Quick Start

```bash
# Run fast tests (recommended for development)
pytest -m "not slow and not integration"

# Run all tests (includes slow/integration tests)
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_tokenizer.py
pytest tests/test_model.py
```

## Test Organization

**Fast Tests (68 tests)** - Run in ~2 seconds
- Unit tests for tokenizer, model, trainer, benchmarks
- Use tiny sample datasets
- Perfect for rapid development

**Slow/Integration Tests (7 tests)** - Require internet, take longer
- Load real datasets from HuggingFace
- Full training epoch tests
- Run separately: `pytest -m "slow or integration"`

## Common Commands

```bash
# Run tests with coverage report
pytest --cov=moellama --cov-report=html

# Coverage with terminal output
pytest --cov=moellama --cov-report=term-missing

# Run and stop at first failure
pytest -x

# Run specific test
pytest tests/test_model.py::TestModelInitialization::test_model_creation

# Show print statements
pytest -s

# Run tests in parallel (faster)
pytest -n auto  # requires: pip install pytest-xdist
```

## Coverage Reports

**Generate HTML coverage report:**
```bash
pytest --cov=moellama --cov-report=html
```

This creates `htmlcov/index.html` - open in browser to see:
- Overall coverage percentage
- Line-by-line coverage for each file
- Uncovered lines highlighted in red
- Branch coverage information

**Terminal coverage report:**
```bash
# Show coverage with missing lines
pytest --cov=moellama --cov-report=term-missing

# Coverage for specific module
pytest tests/test_model.py --cov=moellama.model
```

**Coverage output example:**
```
Name                          Stmts   Miss  Cover   Missing
-----------------------------------------------------------
moellama/__init__.py             12      0   100%
moellama/model.py               245     23    91%   67-71, 89
moellama/tokenizer.py           189     15    92%   145, 201-205
-----------------------------------------------------------
TOTAL                          1234    123    90%
```

**Interpreting coverage:**
- **Stmts**: Total statements in the file
- **Miss**: Statements not executed by tests
- **Cover**: Percentage covered
- **Missing**: Line numbers not covered

**Minimum coverage check:**
```bash
# Fail if coverage below 80%
pytest --cov=moellama --cov-fail-under=80
```

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and test data
├── test_tokenizer.py        # BPE tokenizer tests (17)
├── test_model.py            # Model architecture tests (19)
├── test_dataset_manager.py  # Multi-dataset tests (10)
├── test_trainer.py          # Training loop tests (11)
├── test_benchmarks.py       # Benchmark evaluation tests (16)
└── pytest.ini               # Test configuration
```

## Requirements

Tests automatically use:
- CPU-only execution (no GPU needed)
- Tiny sample datasets (fast)
- Temporary files (auto-cleanup)

All test dependencies are included in the dev group:
```bash
uv sync --group dev
```
