# Claude Code Configuration for GifLab

This file provides project-specific guidance for AI assistants working with the GifLab codebase.

## 🚨 **CRITICAL: Always Use Poetry**

**This project uses Poetry for ALL Python command execution.** Never run Python commands directly.

### ❌ **WRONG** - These will fail with ModuleNotFoundError:
```bash
python -m giflab run --preset quick-test
python -m pytest tests/
python -c "from giflab.metrics import calculate_metrics"
PYTHONPATH=src python -m giflab run
```

### ✅ **CORRECT** - Always use Poetry:
```bash
poetry run python -m giflab run --preset quick-test
poetry run pytest tests/
poetry run python -c "from giflab.metrics import calculate_metrics"
```

## Why Poetry is Required

1. **Dependencies**: `click`, `pytest`, `numpy`, etc. exist only in Poetry's virtual environment
2. **Path Resolution**: Poetry ensures proper module discovery and PYTHONPATH setup
3. **Version Consistency**: Poetry locks dependency versions for reproducible builds
4. **Project Structure**: `src/giflab/` package structure requires Poetry's path handling

## Common Command Patterns

### Testing
```bash
# Fast development tests
poetry run pytest -m "fast" tests/ -n auto --tb=short

# Integration tests  
poetry run pytest -m "not slow" tests/ -n 4 --tb=short

# Full test suite
poetry run pytest tests/ --tb=short
```

### GifLab Operations
```bash
# Pipeline analysis and optimization
poetry run python -m giflab run --preset frame-focus
poetry run python -m giflab run --sampling representative

# Large-scale processing
poetry run python -m giflab run data/raw --workers 8

# Analysis tools
poetry run python -m giflab select-pipelines results.csv --top 3
```

### Development Tools
```bash
# Code quality
poetry run black src/ tests/
poetry run ruff check src/ tests/
poetry run mypy src/

# Interactive Python
poetry run python
poetry run jupyter notebook
```

## Project Structure Notes

- **Source code**: `src/giflab/` (package structure)
- **Tests**: `tests/` (pytest-based)
- **Configuration**: `pyproject.toml` (Poetry + tool config)
- **Dependencies**: Managed entirely through Poetry
- **Scripts**: Defined in `[tool.poetry.scripts]` section

## Environment Setup

The project is already properly configured. Just ensure Poetry is installed:

```bash
# Install Poetry (if needed)
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install

# Verify setup
poetry run python -c "import giflab; print('✅ GifLab ready!')"
```

## Makefile Integration

The project includes a Makefile that properly uses Poetry. You can also use:

```bash
make test-fast      # Development testing
make test-integration  # Pre-commit validation
make data          # Run compression pipeline
```

All Makefile targets internally use `poetry run` commands.

## Troubleshooting

### "ModuleNotFoundError: No module named 'click'"
- **Cause**: Using `python -m` instead of `poetry run python -m`
- **Fix**: Always prefix with `poetry run`

### "ModuleNotFoundError: No module named 'giflab'"
- **Cause**: Wrong working directory or missing Poetry
- **Fix**: Ensure you're in project root with `pyproject.toml`, then use `poetry run`

### "Command not found: giflab"
- **Cause**: Trying to use `giflab` directly instead of module syntax
- **Fix**: Use `poetry run python -m giflab` or `poetry run giflab`

## For AI Assistants: Key Reminders

1. **ALWAYS** use `poetry run` for Python execution
2. **NEVER** use bare `python`, `pip`, or `pytest` commands
3. **CHECK** that you're prefixing commands with `poetry run`
4. **TEST** commands with `poetry run` if unsure
5. **REMEMBER** this is a Poetry project - dependencies require the virtual environment

---

*This configuration ensures reliable, reproducible development workflows for both humans and AI assistants.*