# MergeKitty Development Guide

## Build, Test and Lint Commands

- Install dependencies: `uv pip install -e .`
- Create venv (if needed): `uv venv`
- Run linter: `hatch run lint`
- Fix lint errors: `hatch run lint-fix`
- Format code: `hatch run format`
- Run tests: `python -m pytest tests/`
- Run single test: `python -m pytest tests/test_file.py::TestClass::test_method -v`

## Code Style Guidelines

- Line length: 88 characters (Black/ruff standard)
- Use type annotations (Python 3.10+ supported)
- Class naming: CamelCase
- Function/variable naming: snake_case
- Constants: UPPER_SNAKE_CASE
- Private members: prefix with underscore (`_private_method`)
- Use pydantic models for structured data and validation
- Import order: standard library → third party → local modules
- Use `frozen=True` for immutable data classes
- Task-based approach for computational operations
- Prefer explicit error handling with detailed error messages
- Document public interfaces with docstrings

## Dependencies

The project uses torch, peft, transformers, and related HuggingFace libraries.
Prefer using `LazyTensorLoader` for memory-efficient model handling.