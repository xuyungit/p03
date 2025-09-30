# Repository Guidelines

## Project Overview

This project is a machine learning model for predicting the mechanical properties of bridge components.


## Project Structure & Module Organization
- `src/models/` — training, evaluation, and search harnesses (e.g., `hparam_search_rev03.py`).
- `src/data/` — data utilities (e.g., `noise_utils.py` for synthetic noise/CLI).
- `experiments/` — run outputs, metrics, and artifacts written by training/search.
- `data/` — input CSV datasets (large files are not versioned beyond `.gitignore`).
- `docs/` — usage notes and model recipes (see `docs/3d_model.md`).
- `tests/` — place unit/integration tests here.
- `pyproject.toml` — Python 3.11 project, dependencies, setuptools config; source root is `src/`.

Tip: for ad‑hoc scripts or notebooks, set `PYTHONPATH=src` so `models`/`data` are importable.

## Build, Test, and Development Commands
- Environment: `uv sync` (installs from `pyproject.toml`/`uv.lock`, Python 3.11).
- Run training: `uv run python src/models/rev03_rtd_nf_e3_tree.py ...`
- Hyperparam search: `uv run python src/models/hparam_search_rev03.py --train-csv ... --test-csv ...`
- Add noise to CSV: `uv run python src/data/noise_utils.py --input data/in.csv --output data/out.csv`
- Optional install (editable): `uv run python -m pip install -e .`
- Tests (when added): `uv run pytest -q`

## Coding Style & Naming Conventions
- Python: 4‑space indent, type hints required for public APIs, module docstrings where helpful.
- Naming: functions/variables `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`, files `snake_case.py`.
- Imports: prefer absolute (e.g., `from models...`) from the `src/` root.
- Formatting/linting (recommended): `black` and `ruff`; type‑check with `mypy` or `pyright`.
  - Suggested dev add: `uv add --dev black ruff pytest mypy`

## Testing Guidelines
- Framework: `pytest`.
- Layout: tests under `tests/`, files named `test_*.py`; small CSV fixtures in `tests/fixtures/`.
- Scope: unit tests for core utilities (`src/data/noise_utils.py`) and CLI smoke tests.
- Run locally: `uv run pytest -q`; aim to cover critical data paths and metrics calculations.

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise subject (<72 chars), optional body for rationale. Conventional Commits welcome but not required.
- PRs: clear description, linked issues, reproduction commands, and:
  - For training changes: include dataset list, exact command, and `experiments/...` run directory.
  - For data utilities: example input/output and CLI flags used.
- Keep diffs focused; update `docs/` when behavior or flags change.

## Security & Data
- Do not commit secrets or large raw datasets. Prefer referencing paths under `data/` and storing outputs under `experiments/`.
