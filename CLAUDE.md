# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning project for predicting mechanical properties of bridge components. The project compares multiple model architectures including neural networks (residual MLP), tree-based models (LightGBM), and Gaussian Process models (SVGP - Sparse Variational Gaussian Process).

## Development Commands

### Environment Setup
```bash
uv sync  # Install dependencies from pyproject.toml and uv.lock
```

### Model Training
```bash
# Neural Network (residual MLP)
uv run python src/models/bridge_nn.py \
  --train-csv data/d3d04_all_train_r.csv data/m2_0914_r_all_train.csv data/m2_lhs_0916_train.csv \
  --test-csv data/d3d04_all_test_r.csv data/m2_0914_r_all_test.csv data/m2_lhs_0916_test.csv \
  --input-cols-re '^F[1-6]$, ^N[0-9]+_r$, T_grad' \
  --target-cols-re '^R[0-9]+$, ^Ry_t[0-9]+$, ^Rx_t[0-9]+$, D_t[0-9]+_r$' \
  --no-augment-flip --batch-size 8192 --epochs 2000 --lr 0.0001 --model-type res_mlp

# Tree-based baseline
uv run python src/models/rev03_rtd_nf_e3_tree.py --train-csv ... --test-csv ...

# SVGP GP baseline
uv run python src/models/gp/svgp_baseline.py \
  --train-csv data/d3d04_all_train_r.csv data/m2_0914_r_all_train.csv data/m2_lhs_0916_train.csv \
  --test-csv data/d3d04_all_test_r.csv data/m2_0914_r_all_test.csv data/m2_lhs_0916_test.csv \
  --input-cols-re '^F[1-6]$, ^N[0-9]+_r$, T_grad' \
  --target-cols-re '^R[0-9]+$, ^Ry_t[0-9]+$, ^Rx_t[0-9]+$, D_t[0-9]+_r$' \
  --kernel rbf --ard --inducing 512 --pca-variance 0.98
```

### Hyperparameter Search
```bash
uv run python src/models/hparam_search_rev03.py --train-csv ... --test-csv ...
```

### Data Utilities
```bash
# Add synthetic noise to CSV
uv run python src/data/noise_utils.py --input data/in.csv --output data/out.csv
```

### Testing
```bash
uv run pytest -q  # Run all tests
uv run pytest tests/test_svgp_cli.py -v  # Run specific test
```

### Code Quality (recommended dev additions)
```bash
uv add --dev black ruff mypy
black src/ tests/
ruff check src/ tests/
mypy src/
```

## Architecture Overview

### Model Types
1. **Neural Networks** (`src/models/bridge_nn.py`): Residual MLP architecture with configurable depth, dropout, and learning rate scheduling
2. **Tree-based Models** (`src/models/rev03_rtd_nf_e3_tree.py`): LightGBM with MultiOutputRegressor for tabular data
3. **Gaussian Processes** (`src/models/gp/svgp_baseline.py`): SVGP with PCA dimensionality reduction for multi-output regression

### Core Components
- **Data Handling** (`src/models/training/data.py`): Data loading, augmentation (flip-based), and preprocessing
- **Training Infrastructure** (`src/models/training/`): Training loops, loss functions, data augmentation
- **Evaluation** (`src/models/evaluation/`): Metrics calculation, reporting, uncertainty quantification
- **GP Components** (`src/models/gp/`): GP data modules, inference utilities, common GP functions
- **Utilities** (`src/models/utils/`): Column specifications, shared utilities

### Experiment Structure
All training runs output to `experiments/{model_name}/{timestamp}/` containing:
- `config.json`: Training configuration
- `train_metrics.json`, `val_metrics.json`, `test_metrics.json`: Performance metrics
- Model artifacts and uncertainty estimates (for GP models)

### Column Specification Pattern
The codebase uses regex-based column selection throughout:
- Input columns: `^F[1-6]$, ^N[0-9]+_r$, T_grad` (forces, some reactions, temperature gradient)
- Target columns: `^R[0-9]+$, ^Ry_t[0-9]+$, ^Rx_t[0-9]+$, D_t[0-9]+_r$` (reactions and displacements)

### Data Augmentation
- Flip-based augmentation for structural symmetry
- Configurable noise profiles through `src/models/training/augment_config.py`
- Profile-based augmentation for realistic structural variations

## Testing Strategy
- Unit tests for core utilities (`tests/test_columns.py`, `tests/test_gp_pca.py`)
- Integration/smoke tests for CLI interfaces (`tests/test_svgp_cli.py`)
- Tests use small CSV fixtures in `tests/fixtures/`
- Focus on critical data paths and metrics calculations

## Key Dependencies
- **PyTorch + GPyTorch**: Neural networks and Gaussian Processes
- **LightGBM**: Gradient-boosted trees
- **scikit-learn**: Traditional ML utilities and preprocessing
- **pandas/numpy**: Data manipulation
- **tensorboard**: Training visualization

## Python Path
Set `PYTHONPATH=src` for ad-hoc scripts to make `models` and `data` importable from the source root.