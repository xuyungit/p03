# GP Baseline Usage Notes

## Run Commands

### Neural Network Baseline (residual MLP)
```bash
uv run python src/models/bridge_nn.py \
  --train-csv data/d3d04_all_train_r.csv data/m2_0914_r_all_train.csv data/m2_lhs_0916_train.csv \
  --test-csv data/d3d04_all_test_r.csv data/m2_0914_r_all_test.csv data/m2_lhs_0916_test.csv \
  --input-cols-re '^F[1-6]$, ^N[0-9]+_r$, T_grad' \
  --target-cols-re '^R[0-9]+$, ^Ry_t[0-9]+$, ^Rx_t[0-9]+$, D_t[0-9]+_r$' \
  --no-augment-flip --batch-size 8192 --epochs 2000 --lr 0.0001 --weight-decay 0.0001 \
  --scheduler cosine --warmup-steps 50 --num-layers 6 --hidden-dim 256 --model-type res_mlp \
  --dropout 0 --print-freq 100 --early-stopping --patience 750 --grad-clip 1.0 \
  --val-ratio 0.1 --seed 42 --experiment-root experiments/bridge_nn
```
Change `--seed` for additional replicates (seeds 7 and 1337 used in the latest sweep).

### SVGP Baseline (PCA + independent heads)
```bash
uv run python src/models/gp/svgp_baseline.py \
  --train-csv data/d3d04_all_train_r.csv data/m2_0914_r_all_train.csv data/m2_lhs_0916_train.csv \
  --test-csv data/d3d04_all_test_r.csv data/m2_0914_r_all_test.csv data/m2_lhs_0916_test.csv \
  --input-cols-re '^F[1-6]$, ^N[0-9]+_r$, T_grad' \
  --target-cols-re '^R[0-9]+$, ^Ry_t[0-9]+$, ^Rx_t[0-9]+$, D_t[0-9]+_r$' \
  --no-augment-flip --batch-size 512 --epochs 450 --prewarm-epochs 50 \
  --lr 0.008 --lr-decay cosine --lr-decay-min-lr 0.0005 \
  --early-stopping --patience 90 --val-interval 10 --min-delta 0.0001 \
  --kernel rbf --ard --inducing 512 --inducing-init kmeans --noise-init 0.001 --jitter 1e-6 \
  --pca-variance 0.98 --print-freq 50 --seed 42 \
  --experiment-root experiments/svgp_baseline --device cpu
```
Replace `--seed` as required (seeds 7 and 1337 added for significance checks).

## Multi-Seed Test Metrics

| Model | Seed | MAE | RMSE | R² | MAPE | 0.9-coverage (raw) | 0.9-coverage (tau/final) | Tau | Run dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NN | 42 | 5.57 | 21.04 | 0.997656 | 5.11 | – | – | – | 2025-09-30_10-59-59 |
| NN | 7 | 5.73 | 21.82 | 0.997631 | 5.12 | – | – | – | 2025-09-30_14-36-29 |
| NN | 1337 | 5.85 | 22.09 | 0.997534 | 5.21 | – | – | – | 2025-09-30_14-39-13 |
| SVGP | 42 | 3.71 | 15.87 | 0.998878 | 2.41 | 0.989 | 0.856 | 0.408 | 2025-09-30_13-44-49 |
| SVGP | 7 | 3.02 | 12.84 | 0.999243 | 2.17 | 0.994 | 0.863 | 0.381 | 2025-09-30_14-42-02 |
| SVGP | 1337 | 3.39 | 14.38 | 0.999025 | 2.25 | 0.991 | 0.853 | 0.392 | 2025-09-30_14-57-03 |

- NN mean ± std: MAE 5.72 ± 0.12, RMSE 21.65 ± 0.45, R² 0.9976 ± 0.0001, MAPE 5.14 ± 0.04.
- SVGP mean ± std: MAE 3.37 ± 0.28, RMSE 14.36 ± 1.24, R² 0.9990 ± 0.0001, MAPE 2.28 ± 0.10, raw 0.9-coverage 0.991 ± 0.002, tau-scaled 0.9-coverage 0.857 ± 0.004.

## Uncertainty Calibration

`svgp_baseline.py` now calibrates both a global tau and per-target-family tau values on the validation split. By default (`--per-group-tau-mode auto`) the families are inferred from leading alphabetic prefixes (e.g. `R`, `Ry`, `Rx`, `D`) and stored in `per_group_tau.json` alongside `tau.json`. The inference utility (`src/models/gp/infer.py`) loads these files automatically when `--tau-mode auto` (default) so downstream scoring always uses the best available calibration. Use `--per-group-tau-mode none` to skip group calibration or `--per-group-tau-mode config --per-group-tau-config path.json` to supply custom groups. Inference can be forced to a particular strategy via `--tau-mode {per-group,global,none}`.

Seed 42 numbers (auto mode) remain unchanged numerically but are now emitted directly by the CLI:

- Per-group tau validation fit (targeting 0.9 coverage): `R=0.496`, `Ry=0.476`, `Rx=0.460`, `D=0.445`.
- Test NLL improves from -2.636 (global tau) to -2.684 (per-group), while 0.9 coverage rises from 0.856 to 0.887.

| Family | Raw 0.9 cov | Global tau 0.9 cov | Per-group 0.9 cov |
| --- | --- | --- | --- |
| R | 0.988 | 0.836 | 0.890 |
| Ry | 0.984 | 0.850 | 0.888 |
| Rx | 0.991 | 0.860 | 0.888 |
| D | 0.990 | 0.864 | 0.885 |

Calibration artifacts live under each run directory (e.g. `experiments/svgp_baseline/2025-09-30_13-44-49/per_group_tau.json`).

uv run python src/models/gp/svgp_baseline.py \
  --train-csv data/d3d04_all_train_r.csv data/m2_0914_r_all_train.csv data/m2_lhs_0916_train.csv \
  --test-csv  data/d3d04_all_test_r.csv  data/m2_0914_r_all_test.csv  data/m2_lhs_0916_test.csv \
  --input-cols-re  '^F[1-6]$, ^N[0-9]+_r$, T_grad' \
  --target-cols-re '^R[0-9]+$, ^Ry_t[0-9]+$, ^Rx_t[0-9]+$, D_t[0-9]+_r$' \
  --no-augment-flip --batch-size 1024 \
  --epochs 450 --prewarm-epochs 3 --val-interval 2 --print-freq 50 \
  --lr 0.008 --lr-decay cosine --lr-decay-min-lr 0.0005 \
  --kernel rbf --ard --inducing 512 --inducing-init kmeans --noise-init 0.001 --jitter 1e-6 \
  --pca-variance 0.98 --seed 42 \
  --experiment-root experiments/svgp_smoketest --device cpu