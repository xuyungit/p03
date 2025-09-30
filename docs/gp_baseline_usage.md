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

Global tau on seed 42 (`tau = 0.408`) tightens intervals but remains conservative for 0.9 coverage (0.856). A per-target-family scaling that matches 0.9 coverage on the validation split improves test coverage without hurting NLL:

- Per-group tau values (val 0.9 target): `R=0.496`, `Ry=0.476`, `Rx=0.460`, `D=0.445`.
- Test NLL improves from -2.636 (global tau) to -2.684 (per-group), while 0.9 coverage rises from 0.856 to 0.887.

| Family | Raw 0.9 cov | Global tau 0.9 cov | Per-group 0.9 cov |
| --- | --- | --- | --- |
| R | 0.988 | 0.836 | 0.890 |
| Ry | 0.984 | 0.850 | 0.888 |
| Rx | 0.991 | 0.860 | 0.888 |
| D | 0.990 | 0.864 | 0.885 |

Artifact snapshot stored at `experiments/svgp_baseline/2025-09-30_13-44-49/per_group_tau.json`.
