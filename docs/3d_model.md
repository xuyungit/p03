# 绿云路3D模型

## 模型训练

### 正模型

```bash
uv run python src/models/bridge_nn.py \
      --train-csv data/d3d04_all_train_r.csv \
      data/m2_0914_r_all_train.csv \
      data/m2_lhs_0916_train.csv \
      --test-csv data/d3d04_all_test_r.csv \
      data/m2_0914_r_all_test.csv \
      data/m2_lhs_0916_test.csv \
      --input-cols-re '^F[1-6]$, ^N\d+_r$, T_grad, ' \
      --target-cols-re '^R\d+$ , ^Ry_t\d+$, ^Rx_t\d+$, D_t\d+_r$' \
      --no-augment-flip \
      --dropout 0 --print-freq 100 --epochs 2000 --lr 1e-4 --early-stopping --patience 750   --scheduler cosine --warmup-steps 50 \
      --num-layers 6 --hidden-dim 256 --batch-size 8192 --model-type res_mlp

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


  uv run python src/models/gp/mt_icm.py \
    --train-csv data/d3d04_all_train_r.csv data/m2_0914_r_all_train.csv data/m2_lhs_0916_train.csv \
    --test-csv data/d3d04_all_test_r.csv data/m2_0914_r_all_test.csv data/m2_lhs_0916_test.csv \
    --input-cols-re '^F[1-6]$, ^N[0-9]+_r$, T_grad' \
    --target-cols-re '^R[0-9]+$, ^Ry_t[0-9]+$, ^Rx_t[0-9]+$, D_t[0-9]+_r$' \
    --no-augment-flip --batch-size 512 --epochs 450 --prewarm-epochs 50 \
    --lr 0.008 --lr-decay cosine --lr-decay-min-lr 0.0005 \
    --early-stopping --patience 90 --val-interval 10 --min-delta 0.0001 \
    --kernel rbf --ard --inducing 512 --inducing-init kmeans --noise-init 0.001 --jitter 1e-6 \
    --pca-variance 0.98 --print-freq 50 --seed 42 \
    --experiment-root experiments/mt_icm --device cpu \
    --enable-per-group-tau --group-patterns '^R[0-9]+$' '^Ry_t[0-9]+$' '^Rx_t[0-9]+$' '^D_t[0-9]+_r$'

  uv run python src/models/gp/mt_icm.py \
    --train-csv data/d3d04_all_train_r.csv data/m2_0914_r_all_train.csv data/m2_lhs_0916_train.csv \
    --test-csv data/d3d04_all_test_r.csv data/m2_0914_r_all_test.csv data/m2_lhs_0916_test.csv \
    --input-cols-re '^F[1-6]$, ^N[0-9]+_r$, T_grad' \
    --target-cols-re '^R[0-9]+$, ^Ry_t[0-9]+$, ^Rx_t[0-9]+$, D_t[0-9]+_r$' \
    --no-augment-flip --batch-size 512 --epochs 450 --prewarm-epochs 50 \
    --lr 0.008 --lr-decay cosine --lr-decay-min-lr 0.0005 \
    --early-stopping --patience 90 --val-interval 10 --min-delta 0.0001 \
    --kernel rbf --ard --inducing 512 --inducing-init kmeans --noise-init 0.001 --jitter 1e-6 \
    --pca-variance 0.98 --print-freq 50 --seed 42 \
    --experiment-root experiments/mt_lmc --device cpu \
    --enable-per-group-tau --group-patterns '^R[0-9]+$' '^Ry_t[0-9]+$' '^Rx_t[0-9]+$' '^D_t[0-9]+_r$' \
    --lmc-rank 5
```
