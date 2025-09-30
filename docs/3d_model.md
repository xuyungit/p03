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