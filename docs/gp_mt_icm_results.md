# B 档 ICM 多任务原型实验

## 运行配置
- 命令：
  ```bash
  uv run python src/models/gp/mt_icm.py \
    --train-csv data/d3d04_all_train_r.csv data/m2_0914_r_all_train.csv data/m2_lhs_0916_train.csv \
    --test-csv data/d3d04_all_test_r.csv data/m2_0914_r_all_test.csv data/m2_lhs_0916_test.csv \
    --input-cols-re '^F[1-6]$, ^N[0-9]+_r$, T_grad' \
    --target-cols-re '^R[0-9]+$, ^Ry_t[0-9]+$, ^Rx_t[0-9]+$, D_t[0-9]+_r$' \
    --no-augment-flip --batch-size 512 --epochs 500 --prewarm-epochs 50 \
    --lr 0.008 --lr-decay cosine --lr-decay-min-lr 3e-4 \
    --early-stopping --patience 120 --val-interval 10 --min-delta 5e-5 \
    --kernel rbf --ard --inducing 512 --inducing-init kmeans --icm-rank 8 \
    --noise-init 5e-4 --jitter 1e-6 --pca-variance 0.98 --seed 42 \
    --experiment-root experiments/mt_icm --device cpu
  ```
- 关键超参：PCA latent 维度 16、诱导点 512、LMC latent 数 8（共享诱导点）、Adam + cosine 退火。
- 运行目录：`experiments/mt_icm/2025-09-30_20-57-50`
- 验证最优（epoch 268）：latent 空间 RMSE ≈ 0.897。

## 与基线对比
| 模型 | 运行目录 | MAE | RMSE | R² | MAPE | 0.9 覆盖率 (raw) |
| --- | --- | --- | --- | --- | --- | --- |
| NN (residual MLP) | `experiments/bridge_nn/2025-09-30_10-59-59` | 5.57 | 21.04 | 0.9977 | 5.11 | – |
| A 档 SVGP (独立任务) | `experiments/svgp_baseline/2025-09-30_13-44-49` | 3.37 | 14.36 | 0.9990 | 2.28 | 0.991 (raw) |
| B 档 ICM 原型 | `experiments/mt_icm/2025-09-30_20-57-50` | 20.58 | 72.28 | 0.8168 | 45.46 | 0.897 |

> 说明：NN/SVGP 数据源自 `docs/gp_baseline_usage.md`；B 档指标来自 `test_metrics.json` 与 `test_uncertainty.json`。

## 结果观察
- `R*` 组（反力）MAE ≈ 113，对应相对误差 1–4%；`D*` 位移 MAE ≈ 2.85，误差 80–120%（小量级输出对相对误差敏感）。
- `Ry/Rx` 角度分量的绝对误差 ~1e-4，但因目标值接近 0，MAPE 与 NLL 仍偏大。
- 不确定性：0.9 覆盖率原始 0.897，`tau.json` 给出全局 τ≈1.04，分组 τ `{R:0.971, Ry:1.000, Rx:1.040, D:1.040}`。
- 任务协方差：`task_covariance.csv`、`plots/task_covariance_heatmap.png` 展示了 16×16 latent 相关性，系数矩阵总体主对角占优，暗示当前 LMC 仍偏向近独立解耦。

## 结论与后续方向
- 当前 ICM 原型尚明显落后于 NN / A 档 SVGP，尤其在大幅度的反力与位移维度上 RMSE/MAE 仍高；需要进一步调参。
- 优先改进项：
  1. 调整 latent 维度/ICM 秩（>8）与学习率/噪声先验，探索更强的相关建模。
  2. 引入变分白化或分组学习率，降低训练初期的发散与高偏差。
  3. 在 `Ry/Rx` 小量级输出上引入加权 loss / re-scaling，以抑制 MAPE 过高。
  4. 复现跑 GPU 版本以验证计算瓶颈与更大 batch/m 的可行性。
- 该实验已满足 B 档原型落地与产物规范，可作为后续调优的基线。
