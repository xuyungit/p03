# 使用傅里叶温度拟合2d模型参数

## case 2

```bash

# 反力+支座角度

uv run python src/models/bridge_2d_fit.py \
    --data data/augmented/dt_24hours_data_new_round4_noise5.csv \
    --fixed-kv 1.21875633217039 0.575148946195117 1.03133789506241 0.952294668480702 \
    --use-rotations --rotation-weight 1.0 \
    --maxiter 100 \
    --output results/r_n5_6_theta_round4_fit_kv.csv \
    --temp-basis fourier --fourier-harmonics 8

uv run python src/models/bridge_2d_fit.py \
    --data data/augmented/dt_24hours_data_new_rsensor0501_noise5.csv \
    --fixed-kv 1.21875633217039 0.575148946195117 1.03133789506241 0.952294668480702 \
    --use-rotations --rotation-weight 1.0 \
    --maxiter 100 \
    --output results/r_n5_6_theta_round4_fit_kv.csv \
    --temp-basis fourier --fourier-harmonics 8

# 反力+支座角度+跨种角度
uv run python src/models/bridge_2d_fit.py \
    --data data/augmented/dt_24hours_data_new_round4_noise5.csv \
    --fixed-kv 1.21875633217039 0.575148946195117 1.03133789506241 0.952294668480702 \
    --use-rotations --rotation-weight 1.0 \
    --use-span-rotations --span-rotation-weight 1.0 \
    --maxiter 100 \
    --output results/r_n5_6_theta_round4_fit_kv.csv \
    --temp-basis fourier --fourier-harmonics 8

uv run python src/models/bridge_2d_fit.py \
    --data data/augmented/dt_24hours_data_new_rsensor0501_noise5.csv \
    --fixed-kv 1.21875633217039 0.575148946195117 1.03133789506241 0.952294668480702 \
    --use-rotations --rotation-weight 1.0 \
    --use-span-rotations --span-rotation-weight 1.0 \
    --maxiter 100 \
    --output results/r_n5_6_theta_round4_fit_kv.csv \
    --temp-basis fourier --fourier-harmonics 8


# 反力
uv run python src/models/bridge_2d_fit.py \
    --data data/augmented/dt_24hours_data_new2_round4_noise5.csv \
    --fixed-kv 1.21875633217039 0.575148946195117 1.03133789506241 0.952294668480702 \
    --maxiter 100 \
    --output results/r_n5_6_theta_round4_fit_kv.csv \
    --temp-basis fourier --fourier-harmonics 8 
```

## case 1

```bash
uv run python src/models/bridge_2d_fit.py \
    --data data/augmented/dt_24hours_data_new2_round4_noise5.csv \
    --fixed-kv 1.21875633217039 0.575148946195117 1.03133789506241 0.952294668480702 \
    --use-rotations --rotation-weight 1.0 \
    --maxiter 100 \
    --output results/r_n5_6_theta_round4_fit_kv.csv \
    --temp-basis fourier --fourier-harmonics 8

uv run python src/models/bridge_2d_fit.py \
    --data data/augmented/dt_24hours_data_new2_rsensor0501_noise5.csv \
    --fixed-kv 1.21875633217039 0.575148946195117 1.03133789506241 0.952294668480702 \
    --use-rotations --rotation-weight 1.0 \
    --maxiter 100 \
    --output results/r_n5_6_theta_round4_fit_kv.csv \
    --temp-basis fourier --fourier-harmonics 8

# 反力+支座角度+跨种角度
uv run python src/models/bridge_2d_fit.py \
    --data data/augmented/dt_24hours_data_new2_round4_noise5.csv \
    --fixed-kv 1.21875633217039 0.575148946195117 1.03133789506241 0.952294668480702 \
    --use-rotations --rotation-weight 1.0 \
    --use-span-rotations --span-rotation-weight 1.0 \
    --maxiter 100 \
    --output results/r_n5_6_theta_round4_fit_kv.csv \
    --temp-basis fourier --fourier-harmonics 8

uv run python src/models/bridge_2d_fit.py \
    --data data/augmented/dt_24hours_data_new2_rsensor0501_noise5.csv \
    --fixed-kv 1.21875633217039 0.575148946195117 1.03133789506241 0.952294668480702 \
    --use-rotations --rotation-weight 1.0 \
    --use-span-rotations --span-rotation-weight 1.0 \
    --maxiter 100 \
    --output results/r_n5_6_theta_round4_fit_kv.csv \
    --temp-basis fourier --fourier-harmonics 8


# 反力
uv run python src/models/bridge_2d_fit.py \
    --data data/augmented/dt_24hours_data_new2_round4_noise5.csv \
    --fixed-kv 1.21875633217039 0.575148946195117 1.03133789506241 0.952294668480702 \
    --maxiter 100 \
    --output results/r_n5_6_theta_round4_fit_kv.csv \
    --temp-basis fourier --fourier-harmonics 8 
```


## case3

```bash
uv run python src/models/bridge_2d_fit.py \
    --data data/augmented/dt_24hours_data_new3_round4_noise5.csv \
    --fixed-kv 1.2891632775508988 1.2969883848242527 0.8480060335586834 1.1405218884143073 \
    --use-rotations --rotation-weight 1.0 \
    --maxiter 100 \
    --output results/r_n5_6_theta_round4_fit_kv.csv \
    --temp-basis fourier --fourier-harmonics 8

uv run python src/models/bridge_2d_fit.py \
    --data data/augmented/dt_24hours_data_new3_rsensor0501_noise5.csv \
    --fixed-kv 1.2891632775508988 1.2969883848242527 0.8480060335586834 1.1405218884143073 \
    --use-rotations --rotation-weight 1.0 \
    --maxiter 100 \
    --output results/r_n5_6_theta_round4_fit_kv.csv \
    --temp-basis fourier --fourier-harmonics 8

# 反力+支座角度+跨种角度
uv run python src/models/bridge_2d_fit.py \
    --data data/augmented/dt_24hours_data_new3_round4_noise5.csv \
    --fixed-kv 1.2891632775508988 1.2969883848242527 0.8480060335586834 1.1405218884143073 \
    --use-rotations --rotation-weight 1.0 \
    --use-span-rotations --span-rotation-weight 1.0 \
    --maxiter 100 \
    --output results/r_n5_6_theta_round4_fit_kv.csv \
    --temp-basis fourier --fourier-harmonics 8

uv run python src/models/bridge_2d_fit.py \
    --data data/augmented/dt_24hours_data_new3_rsensor0501_noise5.csv \
    --fixed-kv 1.2891632775508988 1.2969883848242527 0.8480060335586834 1.1405218884143073 \
    --use-rotations --rotation-weight 1.0 \
    --use-span-rotations --span-rotation-weight 1.0 \
    --maxiter 100 \
    --output results/r_n5_6_theta_round4_fit_kv.csv \
    --temp-basis fourier --fourier-harmonics 8


# 反力
uv run python src/models/bridge_2d_fit.py \
    --data data/augmented/dt_24hours_data_new3_round4_noise5.csv \
    --fixed-kv 1.2891632775508988 1.2969883848242527 0.8480060335586834 1.1405218884143073 \
    --maxiter 100 \
    --output results/r_n5_6_theta_round4_fit_kv.csv \
    --temp-basis fourier --fourier-harmonics 8 
```
