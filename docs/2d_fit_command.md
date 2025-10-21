# 使用傅里叶温度拟合2d模型参数

## case 2

```bash

# 反力+支座角度

uv run python src/data/add_data_perturbation.py \
      --input-csv data/augmented/dt_24hours_data_new.csv  \
      --output-csv data/augmented/dt_24hours_data_new_round4.csv  \
      --round-digits 4 \
      --columns-re "^theta_.+rad$"

uv run python src/data/add_data_perturbation.py \
      --input-csv data/augmented/dt_24hours_data_new.csv  \
      --output-csv data/augmented/dt_24hours_data_new_rsensor0501.csv  \
      --apply-angle-sensor-noise \
      --columns-re "^theta_.+rad$"

uv run python src/data/add_data_perturbation.py \
    --input-csv data/augmented/dt_24hours_data_new_rsensor0501.csv \
    --output-csv data/augmented/dt_24hours_data_new_rsensor0501_noise5.csv \
    --add-proportional-noise 0.05 \
    --columns-re "^R_.+_kN$"

uv run python src/data/add_data_perturbation.py \
    --input-csv data/augmented/dt_24hours_data_new_round4.csv \
    --output-csv data/augmented/dt_24hours_data_new_round4_noise5.csv \
    --add-proportional-noise 0.05 \
    --columns-re "^R_.+_kN$"

uv run python src/data/add_data_perturbation.py \
    --input-csv data/augmented/dt_24hours_data_new_rsensor0501_noise5.csv \
    --output-csv data/augmented/dt_24hours_data_new_rsensor0501_noise5_biased.csv \
    --bias-add-new-columns \
    --add-bias-range -0.05 0.05 \
    --columns-re "^theta_.+rad$"

uv run python src/models/bridge_2d_fit.py \
    --data data/augmented/dt_24hours_data_new_round4_noise5.csv \
    --fixed-kv 1.21875633217039 0.575148946195117 1.03133789506241 0.952294668480702 \
    --use-rotations --rotation-weight 1.0 \
    --maxiter 100 \
    --output results/r_n5_6_theta_round4_fit_kv.csv \
    --temp-basis fourier --fourier-harmonics 8 

uv run python src/models/bridge_2d_fit.py \
    --data data/augmented/dt_24hours_data_new_rsensor0501_noise5_biased.csv \
    --fixed-kv 1.21875633217039 0.575148946195117 1.03133789506241 0.952294668480702 \
    --use-rotations --rotation-weight 1.0 \
    --maxiter 100 \
    --output results/r_n5_6_theta_round4_fit_kv.csv \
    --temp-basis fourier --fourier-harmonics 8 \
    --fit-rotation-bias --refit-after-bias --stage1-rotation-weight 20.0

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
    --data data/augmented/dt_24hours_data_new_rsensor0501_noise5_biased.csv \
    --fixed-kv 1.21875633217039 0.575148946195117 1.03133789506241 0.952294668480702 \
    --use-rotations --rotation-weight 1.0 \
    --use-span-rotations --span-rotation-weight 1.0 \
    --maxiter 100 \
    --output results/r_n5_6_theta_round4_fit_kv.csv \
    --temp-basis fourier --fourier-harmonics 8
    --fit-rotation-bias --refit-after-bias --stage1-rotation-weight 20.0
    


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


## case 4

```bash
uv run python src/data/bridge_reaction_sampling.py --params data/augmented/dt_24hours_new2.csv --out data/augmented/dt_24hours_data_new2_373.csv --span-lengths 30:70:30

uv run python src/data/add_data_perturbation.py \
      --input-csv data/augmented/dt_24hours_data_new2_373.csv  \
      --output-csv data/augmented/dt_24hours_data_new2_373_rsensor0501.csv  \
      --apply-angle-sensor-noise \
      --columns-re "^theta_.+rad$"

uv run python src/data/add_data_perturbation.py \
    --input-csv data/augmented/dt_24hours_data_new2_373_rsensor0501.csv \
    --output-csv data/augmented/dt_24hours_data_new2_373_rsensor0501_noise5.csv \
    --add-proportional-noise 0.05 \
    --columns-re "^R_.+_kN$"

uv run python src/models/bridge_2d_fit.py \
    --data data/augmented/dt_24hours_data_new2_373_rsensor0501_noise5.csv \
    --fixed-kv 1.21875633217039 0.575148946195117 1.03133789506241 0.952294668480702 \
    --use-rotations --rotation-weight 1.0 \
    --maxiter 100 \
    --output results/r_n5_6_theta_round4_fit_kv.csv \
    --temp-basis fourier --fourier-harmonics 8

# 反力+支座角度+跨种角度
uv run python src/models/bridge_2d_fit.py \
    --data data/augmented/dt_24hours_data_new2_373_rsensor0501_noise5.csv \
    --fixed-kv 1.21875633217039 0.575148946195117 1.03133789506241 0.952294668480702  \
    --use-rotations --rotation-weight 1.0 \
    --use-span-rotations --span-rotation-weight 1.0 \
    --maxiter 100 \
    --output results/r_n5_6_theta_round4_fit_kv.csv \
    --temp-basis fourier --fourier-harmonics 8

# 反力
uv run python src/models/bridge_2d_fit.py \
    --data data/augmented/dt_24hours_data_new2_373_rsensor0501_noise5.csv \
    --fixed-kv 1.21875633217039 0.575148946195117 1.03133789506241 0.952294668480702  \
    --maxiter 100 \
    --output results/r_n5_6_theta_round4_fit_kv.csv \
    --temp-basis fourier --fourier-harmonics 8 
```

## case 5

```bash
uv run python src/data/bridge_reaction_sampling.py --params data/augmented/dt_24hours_new3.csv --out data/augmented/dt_24hours_data_new3_373.csv --span-lengths 30:70:30

uv run python src/data/add_data_perturbation.py \
      --input-csv data/augmented/dt_24hours_data_new3_373.csv  \
      --output-csv data/augmented/dt_24hours_data_new3_373_rsensor0501.csv  \
      --apply-angle-sensor-noise \
      --columns-re "^theta_.+rad$"

uv run python src/data/add_data_perturbation.py \
    --input-csv data/augmented/dt_24hours_data_new3_373_rsensor0501.csv \
    --output-csv data/augmented/dt_24hours_data_new3_373_rsensor0501_noise5.csv \
    --add-proportional-noise 0.05 \
    --columns-re "^R_.+_kN$"

uv run python src/models/bridge_2d_fit.py \
    --data data/augmented/dt_24hours_data_new3_373_rsensor0501_noise5.csv \
    --fixed-kv 1.2891632775508988 1.2969883848242527 0.8480060335586834 1.1405218884143073 \
    --use-rotations --rotation-weight 1.0 \
    --maxiter 100 \
    --output results/r_n5_6_theta_round4_fit_kv.csv \
    --temp-basis fourier --fourier-harmonics 8

# 反力+支座角度+跨种角度
uv run python src/models/bridge_2d_fit.py \
    --data data/augmented/dt_24hours_data_new3_373_rsensor0501_noise5.csv \
    --fixed-kv 1.2891632775508988 1.2969883848242527 0.8480060335586834 1.1405218884143073  \
    --use-rotations --rotation-weight 1.0 \
    --use-span-rotations --span-rotation-weight 1.0 \
    --maxiter 100 \
    --output results/r_n5_6_theta_round4_fit_kv.csv \
    --temp-basis fourier --fourier-harmonics 8

# 反力
uv run python src/models/bridge_2d_fit.py \
    --data data/augmented/dt_24hours_data_new3_373_rsensor0501_noise5.csv \
    --fixed-kv 1.2891632775508988 1.2969883848242527 0.8480060335586834 1.1405218884143073  \
    --maxiter 100 \
    --output results/r_n5_6_theta_round4_fit_kv.csv \
    --temp-basis fourier --fourier-harmonics 8 
```

