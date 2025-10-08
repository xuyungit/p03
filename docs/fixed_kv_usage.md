# 固定 KV 参数和固定第一个沉降使用指南

## 概述

`fit_multi_case_v2.py` 现在支持：

1. **固定 KV（竖向支座刚度）参数**：使其不参与优化过程
2. **固定第一个沉降（Settlement A）为 0.0**：作为相对沉降的参考点（**默认启用**）

这些功能在以下场景中很有用：

### 固定 KV 参数的应用场景
1. **已知支座刚度**：当支座刚度通过试验或其他方法已知时
2. **减少参数空间**：减少优化参数数量，提高收敛速度
3. **灵敏度分析**：固定 KV 来研究其他参数的影响
4. **分步优化**：先固定 KV 优化其他参数，再放开 KV 进行全局优化

### 固定第一个沉降的原理
沉降是相对量，需要一个参考点。通常选择支座 A 作为参考点，固定其沉降为 0.0，这样：
- 其他支座的沉降值表示相对于 A 的相对沉降
- 减少 1 个优化参数（从 4 个沉降减少到 3 个）
- 避免参数空间的平移不确定性
- **这是工程实践中的标准做法**

## 使用方法

### 基本用法（默认配置）

**默认情况下**：
- Settlement A 固定为 0.0（作为参考点）
- KV 参数参与拟合
- 只需优化 3 个沉降值（B, C, D）

```bash
uv run python src/models/fit_multi_case_v2.py \
    --data data/augmented/dt_24hours_data_4.csv \
    --maxiter 500
```

### 固定 KV 参数

使用 `--fixed-kv` 参数指定四个支座的 KV 因子（依次为 A, B, C, D 支座）：

```bash
uv run python src/models/fit_multi_case_v2.py \
    --data data/augmented/dt_24hours_data_4.csv \
    --fixed-kv 1.0 1.0 1.0 1.0 \
    --maxiter 500
```

如果各支座刚度不同，可以指定不同的值：

```bash
uv run python src/models/fit_multi_case_v2.py \
    --data data/augmented/dt_24hours_data_4.csv \
    --fixed-kv 0.8 1.0 1.2 1.0 \
    --maxiter 500
```

### 不固定第一个沉降（拟合全部 4 个沉降）

如果需要拟合全部 4 个沉降值（不固定 A），使用 `--no-fix-first-settlement` 参数：

```bash
uv run python src/models/fit_multi_case_v2.py \
    --data data/augmented/dt_24hours_data_4.csv \
    --no-fix-first-settlement \
    --maxiter 500
```

**注意**：这会增加 1 个优化参数，并可能导致参数空间的平移不确定性。
uv run python src/models/fit_multi_case_v2.py \
    --data data/augmented/dt_24hours_data_4.csv \
    --fixed-kv 0.8 1.0 1.2 1.0 \
    --maxiter 500
```

### 结合多文件输入

固定 KV 参数也适用于多文件输入：

```bash
uv run python src/models/fit_multi_case_v2.py \
    --data data/augmented/dt_24hours_data_new.csv \
    --data data/augmented/dt_24hours_data_new2.csv \
    --fixed-kv 1.0 1.0 1.0 1.0 \
    --maxiter 500 \
    --output results/fixed_kv_fit.csv
```

### 仅拟合温度参数

如果同时使用 `--no-fit-struct` 和 `--fixed-kv`，则只拟合温度参数：

```bash
uv run python src/models/fit_multi_case_v2.py \
    --data data/augmented/dt_24hours_data_4.csv \
    --no-fit-struct \
    --fixed-kv 1.0 1.0 1.0 1.0 \
    --maxiter 500
```

这种情况下，沉降、EI 因子、KV 因子都使用默认值或固定值，只优化温度场。

## 参数空间对比

### 默认配置（推荐）

- 结构参数：**10 个**
  - 沉降：**3 个**（B, C, D；A 固定为 0）
  - EI 因子：3 个（3个跨）
  - KV 因子：4 个（4个支座）
- 温度参数：n_cases × 3

**总参数数 = 10 + n_cases × 3**

### 固定 KV + 固定第一个沉降（最精简）

- 结构参数：**6 个**
  - 沉降：**3 个**（B, C, D；A 固定为 0）
  - EI 因子：3 个（3个跨）
  - KV 因子：0 个（固定）
- 温度参数：n_cases × 3

**总参数数 = 6 + n_cases × 3**

**参数减少：4 个**，约减少 40% 的结构参数

### 不固定第一个沉降（不推荐）

- 结构参数：11 个
  - 沉降：**4 个**（全部拟合）
  - EI 因子：3 个
  - KV 因子：4 个
- 温度参数：n_cases × 3

**总参数数 = 11 + n_cases × 3**

⚠️ **存在参数空间平移不确定性**：所有沉降同时增加或减少相同值不会改变相对沉降，可能导致收敛问题。

## 输出说明

使用默认配置（固定 Settlement A）时，程序会显示：

```text
多工况拟合设置 (向量化版本 v2):
  工况数量: 100
  每工况约束: 4 (反力)
  总约束数: 400
  KV参数: 参与拟合
  沉降参数: Settlement_A 固定为 0.0（参考点）
  优化特性: 批量求解 + 向量化温度载荷

参数空间:
  结构参数: 10 (沉降: 3, EI: 3, KV: 4)
  温度参数: 300 (100 × 3)
  总参数数: 310
```

使用固定 KV 和固定第一个沉降时：

```text
多工况拟合设置 (向量化版本 v2):
  工况数量: 100
  每工况约束: 4 (反力)
  总约束数: 400
  KV参数: 固定为 (1.0, 1.0, 1.0, 1.0)
  沉降参数: Settlement_A 固定为 0.0（参考点）
  优化特性: 批量求解 + 向量化温度载荷

参数空间:
  结构参数: 6 (沉降: 3, EI: 3, KV: 0)
  温度参数: 300 (100 × 3)
  总参数数: 306
```

拟合结果中：

```text
拟合的结构参数:
  Settlements (mm): (0.0, 6.234, 5.567, 3.890)  # A固定为0，B/C/D拟合
  EI factors:       (0.987, 1.023, 0.956)
  Kv factors:       (1.0, 1.0, 1.0, 1.0)  # 如果使用了--fixed-kv
```

## 注意事项

1. **参数顺序**：`--fixed-kv` 的四个值必须按顺序对应 A、B、C、D 支座
2. **合理范围**：KV 因子通常在 0.1 到 3.0 之间，建议使用物理合理的值
3. **收敛性**：固定 KV 后，如果真实 KV 与固定值相差较大，可能导致其他参数补偿性偏差
4. **兼容性**：该功能与所有其他参数（`--no-fit-struct`, `--temp-spatial-weight` 等）完全兼容
5. **默认行为**：
   - **默认固定 Settlement A = 0.0**：这是推荐的做法，避免参数空间平移不确定性
   - 如需拟合全部 4 个沉降，使用 `--no-fix-first-settlement`（不推荐）
6. **相对沉降**：
   - 固定 Settlement A 后，其他沉降值表示相对于 A 的相对沉降
   - 例如：如果拟合得到 B=6.5mm，表示 B 比 A 高 6.5mm（向上）
   - 负值表示该支座比 A 低（向下）

## 典型工作流程

### 两步优化策略

1. **第一步：固定 KV，优化其他参数**

```bash
uv run python src/models/fit_multi_case_v2.py \
    --data data/your_data.csv \
    --fixed-kv 1.0 1.0 1.0 1.0 \
    --maxiter 500 \
    --output results/step1_fixed_kv.csv
```

2. **第二步：使用第一步结果作为初值，全参数优化**

修改代码中的 `x0` 初值，然后：

```bash
uv run python src/models/fit_multi_case_v2.py \
    --data data/your_data.csv \
    --maxiter 500 \
    --output results/step2_full_opt.csv
```

### 灵敏度分析

测试不同 KV 值对拟合结果的影响：

```bash
# Case 1: KV = 0.5
uv run python src/models/fit_multi_case_v2.py \
    --data data/your_data.csv \
    --fixed-kv 0.5 0.5 0.5 0.5 \
    --output results/kv_0.5.csv

# Case 2: KV = 1.0
uv run python src/models/fit_multi_case_v2.py \
    --data data/your_data.csv \
    --fixed-kv 1.0 1.0 1.0 1.0 \
    --output results/kv_1.0.csv

# Case 3: KV = 2.0
uv run python src/models/fit_multi_case_v2.py \
    --data data/your_data.csv \
    --fixed-kv 2.0 2.0 2.0 2.0 \
    --output results/kv_2.0.csv
```

然后比较不同 KV 值下的拟合质量和其他参数的变化。

## 代码实现细节

固定 KV 的实现涉及三个主要修改：

1. **参数空间调整**：`_build_x0_and_bounds()` 中，当提供 `fixed_kv_factors` 时，不再为 KV 分配优化变量
2. **参数解包**：`_unpack_params()` 中，使用固定的 KV 值而不是从优化变量中提取
3. **系统缓存**：由于 KV 是固定的，当其他结构参数不变时，系统矩阵可以完全复用

这些修改确保了：
- 优化器不会尝试改变 KV 值
- 计算效率保持高效
- 代码向后兼容（不提供 `--fixed-kv` 时行为不变）
