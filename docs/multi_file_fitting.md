# 多文件联合拟合功能说明

## 概述

`fit_multi_case_v2.py` 现在支持多个数据文件的联合拟合，实现了跨文件边界的时序平滑约束。这使得可以处理分段存储的长时间序列数据，同时保持温度梯度在整个时间序列上的连续性。

## 主要特性

### 1. 多文件输入
- 通过 `--data` 参数多次指定，支持任意数量的数据文件
- 文件按指定顺序拼接成连续的时间序列
- `sample_id` 自动重新编号为连续序列

### 2. 跨文件边界的时序平滑约束
- 温度梯度的时序平滑约束 (temporal regularization) 应用于所有相邻样本
- **关键**：约束自然地作用于文件边界处（第一个文件的最后一个样本和第二个文件的第一个样本）
- 确保整个时间序列的温度变化平滑，没有跳变

### 3. 结构参数一致性验证
- 自动验证所有文件的结构参数（沉降、EI因子、Kv因子）是否一致
- 如果参数不一致会给出警告

## 使用方法

### 基本用法

```bash
# 使用两个文件进行拟合
uv run python src/models/fit_multi_case_v2.py \
    --data data/augmented/dt_24hours_data_new.csv \
    --data data/augmented/dt_24hours_data_new2.csv \
    --maxiter 500 \
    --output results/multi_file_fit.csv
```

### 使用三个或更多文件

```bash
uv run python src/models/fit_multi_case_v2.py \
    --data file1.csv \
    --data file2.csv \
    --data file3.csv \
    --maxiter 500
```

### 其他参数

```bash
uv run python src/models/fit_multi_case_v2.py \
    --data data/augmented/dt_24hours_data_new.csv \
    --data data/augmented/dt_24hours_data_new2.csv \
    --max-samples 100 \              # 限制总样本数（拼接后）
    --no-fit-struct \                 # 固定结构参数，仅拟合温度
    --maxiter 500 \                   # 最大迭代次数
    --temp-spatial-weight 1.0 \       # 空间平滑权重
    --temp-temporal-weight 1.0 \      # 时序平滑权重
    --temp-basis fourier \            # (可选) 使用傅里叶降维
    --fourier-harmonics 3 \           # (可选) 谐波次数，默认3
    --output results/output.csv       # 输出文件
```

### 温度降维（傅里叶基）

从 `bridge_fitting` CLI 版本开始，温度梯度支持傅里叶级数降维：

```bash
uv run python src/models/bridge_fitting/cli.py \
    --data data/day1.csv \
    --temp-basis fourier \
    --fourier-harmonics 3 \
    --fourier-period 1440 \
    --fourier-time-column minute_of_day
```

- `--temp-basis fourier` 启用傅里叶参数化（默认沿用逐工况温度）
- `--fourier-harmonics` 控制最高谐波次数（`k=3` 时每条温度曲线仅需 7 个系数）
- `--fourier-period` 指定完整周期（默认等于样本数，适用于日循环）
- `--fourier-time-column` 可选，用于提供实际时间戳列；未指定时使用样本顺序

该方法将原本 1440 × 3 个温度变量压缩为 3 × (2k+1) 个系数，大幅降低优化维度，同时自然保证时间上的平滑性。

## 工作原理

### 数据拼接流程

1. **读取文件**：按指定顺序读取所有CSV文件
2. **拼接**：使用 `pd.concat(dfs, ignore_index=True)` 拼接
3. **重新编号**：`df['sample_id'] = range(len(df))` 生成连续序列
4. **验证**：检查结构参数的一致性

### 时序约束的实现

在 `_residual()` 方法中：

```python
# 3. Vectorized temporal regularization
if self.temp_temporal_weight > 0:
    # Compute differences between consecutive cases
    # NOTE: This works across file boundaries when multiple files are concatenated,
    # ensuring smooth temperature transitions throughout the entire time series
    # dT_matrix: (n_cases, 3), we want dT[i+1] - dT[i] for each span
    dT_diff = dT_matrix[1:, :] - dT_matrix[:-1, :]  # (n_cases-1, 3)
    
    # Soft L1 penalty: max(0, |diff| - 3.0)
    temporal_penalty = np.maximum(0.0, np.abs(dT_diff) - 3.0) * self.temp_temporal_weight
    
    # Flatten: (n_cases-1, 3) -> ((n_cases-1) * 3,)
    self._temporal_residuals[:] = temporal_penalty.ravel()
```

**关键点**：
- `dT_matrix` 包含所有文件的温度数据，已经拼接成连续数组
- `dT_diff = dT_matrix[1:, :] - dT_matrix[:-1, :]` 计算所有相邻样本的差分
- 这自然地包括了文件边界处的约束（例如：样本 1439 和样本 1440 之间，如果第一个文件有 1440 个样本）

## 示例脚本

使用 `scripts/run_multi_file_fit.sh` 快速开始：

```bash
bash scripts/run_multi_file_fit.sh
```

该脚本包含两个示例：
1. 固定结构参数，仅拟合温度（前50个样本）
2. 联合拟合结构和温度参数（前30个样本）

## 实际测试结果

使用 `dt_24hours_data_new.csv` (1440样本) 和 `dt_24hours_data_new2.csv` (1440样本)：

```
加载 2 个数据文件:
  文件 1: dt_24hours_data_new.csv (1440 条数据)
  文件 2: dt_24hours_data_new2.csv (1440 条数据)

合并后数据集:
  总样本数: 2880
  结构参数验证: ✓ 全部为常数

多工况拟合设置 (向量化版本 v2):
  工况数量: 2880
  每工况约束: 4 (反力)
  总约束数: 11520
  优化特性: 批量求解 + 向量化温度载荷
```

## 优势

1. **数据规模灵活**：可以使用任意数量的文件，不受单个文件大小限制
2. **时序连续性**：自动确保跨文件边界的温度梯度平滑
3. **更多数据点**：联合优化可以利用更多数据点，提高拟合精度
4. **易于使用**：只需多次指定 `--data` 参数，无需手动合并文件

## 适用场景

- 长时间序列数据分段存储（如：每天一个文件）
- 需要更多数据点以提高拟合精度
- 多批次实验或监测数据的联合分析
- 保持时间序列连续性的大规模数据处理

## 注意事项

1. **结构参数一致性**：所有文件的结构参数（沉降、EI因子、Kv因子）必须相同
2. **内存使用**：拼接后的数据会全部加载到内存，注意总数据量
3. **文件顺序**：文件拼接顺序很重要，确保按时间顺序指定
4. **时序约束**：`--temp-temporal-weight` 控制时序平滑程度，默认为 1.0

## 技术实现细节

### 数据结构
- `dT_matrix`: shape `(n_cases, 3)` - 所有样本的温度梯度矩阵
- `reactions_matrix`: shape `(n_cases, 4)` - 所有样本的反力测量值

### 约束类型
1. **物理约束** (Physics residuals): `n_cases × 4` 个残差
2. **空间平滑约束** (Spatial regularization): `n_cases × 2` 个残差
3. **时序平滑约束** (Temporal regularization): `(n_cases - 1) × 3` 个残差

### 性能优化
- 批量求解：所有工况同时求解，使用BLAS Level 3
- 向量化温度载荷计算
- 系统矩阵缓存：结构参数不变时重用LU分解
