#!/bin/bash
# 多文件联合拟合示例脚本
# 演示如何使用多个数据文件进行时序约束优化

set -e

echo "============================================"
echo "多文件温度梯度拟合示例"
echo "============================================"
echo ""

# 数据文件
FILE1="data/augmented/dt_24hours_data_new.csv"
FILE2="data/augmented/dt_24hours_data_new2.csv"

# 输出目录
OUTPUT_DIR="results/multi_file_fit"
mkdir -p "$OUTPUT_DIR"

echo "数据文件:"
echo "  1. $FILE1"
echo "  2. $FILE2"
echo ""

# 示例 1: 只拟合温度，结构参数固定
echo "示例 1: 固定结构参数，仅拟合温度梯度 (前50个样本)"
echo "----------------------------------------"
uv run python src/models/fit_multi_case_v2.py \
    --data "$FILE1" \
    --data "$FILE2" \
    --max-samples 50 \
    --no-fit-struct \
    --maxiter 100 \
    --temp-spatial-weight 1.0 \
    --temp-temporal-weight 1.0 \
    --output "$OUTPUT_DIR/temp_only_50samples.csv"

echo ""
echo "示例 2: 联合拟合结构和温度参数 (前30个样本)"
echo "----------------------------------------"
uv run python src/models/fit_multi_case_v2.py \
    --data "$FILE1" \
    --data "$FILE2" \
    --max-samples 30 \
    --maxiter 200 \
    --temp-spatial-weight 1.0 \
    --temp-temporal-weight 1.0 \
    --output "$OUTPUT_DIR/joint_fit_30samples.csv"

echo ""
echo "============================================"
echo "拟合完成！"
echo "结果保存在: $OUTPUT_DIR/"
echo "============================================"
