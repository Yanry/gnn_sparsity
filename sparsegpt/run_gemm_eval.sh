#!/bin/bash
# 使用 GEMM Hook 进行完整模型评估
# 脚本会对 5 种配置 × 3 种数据集 × 5 次推理进行评估
# 总耗时较长，建议在后台运行

set -e

MODEL_PATH="${1:-/home/zhaojun/proj26/models/opt-125m}"
DATA_ROOT="${2:-/home/zhaojun/proj26/datasets}"
OUTPUT_DIR="${3:-$(dirname "$0")/output}"
NUM_INFERENCES="${4:-1}"

echo "=========================================="
echo "SparseGPT GEMM Hook 完整评估脚本"
echo "=========================================="
echo ""
echo "配置信息："
echo "  模型路径:     $MODEL_PATH"
echo "  数据集根目录: $DATA_ROOT"
echo "  输出目录:     $OUTPUT_DIR"
echo "  推理次数:     $NUM_INFERENCES"
echo ""
echo "评估组合数: 3 configs × 3 datasets × $NUM_INFERENCES inferences = $((4 * 3 * NUM_INFERENCES)) 次评估"
echo ""
echo "配置列表:"
echo "  1. gmp     - 幅度剪枝基准 (50%)"
echo "  2. sp50    - SparseGPT 50% 稀疏性"
echo "  3. 2:4     - SparseGPT 2:4 稀疏性"
echo ""
echo "数据集列表:"
echo "  1. wikitext2"
echo "  2. ptb"
echo "  3. c4"
echo ""
echo "=========================================="
echo "开始评估... ($(date))"
echo "=========================================="
echo ""

cd "$(dirname "$0")"

python eval_with_gemm_hook.py \
    "$MODEL_PATH" \
    --num-inferences "$NUM_INFERENCES" \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "评估完成！($(date))"
echo "=========================================="
echo "结果保存目录: $OUTPUT_DIR"
echo "请查看结果文件了解详情"
