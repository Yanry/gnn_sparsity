#!/bin/bash
# 初始化 GEMM Hook 输出目录结构

OUTPUT_DIR="${1:-$(dirname "$0")/output}"

echo "初始化 GEMM Hook 输出目录..."
echo "输出目录: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

# 创建示例子目录结构
configs=("dense" "gmp" "sp50" "2:4" "sp50_q4")
datasets=("wikitext2" "ptb" "c4")
model="opt-125m"

for config in "${configs[@]}"; do
    for dataset in "${datasets[@]}"; do
        dir_name="${OUTPUT_DIR}/${model}_${config}_${dataset}"
        mkdir -p "$dir_name"
        echo "✓ 创建目录: $dir_name"
    done
done

echo ""
echo "目录结构已初始化："
echo ""
tree -L 2 "$OUTPUT_DIR" 2>/dev/null || {
    echo "$OUTPUT_DIR"
    for config in "${configs[@]}"; do
        for dataset in "${datasets[@]}"; do
            echo "├── ${model}_${config}_${dataset}/"
        done
    done
}

echo ""
echo "总共创建了 $(echo "${configs[@]}" | wc -w) × $(echo "${datasets[@]}" | wc -w) = $((${#configs[@]} * ${#datasets[@]})) 个评估目录"
echo ""
echo "准备就绪！现在可以运行评估脚本了："
echo "  python eval_with_gemm_hook.py <model_path> --output-dir $OUTPUT_DIR"
