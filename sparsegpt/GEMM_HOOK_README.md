# GEMM Hook 评估工具使用指南

## 概述

这套工具用于在模型评估(eval)阶段，hook 并记录所有 GEMM (General Matrix Multiply) 操作的输入张量信息，以及稀疏性统计。

## 核心功能

- **Hook 所有线性层(Linear)** - 记录输入张量和权重的形状、稀疏性
- **多配置评估** - 支持 5 种不同的压缩配置
- **多数据集** - 在 3 种数据集上进行评估
- **多次推理** - 每个配置×数据集组合运行 10 次推理
- **详细统计** - 记录每次推理的 GEMM 操作统计和 PPL 结果

## 文件说明

### 核心文件

- **gemm_hook.py** - GEMM Hook 管理器，负责 hook 注册、数据收集和保存
- **eval_with_gemm_hook.py** - 主评估脚本，运行完整的评估流程
- **run_gemm_eval.sh** - 便捷启动脚本
- **analyze_gemm_results.py** - 结果分析和可视化脚本

## 使用方法

### 基础用法

最简单的用法（使用默认参数）:

```bash
cd /home/nizhj/gnn_sparsity/sparsegpt

# 方法 1: 直接运行 Python 脚本
python eval_with_gemm_hook.py /home/zhaojun/proj26/models/opt-125m \
    --data-root /home/zhaojun/proj26/datasets

# 方法 2: 使用 shell 脚本
bash run_gemm_eval.sh /home/zhaojun/proj26/models/opt-125m /home/zhaojun/proj26/datasets
```

### 高级用法

指定输出目录和推理次数:

```bash
python eval_with_gemm_hook.py /home/zhaojun/proj26/models/opt-125m \
    --data-root /home/zhaojun/proj26/datasets \
    --output-dir /home/nizhj/gnn_sparsity/sparsegpt/output \
    --num-inferences 10
```

### 参数说明

```
positional arguments:
  model                 模型路径或 HuggingFace ID (e.g., /path/to/model 或 facebook/opt-125m)

optional arguments:
  --num-inferences      每个配置运行的推理次数，默认: 10
  --data-root          本地数据集根目录，默认: /home/zhaojun/proj26/datasets
  --output-dir         输出目录，默认: ./output
```

## 5 种压缩配置

脚本自动评估以下 5 种配置:

1. **dense** - 无压缩基线
   - 原始未压缩的模型

2. **gmp** - 幅度剪枝基准 (50%)
   - 使用 Gradient Magnitude Pruning 方法
   - 目标稀疏性: 50%

3. **sp50** - SparseGPT 50% 稀疏性
   - 使用 SparseGPT 方法进行一次性剪枝
   - 目标稀疏性: 50%

4. **2:4** - SparseGPT 2:4 结构化稀疏性
   - 遵循 NVIDIA 的 2:4 稀疏性格式
   - 每 4 个权重中有 2 个为零

5. **sp50_q4** - SparseGPT 50% + 4-bit 量化
   - 结合 50% 稀疏性和 4-bit 权重量化

## 3 种评估数据集

- **wikitext2** - WikiText-2 数据集
- **ptb** - Penn Treebank 数据集  
- **c4** - C4 子集数据集

## 输出目录结构

```
output/
├── opt-125m_dense_wikitext2/
│   ├── evaluation_summary.json          # 评估摘要
│   ├── inference_001/
│   │   ├── gemm_stats.json             # GEMM 操作统计
│   │   └── metadata.json               # 元数据和 PPL
│   ├── inference_002/
│   └── ... inference_010/
├── opt-125m_dense_ptb/
├── opt-125m_dense_c4/
├── opt-125m_gmp_wikitext2/
├── ... (remaining 15 combinations)
```

### 输出文件说明

**evaluation_summary.json** - 包含:
```json
{
  "model": "opt-125m",
  "config": "dense",
  "dataset": "wikitext2",
  "ppl_results": [13.45, 13.48, 13.46, ...],
  "average_ppl": 13.46,
  "min_ppl": 13.45,
  "max_ppl": 13.48,
  "total_inferences": 10
}
```

**inference_XXX/metadata.json** - 包含:
```json
{
  "model": "opt-125m",
  "config": "dense",
  "dataset": "wikitext2",
  "ppl": 13.45,
  "timestamp": "2026-03-04T10:30:45.123456",
  "inference_number": 1,
  "total_gemm_operations": 1920
}
```

**inference_XXX/gemm_stats.json** - 包含所有 GEMM 操作的统计:
```json
{
  "linear_0_model.decoder.layers.0.self_attn.q_proj": {
    "input_shape": [1, 2048, 768],
    "weight_shape": [768, 768],
    "input_format": "float32",
    "input_sparsity": 0.15,
    "weight_sparsity": 0.0
  },
  ...
}
```

## 结果分析

### 查看评估结果

```bash
# 详细分析结果
python analyze_gemm_results.py --output-dir ./output

# 导出为 CSV 格式
python analyze_gemm_results.py --output-dir ./output --export-csv results.csv
```

### 结果示例输出

```
================================================================================
GEMM Hook 评估结果分析
================================================================================

模型                 | 配置       | 数据集     |     平均PPL |  推理次数 |  GEMM操作数
--------------------------------------------------------------------------------
opt-125m             | dense      | c4         |      10.2839 |        10 |      1920
opt-125m             | dense      | ptb        |      63.5234 |        10 |      1920
opt-125m             | dense      | wikitext2  |      21.8934 |        10 |      1920
opt-125m             | 2:4        | c4         |      10.3921 |        10 |      1920
...
```

## 计算复杂度

- 总评估组合数: 5 configs × 3 datasets × 10 inferences = **150 次评估**
- 每次评估的 GEMM 操作: 约 1920 个
- 总 GEMM 操作记录: 150 × 1920 = **288,000** 条记录

预计耗时:
- 单次推理: ~5-10 秒 (取决于硬件)
- 全部评估: ~12-20 小时

**建议在后台运行:**

```bash
# 使用 nohup 在后台运行
nohup python eval_with_gemm_hook.py /home/zhaojun/proj26/models/opt-125m \
    --data-root /home/zhaojun/proj26/datasets > eval_log.txt 2>&1 &

# 或在 tmux/screen 中运行
bash run_gemm_eval.sh
```

## 故障排除

### 问题 1: 内存不足

解决方案:
- 减少 `--num-inferences` 参数
- 使用更小的模型

### 问题 2: 数据集加载失败

解决方案:
- 检查 `--data-root` 路径是否正确
- 确保数据集文件完整

### 问题 3: CUDA 错误

解决方案:
- 检查 GPU 是否可用: `nvidia-smi`
- 清理 GPU 缓存: 重启脚本

## 在评估代码中集成

如果想在现有评估代码中集成 GEMM Hook：

```python
from gemm_hook import GEMMHookManager

# 创建 hook 管理器
hook_mgr = GEMMHookManager(
    output_dir='./output',
    model_name='opt-125m',
    config_name='dense',
    dataset_name='wikitext2'
)

# 为每次推理创建新目录
hook_mgr.create_inference_dir()

# 注册 hook
hook_mgr.register_hooks(model)

# ... 执行评估代码 ...

# 移除 hook
hook_mgr.remove_hooks()

# 保存结果
hook_mgr.save_inference_data(ppl_value)
```

## 参考

- SparseGPT 论文: https://arxiv.org/abs/2301.00774
- 本工具源代码位置: `/home/nizhj/gnn_sparsity/sparsegpt/`
