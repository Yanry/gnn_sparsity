# 🚀 GEMM Hook 快速启动指南

## 创建的完整文件清单

### 核心评估脚本
```
/home/nizhj/gnn_sparsity/sparsegpt/
├── gemm_hook.py                  (238 行) - GEMM Hook 管理器
├── eval_with_gemm_hook.py        (180 行) - 主评估脚本
├── analyze_gemm_results.py       (160 行) - 结果分析脚本
├── run_gemm_eval.sh              (56 行)  - 启动脚本
└── init_gemm_output.sh           (45 行)  - 目录初始化脚本
```

### 文档
```
├── GEMM_HOOK_README.md          - 详细用户手册
├── GEMM_HOOK_QUICK_REF.md       - 快速参考卡
└── IMPLEMENTATION_SUMMARY.md    - 实现总结
```

## 🎯 开始使用（3步）

### Step 1: 初始化输出目录

```bash
cd /home/nizhj/gnn_sparsity/sparsegpt
bash init_gemm_output.sh
```

✓ 创建 15 个评估目录 (5 configs × 3 datasets)

### Step 2: 运行评估（使用 bash 脚本）

```bash
bash run_gemm_eval.sh /home/zhaojun/proj26/models/opt-125m \
                      /home/zhaojun/proj26/datasets
```

**或者直接用 Python：**

```bash
python eval_with_gemm_hook.py /home/zhaojun/proj26/models/opt-125m \
    --data-root /home/zhaojun/proj26/datasets \
    --num-inferences 10
```

✓ 自动评估 5 种配置 × 3 种数据集 × 10 次推理

### Step 3: 分析结果

```bash
# 查看详细分析
python analyze_gemm_results.py --output-dir ./output

# 导出为 CSV 便于分析
python analyze_gemm_results.py --output-dir ./output --export-csv results.csv
```

## 📋 5 种自动评估的压缩配置

| # | 配置名 | 说明 | 稀疏性 |
|---|--------|------|--------|
| 1 | **dense** | 无压缩基线 | 0% |
| 2 | **gmp** | 幅度剪枝基准 | 50% |
| 3 | **sp50** | SparseGPT 均匀剪枝 | 50% |
| 4 | **2:4** | SparseGPT 结构化剪枝 | 50% |
| 5 | **sp50_q4** | SparseGPT + 4-bit 量化 | 50% |

## 🗂️ 3 种评估数据集

- **wikitext2** - 常用 NLP 基准
- **ptb** - Penn Treebank 标准数据集
- **c4** - 大规模网页文本

## 📊 生成的输出结构

```
output/
└── opt-125m_config_dataset/           (15 个这样的目录)
    ├── evaluation_summary.json         ← 评估汇总（平均 PPL 等）
    ├── inference_001/
    │   ├── gemm_stats.json            ← GEMM 操作统计
    │   └── metadata.json              ← 元数据和 PPL
    ├── inference_002/
    │   ├── gemm_stats.json
    │   └── metadata.json
    └── ... inference_010/
```

## ⏱️ 性能预期

| 指标 | 值 |
|------|-----|
| 单次推理 | 5-10 秒 |
| 单个组合 (dense+wikitext2) | 50-100 秒 |
| 全部 15 组合 | 12-20 小时 |
| **总推理次数** | **150 次** |
| **总 GEMM 操作记录** | **288,000+** |

## 💾 每个推理生成的数据

### gemm_stats.json 示例
```json
{
  "linear_0_model.decoder.layers.0.self_attn.q_proj": {
    "input_shape": [1, 2048, 768],
    "weight_shape": [768, 768],
    "input_sparsity": 0.15,
    "weight_sparsity": 0.0
  },
  // ... 1920 个这样的 GEMM 操作
}
```

### metadata.json 示例
```json
{
  "model": "opt-125m",
  "config": "dense",
  "dataset": "wikitext2",
  "ppl": 21.89,
  "total_gemm_operations": 1920,
  "inference_number": 1
}
```

## 🔧 高级用法

### 自定义推理次数
```bash
python eval_with_gemm_hook.py model_path --num-inferences 5
```

### 自定义输出目录
```bash
python eval_with_gemm_hook.py model_path --output-dir /custom/path
```

### 后台运行（推荐）
```bash
# 方式 1: nohup
nohup python eval_with_gemm_hook.py model_path > eval.log 2>&1 &

# 方式 2: tmux
tmux new-session -d -s eval "python eval_with_gemm_hook.py model_path"

# 查看进度
tail -f eval.log
```

### 并行运行多个模型
```bash
# 终端 1
python eval_with_gemm_hook.py model1 --output-dir output1 &

# 终端 2  
python eval_with_gemm_hook.py model2 --output-dir output2 &
```

## 📈 结果分析

### 查看汇总统计
```bash
python analyze_gemm_results.py --output-dir ./output
```

输出示例：
```
模型                 | 配置       | 数据集     |     平均PPL |  推理次数 |  GEMM操作数
================================================================================
opt-125m             | dense      | c4         |      10.2839 |        10 |      1920
opt-125m             | dense      | ptb        |      63.5234 |        10 |      1920
opt-125m             | dense      | wikitext2  |      21.8934 |        10 |      1920
opt-125m             | gmp        | c4         |      10.3921 |        10 |      1920
...
```

### 导出为 CSV 进行高级分析
```bash
python analyze_gemm_results.py --output-dir ./output --export-csv results.csv

# 然后在 Excel/Python 中打开 results.csv 进行分析
```

## 🐛 常见问题

### Q: 推理太慢怎么办？
**A:** 
- 检查 GPU 使用率: `nvidia-smi`
- 减少 `--num-inferences`
- 使用更小的模型

### Q: 数据集加载失败？
**A:**
- 检查 `--data-root` 路径
- 确认数据集文件完整
- 查看之前的 opt.py 输出是否有错误

### Q: 如何恢复已完成的推理？
**A:** 脚本会跳过已存在的 `inference_*` 目录，自动继续

### Q: 能否中途停止再继续？
**A:** 是的，脚本支持增量式运行。停止后再次运行会从中断处继续

## 📚 查看详细文档

| 文档 | 用途 |
|------|------|
| **GEMM_HOOK_QUICK_REF.md** | 这里（快速参考） |
| **GEMM_HOOK_README.md** | 完整功能说明和故障排除 |
| **IMPLEMENTATION_SUMMARY.md** | 实现细节和设计说明 |

## 🎓 技术细节

### 什么是 Hook？
Hook 是 PyTorch 中的机制，可以在 forward/backward 时拦截层的输入和输出。

### 如何计算稀疏性？
```
稀疏性 = (零元素数) / (总元素数)
例如: 1000 中有 500 个 0 → 稀疏性 = 50%
```

### 为什么要记录 GEMM 操作？
- 分析压缩对计算模式的影响
- 优化硬件加速器设计
- 验证稀疏性声称的准确性

## ✨ 关键特性

✅ **全自动** - 一条命令完成全部评估  
✅ **详细记录** - 逐次推理保存 GEMM 操作  
✅ **易于分析** - JSON 格式便于后处理  
✅ **高度灵活** - 支持自定义配置和参数  
✅ **与现有代码兼容** - 无需修改原有脚本  

## 🚀 现在就开始吧！

```bash
# 1. 进入目录
cd /home/nizhj/gnn_sparsity/sparsegpt

# 2. 初始化
bash init_gemm_output.sh

# 3. 运行评估（最简单的方式）
bash run_gemm_eval.sh /home/zhaojun/proj26/models/opt-125m \
                      /home/zhaojun/proj26/datasets

# 4. 分析结果
python analyze_gemm_results.py --output-dir ./output
```

**预期结果：** 150 次评估完成，288,000+ GEMM 操作数据记录

---

## 📝 脚本参数速查

```bash
# eval_with_gemm_hook.py 参数
model                      - 模型路径或 HuggingFace ID (必需)
--num-inferences N         - 推理次数 (默认: 10)
--data-root PATH          - 数据集根目录 (默认: /home/zhaojun/proj26/datasets)
--output-dir PATH         - 输出目录 (默认: ./output)

# analyze_gemm_results.py 参数
--output-dir PATH         - 分析目录
--export-csv FILE         - 导出为 CSV
```

## 💡 建议

- 第一次运行时用较少推理次数测试（如 2-3 次）
- 保留日志文件便于调试
- 定期备份 output/ 目录
- 在高性能 GPU 上运行以加快速度

---

**⏰ 开始时间估计：** 12-20 小时（取决于硬件）  
**💾 磁盘空间：** ~1-2 GB（取决于数据集）  
**🖥️ 内存需求：** <1 GB
