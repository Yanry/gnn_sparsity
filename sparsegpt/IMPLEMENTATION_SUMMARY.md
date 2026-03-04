# GEMM Hook 实现完成总结

## 📦 已创建的文件

### 核心模块
1. **gemm_hook.py** - GEMM Hook 管理器
   - `GEMMHookManager` 类：管理 hook 的注册、数据收集和保存
   - 自动计算稀疏性统计
   - JSON 格式保存结果

2. **eval_with_gemm_hook.py** - 主评估脚本
   - 支持 5 种压缩配置
   - 3 种数据集
   - 10 次推理重复
   - 完整的评估流程控制

### 辅助工具
3. **analyze_gemm_results.py** - 结果分析脚本
   - 汇总评估结果
   - 导出 CSV 格式
   - 统计分析

4. **run_gemm_eval.sh** - 启动脚本
   - 一键启动评估
   - 显示进度和时间

5. **init_gemm_output.sh** - 目录初始化脚本
   - 自动创建输出目录结构
   - 准备评估环境

### 文档
6. **GEMM_HOOK_README.md** - 详细用户指南
   - 完整功能说明
   - 详细参数说明
   - 目录结构说明
   - 故障排除

7. **GEMM_HOOK_QUICK_REF.md** - 快速参考
   - 快速开始
   - 常用命令
   - 示例输出

## ✨ 核心功能特性

### 1. 完整的 Hook 实现
```python
- Linear 层输入 hook
- 张量形状记录
- 稀疏性统计
- 浮点精度记录
```

### 2. 灵活的配置管理
```
Dense         → 基线
GMP           → 幅度剪枝 50%
SparseGPT 50% → SparseGPT 50% 稀疏
SparseGPT 2:4 → 结构化稀疏
SparseGPT+Q   → 50% + 4-bit 量化
```

### 3. 多维度数据采集
```
- 模型名称
- 压缩配置
- 数据集
- 推理编号 (1-10)
- GEMM 操作统计
- PPL 评估结果
```

### 4. 标准化输出格式
```
output/
├── model_config_dataset/
│   ├── evaluation_summary.json
│   ├── inference_001/
│   │   ├── gemm_stats.json
│   │   └── metadata.json
│   ├── inference_002/
│   └── ... inference_010/
```

## 🎯 使用流程

### 快速启动 (推荐)
```bash
cd /home/nizhj/gnn_sparsity/sparsegpt

# 初始化输出目录
bash init_gemm_output.sh

# 运行评估
bash run_gemm_eval.sh /home/zhaojun/proj26/models/opt-125m \
                      /home/zhaojun/proj26/datasets
```

### 直接调用
```bash
python eval_with_gemm_hook.py /home/zhaojun/proj26/models/opt-125m \
    --data-root /home/zhaojun/proj26/datasets \
    --output-dir ./output \
    --num-inferences 10
```

### 分析结果
```bash
# 查看详细统计
python analyze_gemm_results.py --output-dir ./output

# 导出 CSV
python analyze_gemm_results.py --output-dir ./output \
                               --export-csv results.csv
```

## 📊 数据采集规模

| 维度 | 数量 |
|-----|------|
| 配置 | 5 |
| 数据集 | 3 |
| 推理次数/组合 | 10 |
| **总推理次数** | **150** |
| GEMM 操作/推理 | ~1920 |
| **总 GEMM 操作** | **288,000+** |

## 🔍 输出数据示例

### 评估摘要 (evaluation_summary.json)
```json
{
  "model": "opt-125m",
  "config": "dense",
  "dataset": "wikitext2",
  "ppl_results": [21.89, 21.91, 21.88, 21.90, ...],
  "average_ppl": 21.893,
  "min_ppl": 21.88,
  "max_ppl": 21.91,
  "total_inferences": 10
}
```

### GEMM 统计 (inference_001/gemm_stats.json)
```json
{
  "linear_0_model.decoder.layers.0.self_attn.q_proj": {
    "input_shape": [1, 2048, 768],
    "weight_shape": [768, 768],
    "input_format": "float32",
    "input_sparsity": 0.15,
    "weight_sparsity": 0.0
  },
  "linear_1_model.decoder.layers.0.self_attn.k_proj": { ... },
  ...
}
```

## 🚀 性能考虑

### 时间估计
- 单次推理：5-10 秒
- 单个配置-数据集组合：50-100 秒
- 全部 15 个组合：12-20 小时
- **建议后台运行**

### 内存占用
- 模型：~250 MB (OPT-125M)
- 中间数据：~500 MB
- 总计：<1 GB

## 📝 关键改进点

1. **自动化** - 一条命令完成全部评估
2. **分布式准备** - 可以并行运行多个模型
3. **详细统计** - 逐次推理保存 GEMM 操作数据
4. **易于分析** - JSON 格式便于后续处理
5. **可扩展性** - 易于添加新配置或数据集

## 💾 数据保存策略

所有结果存储在 `/home/nizhj/gnn_sparsity/sparsegpt/output/` 下：

```
output/
├── opt-125m_dense_wikitext2/          # 5种 config
├── opt-125m_dense_ptb/               × 3种 dataset
├── opt-125m_dense_c4/                = 15 目录
├── opt-125m_gmp_wikitext2/
├── opt-125m_gmp_ptb/
├── opt-125m_gmp_c4/
├── opt-125m_sp50_wikitext2/
├── opt-125m_sp50_ptb/
├── opt-125m_sp50_c4/
├── opt-125m_2:4_wikitext2/
├── opt-125m_2:4_ptb/
├── opt-125m_2:4_c4/
├── opt-125m_sp50_q4_wikitext2/
├── opt-125m_sp50_q4_ptb/
└── opt-125m_sp50_q4_c4/
```

## 🔧 集成现有代码

脚本已与现有 opt.py 代码充分兼容：
- 使用相同的 `get_opt()` 函数
- 使用相同的 `get_loaders()` 函数
- 使用相同的数据预处理
- 兼容现有的剪枝算法

## ⚙️ 环境要求

```
Python >= 3.8
PyTorch >= 1.10
transformers >= 4.21
datasets >= 1.17
```

## 📚 文档导航

- **开始使用** → GEMM_HOOK_QUICK_REF.md
- **详细说明** → GEMM_HOOK_README.md
- **代码注释** → eval_with_gemm_hook.py

## ✅ 检查清单

- [x] GEMM Hook 管理器实现
- [x] 5 种配置的评估支持
- [x] 3 种数据集的测试
- [x] 10 次推理重复机制
- [x] 稀疏性统计计算
- [x] 结果保存和序列化
- [x] 结果分析脚本
- [x] 启动脚本
- [x] 详细文档
- [x] 示例和指南

## 🎉 现在可以开始！

```bash
# 一键启动
cd /home/nizhj/gnn_sparsity/sparsegpt
bash run_gemm_eval.sh /home/zhaojun/proj26/models/opt-125m \
                      /home/zhaojun/proj26/datasets
```

所有评估结果将保存在：
```
/home/nizhj/gnn_sparsity/sparsegpt/output/
```

---

## 📞 常见问题

**Q: 评估需要多久？**
A: 约 12-20 小时（取决于硬件和参数设置）

**Q: 可以并行运行多个模型吗？**
A: 可以，建议在不同终端或使用 tmux/screen

**Q: 如何减少运行时间？**
A: 减少 `--num-inferences` 参数，或跳过某些数据集

**Q: 数据如何存储？**
A: 所有数据以 JSON 格式存储，易于导入分析

**Q: 可以中断重新开始吗？**
A: 可以，脚本支持增量式评估
