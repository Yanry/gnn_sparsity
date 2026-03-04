# GEMM Hook 快速参考

## 🚀 快速开始

### 最简单的方式 - 一行命令

```bash
cd /home/nizhj/gnn_sparsity/sparsegpt
python eval_with_gemm_hook.py /home/zhaojun/proj26/models/opt-125m \
    --data-root /home/zhaojun/proj26/datasets
```

### 特点

✓ 自动评估 **5 种压缩配置**  
✓ 在 **3 种数据集** 上测试  
✓ 进行 **10 次推理** 重复  
✓ 记录每次推理的 **所有 GEMM 操作**  
✓ 保存 **稀疏性统计** 和 **PPL 结果**  

## 📊 输出结果

存储位置: `/home/nizhj/gnn_sparsity/sparsegpt/output/opt-125m_config_dataset/`

每个目录包含:
- `evaluation_summary.json` - 整体评估统计
- `inference_001/` 到 `inference_010/` - 10 次推理的详细数据
  - `gemm_stats.json` - GEMM 操作统计
  - `metadata.json` - PPL 和元数据

## 📈 查看结果

```bash
# 分析并打印结果
python analyze_gemm_results.py --output-dir ./output

# 导出为 CSV
python analyze_gemm_results.py --output-dir ./output --export-csv results.csv
```

## ⏱️ 运行时间

- 单个推理: ~5-10 秒
- 全部评估: ~12-20 小时
- **总计**: 150 次评估

## 🎯 5 种压缩配置

| 配置   | 说明                          | 稀疏性 |
|--------|-------------------------------|--------|
| dense  | 无压缩基线                    | 0%     |
| gmp    | 幅度剪枝基准                  | 50%    |
| sp50   | SparseGPT 50% 稀疏性          | 50%    |
| 2:4    | SparseGPT 2:4 结构化稀疏      | 50%    |
| sp50_q4| SparseGPT 50% + 4-bit 量化   | 50%    |

## 🔍 3 种评估数据集

- **wikitext2** - 21GB，常用基准数据集
- **ptb** - Penn Treebank，NLP 标准数据集
- **c4** - 大规模网页文本数据集

## 🛠️ 高级用法

### 自定义推理次数

```bash
python eval_with_gemm_hook.py /home/zhaojun/proj26/models/opt-125m \
    --num-inferences 5
```

### 自定义输出目录

```bash
python eval_with_gemm_hook.py /home/zhaojun/proj26/models/opt-125m \
    --output-dir /path/to/output
```

### 后台运行

```bash
# 使用 nohup
nohup python eval_with_gemm_hook.py ... > eval.log 2>&1 &

# 查看日志
tail -f eval.log
```

## 📝 输出示例

### evaluation_summary.json
```json
{
  "model": "opt-125m",
  "config": "dense",
  "dataset": "wikitext2",
  "ppl_results": [21.89, 21.91, 21.88],
  "average_ppl": 21.89,
  "min_ppl": 21.88,
  "max_ppl": 21.91,
  "total_inferences": 10
}
```

### metadata.json (每次推理)
```json
{
  "model": "opt-125m",
  "config": "dense",
  "dataset": "wikitext2",
  "ppl": 21.89,
  "inference_number": 1,
  "total_gemm_operations": 1920
}
```

### gemm_stats.json (GEMM 操作统计)
```json
{
  "linear_0_model.decoder.layers.0.self_attn.q_proj": {
    "input_shape": [1, 2048, 768],
    "weight_shape": [768, 768],
    "input_sparsity": 0.15,
    "weight_sparsity": 0.0
  }
}
```

## 🔧 故障排除

| 问题 | 解决方案 |
|------|--------|
| 内存不足 | 减少 `--num-inferences` |
| 数据集加载失败 | 检查 `--data-root` 路径 |
| CUDA 错误 | 运行 `nvidia-smi` 检查 GPU |
| 速度太慢 | 使用更好的 GPU 或减少数据集 |

## 📚 文件清单

```
sparsegpt/
├── eval_with_gemm_hook.py      # 主评估脚本
├── gemm_hook.py                # GEMM Hook 管理器
├── analyze_gemm_results.py     # 结果分析脚本
├── run_gemm_eval.sh            # 启动脚本
├── init_gemm_output.sh         # 目录初始化脚本
├── GEMM_HOOK_README.md         # 详细文档
├── GEMM_HOOK_QUICK_REF.md      # 快速参考（本文件）
└── output/                     # 评估结果输出目录
    ├── opt-125m_dense_wikitext2/
    ├── opt-125m_dense_ptb/
    ├── opt-125m_dense_c4/
    ├── opt-125m_gmp_wikitext2/
    └── ... (15 more combinations)
```

## 💡 提示

1. **首次运行前** - 初始化输出目录:
   ```bash
   bash init_gemm_output.sh
   ```

2. **监控进度** - 在另一个终端查看输出:
   ```bash
   tail -f eval.log
   ```

3. **并行运行** - 可以同时运行多个配置:
   ```bash
   python eval_with_gemm_hook.py model1 &
   python eval_with_gemm_hook.py model2 &
   ```

4. **定时备份** - 定期备份 `output/` 目录:
   ```bash
   tar -czf output_backup_$(date +%Y%m%d).tar.gz output/
   ```

## 🎓 关键概念

- **GEMM** - 矩阵乘法操作，是深度学习的核心计算
- **Hook** - PyTorch 中的钩子机制，用于在前向传播中拦截数据
- **稀疏性** - 权重矩阵中零值的比例
- **PPL** - 困惑度，语言模型评估指标

## 📞 支持

如有问题，请查看:
- 详细文档: `GEMM_HOOK_README.md`
- 代码注释: `eval_with_gemm_hook.py`
- 报错信息: 检查日志文件

---

**快速开始**: `python eval_with_gemm_hook.py /home/zhaojun/proj26/models/opt-125m --data-root /home/zhaojun/proj26/datasets`
