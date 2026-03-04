# OPT Attention 数据捕获 - 使用指南

## 快速开始

针对你提出的三个问题，我提供了三个配套的工具和改进：

### 📁 新文件清单

| 文件 | 用途 |
|------|------|
| `gemm_hook_improved.py` | 改进的 GEMM hook，支持更细致的投影层命名 |
| `attention_data_capture.py` | 直接捕获 Q/K/V、QK^T、attention_weights 等中间层数据 |
| `analyze_attention.py` | 分析和可视化 attention 数据 |
| `ATTENTION_ANALYSIS_REPORT.md` | 详细的技术分析报告 |

---

## 问题 1: Tensor 命名 ✅

### 问题
目前保存为 `attn_output_0, attn_output_1, ...`，无法知道是哪个 block 的哪个投影层

### 解决方案
使用 `gemm_hook_improved.py` 中的改进 hook，自动按以下格式命名：

```
block_0_q_proj_output      ← Layer 0 Q 投影的输出
block_0_k_proj_output      ← Layer 0 K 投影的输出
block_0_v_proj_output      ← Layer 0 V 投影的输出
block_0_out_proj_output    ← Layer 0 输出投影的输出
block_1_q_proj_output
...
block_11_q_proj_output
```

### 集成到你的代码
在 `eval_with_gemm_hook.py` 中修改：

```python
# 原来的
from gemm_hook import GEMMHookManager

# 改为
from gemm_hook_improved import ImprovedGEMMHookManager as GEMMHookManager

# 其余代码保持不变
hook_manager = GEMMHookManager(output_dir, model_name, config_name, dataset_name)
```

---

## 问题 2: QK^T 的 Shape ✅

### 问题
想知道 QK^T 的具体形状

### 答案

| 项目 | 值 |
|------|---|
| Q shape | `[1, 2048, 768]` |
| K shape | `[1, 2048, 768]` |
| **QK^T (多头形式)** | **`[1, 12, 2048, 2048]`** |
| **QK^T (全部元素数)** | **50,331,648** |
| attention_weights shape | `[1, 12, 2048, 2048]` |
| attention_output shape | `[1, 12, 2048, 64]` |

### 详见
→ `ATTENTION_ANALYSIS_REPORT.md` 的 "问题 2" 部分

---

## 问题 3: 缺失的 Matmul 操作 ✅

### 问题
为什么 softmax(QK^T) @ V 的 matmul 没有被捕获？

### 原因
OPT 使用了优化的 attention 实现（如 `torch.nn.functional.scaled_dot_product_attention`），导致 torch.matmul hook 失效。

### 解决方案
**推荐：使用 `attention_data_capture.py`**

这个工具直接在 attention 的前向传播中捕获：
- Q/K/V 投影的输出
- **QK^T matmul 的结果和统计信息**
- **softmax(QK^T) 的结果和统计信息**
- **softmax(QK^T) @ V matmul 的结果和统计信息**

### 集成到你的 eval 脚本

在 `eval_with_gemm_hook.py` 中：

```python
from attention_data_capture import AttentionDataCapture

def evaluate_with_gemm_hook(model, dataloader, testloader, hook_manager, 
                           dataset_name, inference_idx=0, log_wandb=False):
    """改进的评估函数，同时捕获 GEMM 和 Attention 数据"""
    
    model.eval()
    model = model.to(DEV)
    
    # 同时注册两个 hook
    hook_manager.register_hooks(model)
    
    # ★ 新增：Attention 数据捕获
    attn_capture = AttentionDataCapture()
    attn_capture.register_attention_capture_hooks(model)
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    # ... 评估代码 ...
    
    with torch.no_grad():
        for i in range(min(nsamples, 5)):
            batch = testenc[:, start_idx:end_idx].to(DEV)
            outputs = model(batch)
            # ... 计算 loss ...
    
    # 移除 hooks
    hook_manager.remove_hooks()
    attn_capture.remove_hooks()  # ★ 新增
    
    model.config.use_cache = use_cache
    
    # ★ 新增：保存 Attention 数据
    attn_data_file = os.path.join(hook_manager.current_inference_dir, 'attention_data.json')
    attn_capture.save_to_json(attn_data_file)
    
    return float(ppl.item())
```

---

## 数据输出示例

运行改进后的 eval，你会在 `inference_001/` 目录下获得：

```
inference_001/
├── gemm_stats.json              # 原有的 GEMM 统计
├── attention_data.json          # ★ 新增：Attention 中间层数据
├── metadata.json
└── tensors/
    ├── block_0_q_proj_output.pt    # ★ 更清晰的命名
    ├── block_0_k_proj_output.pt
    ├── block_0_v_proj_output.pt
    ├── block_0_out_proj_output.pt
    ├── block_1_q_proj_output.pt
    ...
```

### attention_data.json 的结构

```json
{
  "layer_0": {
    "Q_proj": {
      "shape": [1, 2048, 768],
      "sparsity": 0.000001,
      "numel": 1572864
    },
    "K_proj": {
      "shape": [1, 2048, 768],
      "sparsity": 0.000001,
      "numel": 1572864
    },
    "V_proj": {
      "shape": [1, 2048, 768],
      "sparsity": 0.000000,
      "numel": 1572864
    },
    "QK^T": {
      "shape": [1, 12, 2048, 2048],
      "expected_shape": [1, 12, 2048, 2048],
      "sparsity": 0.0,
      "numel": 50331648,
      "min_value": -1000000.0,
      "max_value": 2.5,
      "mean_value": 0.0
    },
    "attention_weights": {
      "shape": [1, 12, 2048, 2048],
      "sparsity": 0.0,
      "numel": 50331648,
      "min_value": 0.0,
      "max_value": 1.0,
      "mean_value": 0.000488
    },
    "attention_output": {
      "shape": [1, 12, 2048, 64],
      "expected_shape": [1, 12, 2048, 64],
      "sparsity": 0.000000,
      "numel": 1572864
    },
    "final_output": {
      "shape": [1, 2048, 768],
      "sparsity": 0.000001,
      "numel": 1572864
    }
  },
  "layer_1": { ... },
  ...
  "layer_11": { ... }
}
```

---

## 分析 Attention 数据

使用提供的分析脚本：

```bash
# 分析 gemm_stats.json
conda run -n torch210 python analyze_attention.py \
  output/opt-125m_gmp_ptb/inference_001/gemm_stats.json

# 或使用改进的绘图脚本（待创建）
python plot_attention_data.py \
  output/opt-125m_gmp_ptb/inference_001/attention_data.json
```

---

## 关键发现与所有你的问题的总结

### 问题 1: Tensor 命名
**解决：** ✅ 使用 `gemm_hook_improved.py`
- 改为 `block_{id}_{q|k|v|out}_proj_output` 格式
- 清晰指明所属 layer 和投影类型

### 问题 2: QK^T Shape
**解决：** ✅ 已计算并验证
```
Q: [1, 2048, 768]
K: [1, 2048, 768]
QK^T: [1, 12, 2048, 2048]  ← 多头形式（这才是实际计算形式）
```

### 问题 3: Softmax(QK^T) @ V 的 Matmul
**解决：** ✅ 使用 `attention_data_capture.py`
- 直接在 forward hook 中捕获 Q/K/V 和 matmul
- 绕过 torch.matmul hook 的问题
- 保存所有中间层数据到 `attention_data.json`

---

## 下一步建议

1. **立即尝试** (5分钟)
   ```bash
   conda run -n torch210 python analyze_attention.py output/opt-125m_gmp_ptb/inference_001/gemm_stats.json
   ```

2. **集成改进的 hook** (15分钟)
   - 复制 `attention_data_capture.py` 中的 AttentionDataCapture 类到你的 eval 脚本
   - 按上面的示例修改 `eval_with_gemm_hook.py`

3. **运行改进版本** (看你的数据大小)
   - 重新运行 eval，会输出 `attention_data.json`

4. **分析稀疏性** (可选)
   - 分析 attention_weights 中有多少接近 0 的值
   - 可用于后续的 token pruning 或 head pruning

---

## 常见问题

**Q: 为什么 QK^T 是 [1, 12, 2048, 2048] 而不是 [1, 2048, 2048]？**
A: 因为有 12 个注意力头，每个头独立计算 QK^T。这是标准的 Multi-Head Attention 实现。

**Q: Attention 数据的稀疏性有什么用？**
A: 可用于：
- Token pruning：如果某行都很小，可以剪枝那个 token
- Head pruning：如果某个 head 的权重分布相似，可以去掉重复的 head
- Attention 压缩：开发更高效的 attention 算法

**Q: 能否只捕获特定 layer 的数据？**
A: 可以，修改 `register_attention_capture_hooks` 中的 loop 条件即可。

---

## 文件位置

```
/home/nizhj/gnn_sparsity/sparsegpt/
├── gemm_hook_improved.py           ← 改进的 hook
├── attention_data_capture.py        ← Attention 数据捕获工具 ★
├── analyze_attention.py             ← 分析脚本
├── ATTENTION_ANALYSIS_REPORT.md    ← 详细报告
└── eval_with_gemm_hook.py           ← 你的 eval 脚本（需要修改）
```

---

**祝数据采集顺利！有问题欢迎继续提问。** 🚀
