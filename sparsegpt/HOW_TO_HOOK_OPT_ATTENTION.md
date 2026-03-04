# 如何 Hook OPT Attention 中的 Matmul 操作

## 问题解决方案

你的问题：**为什么 softmax(Q@K^T) @ V 的 matmul 没有被捕获？**

### 根本原因

OPT 模型默认使用 **SDPA (Scaled Dot Product Attention)** 优化实现，而不是显式的 matmul 操作。这是 PyTorch 2.0+ 的优化特性。

```python
# 默认配置
model = OPTForCausalLM.from_pretrained('facebook/opt-125m')
print(model.config._attn_implementation)  # 输出: 'sdpa'
```

### 解决方案

强制使用 **eager** 实现，然后 hook `eager_attention_forward` 函数。

---

## 实现步骤

### 1. 使用 OPTAttentionHook

已创建的文件：`opt_attention_hook.py`

这个 hook 能捕获：
- ✅ **Q @ K^T matmul** 及其输出
- ✅ **softmax(Q@K^T)** 的结果
- ✅ **softmax(Q@K^T) @ V matmul** 及其输出

### 2. 修改模型加载方式

在 `eval_with_gemm_hook.py` 或其他脚本中，强制使用 eager 实现：

```python
# 原来的
from transformers import OPTForCausalLM
model = OPTForCausalLM.from_pretrained('facebook/opt-125m')

# 改为
model = OPTForCausalLM.from_pretrained('facebook/opt-125m', 
                                        attn_implementation="eager")
```

或者在 `opt.py` 中的 `get_opt` 函数中修改：

```python
def get_opt(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    # ★ 添加 attn_implementation="eager"
    model = OPTForCausalLM.from_pretrained(model, torch_dtype='auto', 
                                          attn_implementation="eager")
    model.seqlen = model.config.max_position_embeddings
    return model
```

### 3. 集成到 eval 脚本

在 `eval_with_gemm_hook.py` 中添加：

```python
from opt_attention_hook import OPTAttentionHook

def evaluate_with_gemm_hook(model, dataloader, testloader, hook_manager, 
                           dataset_name, inference_idx=0, log_wandb=False):
    """改进的评估函数"""
    
    model.eval()
    model = model.to(DEV)
    
    # 注册 GEMM hooks
    hook_manager.register_hooks(model)
    
    # ★ 新增：注册 Attention hooks
    attn_hook = OPTAttentionHook(save_tensors=False)
    attn_hook.install_hook()
    
    # ... 评估代码 ...
    
    with torch.no_grad():
        for i in range(min(nsamples, 5)):
            batch = testenc[:, start_idx:end_idx].to(DEV)
            outputs = model(batch)
            # ... 计算 loss ...
    
    # 移除 hooks
    hook_manager.remove_hooks()
    attn_hook.remove_hook()  # ★ 新增
    
    # ★ 新增：保存 Attention 数据
    attn_data_file = os.path.join(hook_manager.current_inference_dir, 
                                   'attention_data.json')
    attn_hook.save_to_json(attn_data_file)
    
    return float(ppl.item())
```

---

## 捕获的数据格式

运行后会生成 `attention_data.json`，包含：

```json
{
  "layer_0_QK^T": {
    "operation": "Q @ K^T",
    "query_shape": [1, 12, 2048, 64],
    "key_shape": [1, 12, 2048, 64],
    "key_transposed_shape": [1, 12, 64, 2048],
    "output_shape": [1, 12, 2048, 2048],  ← QK^T 的形状!
    "scaling_factor": 1.0,
    "output_numel": 50331648,
    "output_sparsity": 0.0,
    "output_min": -20.59,
    "output_max": 13.02,
    "output_mean": -0.79,
    "output_std": 2.17
  },
  "layer_0_attention_weights": {
    "operation": "softmax(Q@K^T)",
    "output_shape": [1, 12, 2048, 2048],
    "output_sparsity": 0.0,
    "output_min": 0.0,
    "output_max": 1.0,
    "output_mean": 0.00048828125,  ← 平均注意力权重
    "top_1_percent_threshold": 0.956,  ← 99分位数
    "top_10_percent_threshold": 0.123
  },
  "layer_0_attention@V": {
    "operation": "softmax(Q@K^T) @ V",
    "attention_weights_shape": [1, 12, 2048, 2048],
    "value_shape": [1, 12, 2048, 64],
    "output_shape": [1, 12, 2048, 64],
    "output_sparsity": 0.0000012,
    "output_min": -0.35,
    "output_max": 0.58,
    "output_mean": -0.0003
  },
  ...  // layer_1, layer_2, ..., layer_11
}
```

---

## 完整示例

### 1. 测试 Hook 功能

```bash
cd /home/nizhj/gnn_sparsity/sparsegpt
conda run -n torch210 python test_opt_attention_hook.py
```

输出：
```
✓ 已安装 OPT Attention Hook
捕获的层数: 12
总操作数: 36
  捕获的 Q@K^T 操作: 12  ✓
  捕获的 softmax 操作: 12  ✓
  捕获的 attention@V 操作: 12  ✓
```

### 2. 集成到你的 eval 脚本

修改 `opt.py`:

```python
def get_opt(model):
    # ... 原有代码 ...
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(
        model, 
        torch_dtype='auto',
        attn_implementation="eager"  # ★ 添加这一行
    )
    model.seqlen = model.config.max_position_embeddings
    return model
```

然后正常运行你的 eval：

```bash
cd /home/nizhj/gnn_sparsity/sparsegpt
./run_gemm_eval.sh
```

现在会同时输出：
- `gemm_stats.json` - Linear 层的输入输出
- `attention_data.json` - QK^T 和 attention@V 的 matmul 数据 ★

---

## 关键问题回答

### Q1: QK^T 的 Shape 是什么？

**答案：`[batch=1, num_heads=12, seq_len=2048, seq_len=2048]`**

```json
{
  "layer_0_QK^T": {
    "output_shape": [1, 12, 2048, 2048],
    "output_numel": 50331648
  }
}
```

### Q2: 为什么之前的 torch.matmul hook 没有工作？

**答案：** OPT 默认使用 SDPA (Scaled Dot Product Attention)，它是 C++ 或 CUDA kernel 实现的，不走 Python 的 `torch.matmul`。

**解决方案：** 强制使用 `attn_implementation="eager"`，就会使用 Python 实现的 attention，里面有显式的 matmul 调用。

### Q3: 如何 Hook 这些 matmul？

**答案：** 用 monkey patch 替换 `eager_attention_forward` 函数：

```python
import transformers.models.opt.modeling_opt as opt_modeling

# 保存原始函数
original_forward = opt_modeling.eager_attention_forward

# 替换为包装版本
opt_modeling.eager_attention_forward = your_hooked_function
```

这样就能在 attention 计算的**内部**捕获 QK^T 和 attention@V 的 matmul。

---

## 性能影响

⚠️ **注意：** 使用 `eager` 实现会比 `sdpa` 慢，因为：
- SDPA 使用优化的 CUDA kernel
- Eager 是纯 Python + PyTorch 实现

但对于**数据收集和分析**，这是必要的权衡。

如果只需要最终结果，可以继续使用 SDPA。
如果需要分析中间层数据，必须用 eager + hook。

---

## 文件清单

| 文件 | 用途 |
|------|------|
| `opt_attention_hook.py` | ★ Attention matmul hook 实现 |
| `test_opt_attention_hook.py` | 测试脚本 |
| `test_opt_attention_data.json` | 测试输出示例 |
| `opt.py` | 需要修改：添加 `attn_implementation="eager"` |

---

## 下一步

1. **立即体验：** 运行 `test_opt_attention_hook.py` 
2. **集成到 eval：** 修改 `opt.py` 的 `get_opt` 函数
3. **运行完整 eval：** `./run_gemm_eval.sh`
4. **分析数据：** 查看生成的 `attention_data.json`

现在你可以完整地追踪 Attention 中的所有 matmul 操作了！🎉
