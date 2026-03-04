# OPT-125m Attention 操作详细分析报告

## 问题总结

你提出了三个关键问题：
1. **Tensor 保存命名** - 如何按照 block 和投影类型来保存 tensor
2. **QK^T 的 Shape** - QK^T 操作的具体形状是什么
3. **缺失的 Matmul** - 为什么 softmax(Q@K^T) @ V 的 matmul 没有被捕获

---

## 问题 1: Tensor 保存命名 ✅

### 当前问题
目前的 hook 只是笼统地保存 "attn_output_X"，没有指明是哪个 block 的哪个投影层。

### 解决方案
**改进的 hook** (gemm_hook_improved.py) 采用更清晰的命名格式：

```
block_0_q_proj_output      # Layer 0 的 Q 投影输出
block_0_k_proj_output      # Layer 0 的 K 投影输出
block_0_v_proj_output      # Layer 0 的 V 投影输出
block_0_out_proj_output    # Layer 0 的输出投影
```

**优势：**
- ✅ 明确指出是哪个 block
- ✅ 明确指出是 q/k/v/out_proj 的哪一个
- ✅ 便于后续追踪 attention 的完整数据流

### 实现代码
```python
def _hook_projection(self, linear_module, proj_name):
    """为投影层（q/k/v/out_proj）添加 hook"""
    def proj_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            out_cpu = output.detach().cpu().float()
            self.gemm_data[f"{proj_name}_output"] = {
                'operation': 'projection_output',
                'projection': proj_name,
                'output_shape': tuple(out_cpu.shape),
                'output_sparsity': float(self._compute_sparsity(out_cpu)),
            }
            # 直接以 proj_name 保存，如 "block_0_q_proj_output.pt"
            self.tensor_cache[f"{proj_name}_output"] = out_cpu
    
    hook = linear_module.register_forward_hook(proj_hook)
    self.hooks.append(hook)
```

---

## 问题 2: QK^T 的 Shape ✅

### 基础配置（OPT-125m）
| 配置项 | 值 |
|------|---|
| 隐层维度 | 768 |
| 注意力头数 | 12 |
| 每个头的维度 | 64 (768÷12) |
| 序列长度 | 2048 |
| 批处理大小 | 1 |

### Q 和 K 的形状

**Q 的形状：**
```
原始形状: [batch=1, seq_len=2048, hidden_dim=768]
        = [1, 2048, 768]

多头拆分: [batch=1, num_heads=12, seq_len=2048, head_dim=64]
        = [1, 12, 2048, 64]
```

**K 的形状：**
```
原始形状: [batch=1, seq_len=2048, hidden_dim=768]
        = [1, 2048, 768]

多头拆分: [batch=1, num_heads=12, seq_len=2048, head_dim=64]
        = [1, 12, 2048, 64]
```

### **QK^T 的形状（关键结果）**

```
操作: Q @ K^T
    [1, 2048, 768] @ [768, 2048]

多头情况下（实际计算）:
    [1, 12, 2048, 64] @ [1, 12, 64, 2048]  (对每个头分别计算)
    ↓
    输出: [1, 12, 2048, 2048]
    
或全局形式:
    [1, 2048, 2048]  (如果不显式拆分多头)

元素数量:
    单个 head 的 QK^T: 1 × 2048 × 2048 = 4,194,304 个元素
    全部 12 heads 的总数: 4,194,304 × 12 = 50,331,648 个元素
```

### 数据流验证（从你的 gemm_stats.json）

```json
{
  "linear_2_model.decoder.layers.0.self_attn.q_proj": {
    "input_shape": [1, 2048, 768],        ← Q 的线性层输入
    "weight_shape": [768, 768],           ← Q 投影权重
    "output": [1, 2048, 768]              ← Q 投影输出
  },
  
  "linear_0_model.decoder.layers.0.self_attn.k_proj": {
    "input_shape": [1, 2048, 768],        ← K 的线性层输入
    "weight_shape": [768, 768],           ← K 投影权重
    "output": [1, 2048, 768]              ← K 投影输出
  }
}

⇒ 因此 QK^T 会是: [1, 2048, 768] × [768, 2048] = [1, 2048, 2048]
```

### softmax(QK^T) @ V 的形状

```
操作1: softmax(QK^T)
    输入: [1, 12, 2048, 2048] (attention scores)
    输出: [1, 12, 2048, 2048] (softmax 只改变值，不改变形状)

操作2: softmax(QK^T) @ V
    attention_weights: [1, 12, 2048, 2048]
    V 投影输出:        [1, 12, 2048, 64]
    ↓
    输出:              [1, 12, 2048, 64]
    
最终拼接回:           [1, 2048, 768]
```

---

## 问题 3: 为什么缺少 Matmul 操作？ ⚠️

### 诊断结果

**当前状态：❌ matmul 操作没有被捕获**

从分析输出可以看到：
```
⚠️ 没找到 matmul 操作记录
   这说明当前的 hook 可能没有正确捕获 matmul 操作
```

### 根本原因

#### 原因 1: OPT 使用了优化的 Attention 实现
OPT 模型可能使用了 PyTorch 的优化实现，如 `torch.nn.functional.scaled_dot_product_attention()`，而不是显式的矩阵乘法。

```python
# OPT 内部可能是这样实现的
attn_output = F.scaled_dot_product_attention(
    Q, K, V, attn_mask=attention_mask
)

# 而不是显式的
scores = Q @ K.transpose(-2, -1) / sqrt(d_k)
scores = F.softmax(scores, dim=-1)
output = scores @ V
```

#### 原因 2: Matmul Hook 的替换可能失效
全局替换 `torch.matmul` 和 `torch.bmm` 的方法在以下情况下会失败：
- C++ 编译的 kernel (如 CUDA kernel) 不走 Python 层
- 函数已经被编译或优化
- 某些情况下 hook 没有正确传播

### 解决方案

#### 方案 A: 修改 OPT 源代码（最有效）❌ 不推荐，破坏性大

#### 方案 B: 使用改进的 Hook（推荐）✅

创建一个**定制化的 Attention Wrapper**：

```python
class AttentionHookWrapper(nn.Module):
    """包装 OPT 的 attention 层以捕获中间结果"""
    
    def __init__(self, original_attn, block_id):
        super().__init__()
        self.attn = original_attn
        self.block_id = block_id
        self.intermediate_cache = {}
    
    def forward(self, hidden_states, attention_mask=None, ...):
        # 获取 Q, K, V
        Q = self.attn.q_proj(hidden_states)
        K = self.attn.k_proj(hidden_states)
        V = self.attn.v_proj(hidden_states)
        
        # 保存到 cache（这样就可以后续分析 QK^T）
        self.intermediate_cache[f'block_{self.block_id}_Q'] = Q
        self.intermediate_cache[f'block_{self.block_id}_K'] = K
        self.intermediate_cache[f'block_{self.block_id}_V'] = V
        
        # 调用原始 attention
        # 可以通过 hook 这里选择性地记录中间状态
        ...
```

#### 方案 C: 使用 torch.jit.trace 追踪

```python
# 追踪完整的 forward 过程
traced_attention = torch.jit.trace(attn_module, example_inputs)
# 这会产生一个 IR 描述，可以分析所有在其中发生的操作
```

#### 方案 D: 直接修改 eval 脚本（最实用）✅

在 `eval_with_gemm_hook.py` 中，在 forward 之前：

```python
# 手动提取 Q, K, V 并记录
with torch.no_grad():
    # 获取每层的 hidden states
    for layer_idx, layer in enumerate(model.model.decoder.layers):
        hidden = ...  # 来自上一层的输出
        
        # 手动调用 attention 的投影
        q = layer.self_attn.q_proj(hidden)
        k = layer.self_attn.k_proj(hidden)
        v = layer.self_attn.v_proj(hidden)
        
        # 手动计算 QK^T
        # reshape 为多头形式
        batch_size = q.shape[0]
        seq_len = q.shape[1]
        num_heads = 12
        head_dim = 64
        
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        # 计算 QK^T
        qkt = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        # 现在可以记录 qkt 的形状、稀疏性等
        
        # 继续正常的 attention
        attn_output = layer(...)
```

---

## 建议行动方案

### 短期（立即可做）
1. ✅ 使用 `gemm_hook_improved.py` 替换当前的 hook
2. ✅ 运行修改后的 eval 脚本，获取更细致的投影层数据
3. ✅ 使用分析脚本 `analyze_attention.py` 生成报告

### 中期（需要代码修改）
4. 实现**方案 D**：在 eval 脚本中手动提取 Q/K/V 并保存
5. 计算 QK^T 矩阵并分析其稀疏性

### 长期（研究方向）
6. 研究 OPT/LLaMA 等大模型中 Attention scores 的稀疏性规律
7. 基于 attention 稀疏性的剪枝策略
8. Token 级别的动态修剪

---

## 快速参考表

| 項目 | 值 |
|------|---|
| Q 投影输出 | `[1, 2048, 768]` |
| K 投影输出 | `[1, 2048, 768]` |
| V 投影输出 | `[1, 2048, 768]` |
| **QK^T 的形状** | **`[1, 12, 2048, 2048]`** (多头) |
| **QK^T 的元素数** | **50,331,648** (全部头) |
| Attention Output | `[1, 2048, 768]` |
| 层数 | 12 |

---

## 文件参考

- 📄 改进的 hook: `/home/nizhj/gnn_sparsity/sparsegpt/gemm_hook_improved.py`
- 📊 分析脚本: `/home/nizhj/gnn_sparsity/sparsegpt/analyze_attention.py`
- 📈 你的数据: `/home/nizhj/gnn_sparsity/sparsegpt/output/opt-125m_gmp_ptb/inference_001/gemm_stats.json`
