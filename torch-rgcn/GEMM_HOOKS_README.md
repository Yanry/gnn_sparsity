# GEMM/SpGEMM Hook 功能说明

## 功能概述

已为 `torch-rgcn` 项目添加了 GEMM/SpGEMM 算子监控和模型保存功能：

1. **模型保存**：训练结束后自动保存模型到 `ckpt/<model_name>/model_final.pt`
2. **GEMM 算子监控**：在评估（eval）阶段记录所有 GEMM/SpGEMM 操作
3. **输入输出记录**：保存每次算子调用的输入 tensor、输出 tensor 及元数据

## 修改的文件

### 新增文件
- `utils/gemm_hooks.py`: GEMM 算子 hook 工具模块

### 修改的文件
- `experiments/classify_nodes.py`: 添加模型保存和 GEMM 监控（仅在 eval 时）
- `experiments/predict_links.py`: 添加模型保存和 GEMM 监控（仅在 final eval 时）
- `torch_rgcn/utils.py`: 修复 deprecated API 警告（torch.sparse.FloatTensor -> torch.sparse_coo_tensor）

## 监控的算子类型

以下 PyTorch 算子会被监控：
- `torch.mm` - 矩阵乘法
- `torch.matmul` - 通用矩阵乘法
- `torch.einsum` - Einstein summation
- `torch.spmm` - 稀疏矩阵乘法（如果可用）
- `torch.sparse.mm` - 稀疏矩阵乘法
- `torch.addmm` - 带偏置的矩阵乘法
- `torch.bmm` - 批量矩阵乘法

特别关注 `torch_rgcn/layers.py` 中的以下操作：
- `torch.einsum('rb, bio -> rio', ...)` - basis 分解中的组合
- `torch.einsum('rio, rni -> no', ...)` - 垂直堆叠的消息传递
- `torch.einsum('ni, rio -> rno', ...)` - 水平堆叠的特征-权重乘法
- `torch.einsum('nbi, rbio -> rnbo', ...)` - block 分解中的分块操作
- `torch.mm(adj, ...)` - 邻接矩阵与特征的乘法
- `torch.spmm(adj, features)` - 稀疏矩阵与特征的乘法

## 目录结构

```
torch-rgcn/
├── ckpt/                    # 模型检查点目录
│   └── <model_name>/       # 每个模型的子目录
│       └── model_final.pt  # 最终模型权重
└── output/                  # GEMM 记录输出目录
    └── <model_name>/       # 每个模型的子目录
        ├── gemm_metadata.json         # 元数据总览
        ├── <op_type>_call_0000_input1.pt   # 输入 tensor 1
        ├── <op_type>_call_0000_input2.pt   # 输入 tensor 2
        └── <op_type>_call_0000_output.pt   # 输出 tensor
```

## 模型命名规则

模型名称自动生成，格式为：
- 节点分类: `{dataset}_{model_type}_h{hidden}_l{layers}_{decomp_type}`
- 链接预测: `{dataset}_{encoder}_{decoder}_h{hidden}_l{layers}_{decomp_type}`

示例：
- `aifb_rgcn_h16_l2`
- `WN18_rgcn_distmult_h500_l2_basis`

## 元数据格式

`gemm_metadata.json` 包含：
```json
{
  "records": [
    {
      "op_type": "torch.einsum",
      "call_id": 0,
      "input1_shape": [4, 100, 500],
      "input2_shape": [4, 272, 100],
      "output_shape": [272, 500],
      "input1_dtype": "torch.float32",
      "input2_dtype": "torch.float32",
      "input1_device": "cuda:0",
      "input2_device": "cuda:0",
      "input1_sparse": false,
      "input2_sparse": false,
      "equation": "rio, rni -> no"
    }
  ],
  "summary": {
    "total_operations": 150,
    "operations_by_type": {
      "torch.mm": 50,
      "torch.einsum": 80,
      "torch.spmm": 20
    }
  }
}
```

## 使用方法

### 运行节点分类实验
```bash
cd /home/nizhj/gnn_sparsity/torch-rgcn
python experiments/classify_nodes.py with configs/rgcn/nc-aifb.yaml
```

### 运行链接预测实验
```bash
cd /home/nizhj/gnn_sparsity/torch-rgcn
python experiments/predict_links.py with configs/rgcn/lp-WN18.yaml
```

### 加载保存的模型
```python
import torch
from torch_rgcn.models import NodeClassifier, LinkPredictor

# 加载检查点
checkpoint = torch.load('ckpt/<model_name>/model_final.pt')
model_config = checkpoint['model_config']
state_dict = checkpoint['model_state_dict']

# 重建模型
model = NodeClassifier(...)  # 使用 model_config 中的参数
model.load_state_dict(state_dict)
model.eval()
```

### 读取 GEMM 记录
```python
import torch
import json

# 读取元数据
with open('output/<model_name>/gemm_metadata.json', 'r') as f:
    metadata = json.load(f)

# 加载特定算子的输入输出
input1 = torch.load('output/<model_name>/torch.mm_call_0000_input1.pt')
input2 = torch.load('output/<model_name>/torch.mm_call_0000_input2.pt')
output = torch.load('output/<model_name>/torch.mm_call_0000_output.pt')

# 对于稀疏张量
sparse_data = torch.load('output/<model_name>/torch.spmm_call_0000_input1.pt')
sparse_tensor = torch.sparse_coo_tensor(
    sparse_data['indices'],
    sparse_data['values'],
    sparse_data['shape']
)
```

## 关键修复

1. **递归错误修复**：hook wrapper 现在调用原始函数而不是被替换的函数
2. **仅 eval 时监控**：训练期间禁用 hook，仅在评估时启用，避免性能开销和大量数据
3. **Deprecated API 修复**：将 `torch.cuda.sparse.FloatTensor` 替换为 `torch.sparse_coo_tensor`

## 性能说明

- **训练阶段**：hook 被禁用，对训练速度无影响
- **评估阶段**：hook 启用，会记录所有 GEMM 操作，可能略微降低评估速度
- **存储空间**：取决于模型大小和评估数据量，每次算子调用保存 3 个 tensor 文件

## 注意事项

1. 确保有足够的磁盘空间存储 tensor 数据
2. 大型模型可能产生大量文件，建议评估后及时分析和清理
3. 稀疏 tensor 以 COO 格式保存（indices, values, shape）
4. 所有 tensor 在保存前会移到 CPU 以节省 GPU 内存
