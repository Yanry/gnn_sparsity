# LightGCN
```
$ cd LightGCN
$ python lightgcn_main.py --dataset amazon-book --adj_type pre --Ks [20]
```
output: LightGCN/output

# Graph_Transformer_Networks
Dataset: Download datasets (DBLP, ACM, IMDB) from this [link](https://drive.google.com/file/d/1Nx74tgz_-BDlqaFO75eQG6IkndzI92j4/view?usp=sharing) and extract data.zip into data folder.
``` 
$ mkdir data
$ cd data
$ unzip data.zip
$ cd ..
```
```
$ cd Graph_Transformer_Networks
$ python fastgtn_main.py --dataset ACM/DBLP/IMDB --non_local
```
**Notice:** --non_local must be True

output: Graph_Transformer_Networks/output

# HGSL
## 使用方法

```bash

# 评估特定数据集
python eval_with_hook.py --dataset acm
python eval_with_hook.py --dataset yelp
python eval_with_hook.py --dataset dblp

# 指定GPU和输出目录
python eval_with_hook.py --dataset acm --gpu_id 0 --output_dir output/acm
python eval_with_hook.py --dataset yelp --gpu_id 0 --output_dir output/yelp
python eval_with_hook.py --dataset dblp --gpu_id 0 --output_dir output/dblp

## 输出文件说明

脚本将在 `HGSL/output` 目录生成以下文件：

### 1. SPMM数据文件
- `spmm_call_####_sparse.pt` - torch.spmm第一个输入（稀疏/密集矩阵）
- `spmm_call_####_dense.pt` - torch.spmm第二个输入（密集矩阵）  
- `spmm_call_####_output.pt` - torch.spmm的输出

### 2. 元数据和摘要
- `spmm_metadata.json` - 所有SPMM调用的详细元数据（JSON格式）
- `spmm_summary.txt` - SPMM调用的可读摘要

### 3. 评估结果
- `{dataset}_eval_results.json` - 模型评估的F1得分等指标
- `{dataset}_logits.pt` - 模型输出的logits
- `{dataset}_adj_new.pt` - 模型学习到的新邻接矩阵

### 捕获的信息
```json
{
  "call_id": 0,
  "sparse_shape": [7305, 334],
  "sparse_is_sparse": false,
  "dense_shape": [334, 64],
  "sparse_path": "路径/spmm_call_0000_sparse.pt",
  "dense_path": "路径/spmm_call_0000_dense.pt",
  "output_path": "路径/spmm_call_0000_output.pt",
  "output_shape": [7305, 64],
  "sparse_nnz": "unknown"  # 仅对真正的稀疏矩阵有效
}
```

## 加载保存的数据

```python
import torch

# 加载SPMM的输入和输出
sparse_mat = torch.load('spmm_call_0000_sparse.pt')
dense_mat = torch.load('spmm_call_0000_dense.pt')
output = torch.load('spmm_call_0000_output.pt')

# 加载模型输出
logits = torch.load('dblp_logits.pt')
adj_new = torch.load('dblp_adj_new.pt')

# 加载评估结果
import json
with open('dblp_eval_results.json') as f:
    results = json.load(f)
    print(f"Test F1: {results['test_f1']}")
```

## 示例日志输出

```
📊 Loading dataset: dblp
🏗️  Creating model...
📦 Loading checkpoint: /path/to/model.pt
🎣 Setting up SPMM hook...
🔄 Running forward pass...

📈 Evaluating results...

✅ Evaluation Results:
   Test F1: 0.9201
   Test Micro-F1: 0.9285
   Val F1: 0.9239
   Val Micro-F1: 0.9233
   Total SPMM calls: 4
✓ Saved SPMM metadata
✓ Saved SPMM summary
✓ Saved logits and adjacency matrix
✓ All outputs saved to: HGSL/output
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | dblp | 数据集名称 (acm, dblp, yelp) |
| `--gpu_id` | 0 | GPU设备ID，-1表示使用CPU |
| `--model_path` | None | 模型检查点路径，不指定则自动查找最新 |
| `--output_dir` | HGSL/output | 输出目录 |