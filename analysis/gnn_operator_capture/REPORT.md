# GNN Operator Capture Report

## Coverage
- Models: gat, gcn, gin, graphsage
- Datasets: citeseer, cora, ogbn-arxiv, pubmed
- Operator types: matmul, spmm
- Captured invocations: 96

## Typical Tensor Shapes
- `matmul` input A shapes: 169343x128, 169343x40, 169343x64, 19717x3, 19717x500, 19717x64, 2708x1433, 2708x64
- `matmul` input B shapes: 3, 3x64, 40, 40x64, 6, 64, 64x128, 64x1433
- `spmm` input A shapes: 169343x169343, 19717x19717, 2708x2708, 3327x3327
- `spmm` input B shapes: 169343x128, 169343x40, 169343x64, 19717x3, 19717x500, 19717x64, 2708x1433, 2708x64

## Sparsity Observations
- `matmul` average input sparsity: A=0.3711, B=0.0000
- `spmm` average input sparsity: A=0.9993, B=0.2440

## Representative High-Sparsity Captures
- `gat` on `ogbn-arxiv` at `layer0_spmm` (spmm) has input A sparsity 1.0000 with shape 169343x169343
- `gat` on `ogbn-arxiv` at `layer1_spmm` (spmm) has input A sparsity 1.0000 with shape 169343x169343
- `gin` on `ogbn-arxiv` at `layer0_spmm` (spmm) has input A sparsity 1.0000 with shape 169343x169343
- `gin` on `ogbn-arxiv` at `layer1_spmm` (spmm) has input A sparsity 1.0000 with shape 169343x169343

## Per Model-Dataset Notes
- `gat` on `citeseer`: 6 matmul captures, 2 spmm captures, mean operand sparsity A=0.3803, B=0.0024
- `gat` on `cora`: 6 matmul captures, 2 spmm captures, mean operand sparsity A=0.3731, B=0.0000
- `gat` on `ogbn-arxiv`: 6 matmul captures, 2 spmm captures, mean operand sparsity A=0.2886, B=0.0129
- `gat` on `pubmed`: 6 matmul captures, 2 spmm captures, mean operand sparsity A=0.3624, B=0.0000
- `gcn` on `citeseer`: 2 matmul captures, 2 spmm captures, mean operand sparsity A=0.8678, B=0.0011
- `gcn` on `cora`: 2 matmul captures, 2 spmm captures, mean operand sparsity A=0.8732, B=0.0000
- `gcn` on `ogbn-arxiv`: 2 matmul captures, 2 spmm captures, mean operand sparsity A=0.6237, B=0.0000
- `gcn` on `pubmed`: 2 matmul captures, 2 spmm captures, mean operand sparsity A=0.8517, B=0.0000
- `gin` on `citeseer`: 4 matmul captures, 2 spmm captures, mean operand sparsity A=0.6561, B=0.1652
- `gin` on `cora`: 4 matmul captures, 2 spmm captures, mean operand sparsity A=0.6780, B=0.1646
- `gin` on `ogbn-arxiv`: 4 matmul captures, 2 spmm captures, mean operand sparsity A=0.4963, B=0.0000
- `gin` on `pubmed`: 4 matmul captures, 2 spmm captures, mean operand sparsity A=0.6332, B=0.1500
- `graphsage` on `citeseer`: 4 matmul captures, 2 spmm captures, mean operand sparsity A=0.8070, B=0.2517
- `graphsage` on `cora`: 4 matmul captures, 2 spmm captures, mean operand sparsity A=0.7789, B=0.2482
- `graphsage` on `ogbn-arxiv`: 4 matmul captures, 2 spmm captures, mean operand sparsity A=0.4818, B=0.0721
- `graphsage` on `pubmed`: 4 matmul captures, 2 spmm captures, mean operand sparsity A=0.7397, B=0.2282

## Limitations and Notes
- Edge-wise attention coefficients are materialized as a sparse matrix before multiplication.
