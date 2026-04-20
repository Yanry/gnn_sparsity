# GNN Operator Capture

This project captures the two logical input operands of dense matrix multiplication and sparse-dense matrix multiplication used inside representative GNN models, then computes per-tensor sparsity statistics and saves both raw tensors and summaries.

## Current scope

- Models: `gcn`, `graphsage`, `gin`, `gat`
- Datasets: `cora`, `citeseer`, `pubmed`, `ogbn-arxiv`
- Execution mode: inference-first, full-batch
- Dense operand capture: `torch.nn.functional.linear`, `torch.mm`, `torch.matmul`
- Sparse operand capture: `torch.sparse.mm`

The implementation deliberately instruments the actual linear algebra call sites instead of only recording module-level inputs and outputs.

## Layout

```text
analysis/gnn_operator_capture/
  data/
  scripts/
    run_capture.py
  outputs/
    <model_name>/
      <dataset_name>/
        tensors/
        stats/
        plots/
  src/gnn_operator_capture/
```

## Usage

Activate the requested environment first:

```bash
conda activate vit
python analysis/gnn_operator_capture/scripts/run_capture.py --models gcn --datasets cora
```

Run multiple model-dataset pairs:

```bash
python analysis/gnn_operator_capture/scripts/run_capture.py \
  --models gcn graphsage gin gat \
  --datasets cora citeseer pubmed ogbn-arxiv
```

Skip plots when you only want raw tensors and statistics:

```bash
python analysis/gnn_operator_capture/scripts/run_capture.py --models gcn --datasets cora --skip-plots
```

## What gets saved

For each operator invocation:

- raw operand tensors as `.pt`
- per-invocation JSON statistics
- aggregated CSV and JSON summaries
- optional histograms and small-matrix heatmaps

The summary includes:

- tensor identity and hook source
- model, dataset, layer, operator type
- shape, dtype, device, nnz, zero ratio, sparsity ratio
- min, max, mean, std
- sparse layout metadata when available
- block sparsity for `1x1`, `4x4`, `8x8`, `16x16`, `32x32`
- thresholded near-zero sparsity for floating-point dense tensors

## Design notes

- `GCN`, `GraphSAGE`, and `GIN` are implemented with explicit sparse adjacency matrices and `torch.sparse.mm` so that the propagation operand pair is directly available.
- `GAT` materializes the edge-wise attention coefficients into a sparse attention matrix before the propagation multiply. This preserves the intended semantics of the SpMM input pair, but the attention matrix is reconstructed from edge coefficients rather than intercepted inside a fused vendor kernel.
- The dense side uses low-level patches around `torch.nn.functional.linear`, `torch.mm`, and `torch.matmul`, with layer-scoped context to keep the metadata semantically meaningful.

## Known limitations

- This first version is full-batch only. It is enough for the requested citation benchmarks and `ogbn-arxiv`, but Reddit-style neighbor sampling is not yet implemented.
- Captured weights reflect the initialized model parameters unless you add training before the forward pass.
- If a future model relies on fused CUDA kernels that bypass the patched PyTorch entry points, the exact internal operands may not be observable without deeper backend-specific instrumentation.
