# Longformer Attention Extraction

This directory contains a HuggingFace-based pipeline for extracting Longformer attention probabilities after softmax during the prefill forward pass.

## Files

- `extract_longformer_attention.py`: end-to-end extraction, quantize/dequantize, sparsity analysis, and visualization.
- `run.py`: original reference script kept unchanged.
- `output/`: saved `.pt` tensors, `.png` heatmaps, and aggregate statistics.

## How the attention is hooked

The extractor registers a forward hook on every `layer.attention.self` module inside HuggingFace `LongformerSelfAttention`.

The hook captures:

- `attn_probs`: the real tensor produced by `softmax(attn_scores, dim=-1)` inside `LongformerSelfAttention.forward`
- `global_attn_probs`: the real tensor produced by the global-attention softmax path when global tokens are present

The pipeline does not save:

- pre-softmax logits
- masks
- attention outputs after multiplying by `V`

## Tensor format

Longformer computes sparse sliding-window attention internally. The hook therefore sees the exact sparse softmax weights used by the model.

For storage, the pipeline reconstructs an exact dense per-layer attention tensor of shape:

- `[num_heads, seq_len, seq_len]`

This reconstruction is exact for the processed sequence length:

- local sliding-window weights are written into their absolute token positions
- attention to global keys is written into the corresponding columns
- rows for global-query tokens are overwritten with `global_attn_probs`, which are the actual probabilities used for those queries

The saved tensor is then quantized and dequantized as required:

1. `q = round(attn * 255)` stored temporarily as `uint8`
2. `attn_fp16 = q / 255` stored as `float16`

Only the dequantized `float16` tensor is written to disk.

## Outputs

Each layer tensor is saved as:

- `output/model_<modelname>_sample_<id>_layer_<layerid>.pt`

Each heatmap is saved with the same prefix and `.png`.

Aggregate sparsity statistics are written to:

- `output/sparsity_stats.csv`
- `output/sparsity_stats.json`

Sample metadata is written to:

- `output/sample_manifest.json`

## Notes and limits

- The pipeline uses the provided local checkpoints, remapped from the original `roberta.*` key layout into current HuggingFace `longformer.*` names.
- The visualization is a head-averaged heatmap for each saved 3D tensor. The `.pt` file still contains all heads.
- Dense `[heads, seq_len, seq_len]` tensors become very large for multi-thousand-token inputs, so the default sample set uses long high-hundreds to low-thousands token lengths to keep the full export practical.

## Example

```bash
conda run -n vit python /home/nizhj/gnn_sparsity/longformer/extract_longformer_attention.py \
  --models longformer-base-4096 \
  --samples-per-model 1 \
  --overwrite
```
