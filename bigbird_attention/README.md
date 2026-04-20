# BigBird Attention Extraction

This pipeline extracts BigBird attention weights after softmax from a local model at:

- `/home/zhaojun/proj26/models/bigbird`

It uses local WikiText data from:

- `/home/zhaojun/proj26/datasets/wikitext`

It saves outputs under:

- `/home/nizhj/gnn_sparsity/bigbird_output`

## What is captured

### `block_sparse`

The pipeline hooks the real `nn.functional.softmax(...)` outputs inside `BigBirdBlockSparseAttention.bigbird_block_sparse_attention`.

These are the real post-softmax tensors used in the attention-times-value computation:

- `first_attn_weights`
- `second_attn_weights`
- `middle_attn_weights`
- `second_last_attn_weights`
- `last_attn_weights`

Those tensors are the true internal sparse/block-structured attention probabilities. They are not square `[seq_len, seq_len]` maps.

HuggingFace can optionally return a dense square attention tensor when `output_attentions=True`, but the source code explicitly notes that this branch is only for visualization and not used by the forward pass. For that reason, this pipeline saves the real sparse tensors and only reconstructs a square map for plotting and analysis.

### `original_full`

The pipeline captures the returned `attention_probs` tensor from `BigBirdSelfAttention.forward`. In eval mode, dropout is disabled, so that tensor is the normalized post-softmax attention map actually used by the layer.

## Quantization

Every saved attention tensor is quantized and dequantized as requested:

1. `q = round(attn * 255)`
2. `attn_fp16 = q / 255`

Only the dequantized `float16` tensors are written to disk.

## Tokenization and inputs

The local tokenizer loading path in this Transformers build is unreliable, so the script uses the local `spiece.model` directly through `sentencepiece`.

Samples are built from local WikiText-103 raw parquet rows. The script concatenates natural-text rows until the target token length is reached, then truncates to the requested length.

The default target lengths are multiples of 64 and all exceed the threshold required for BigBird to stay in true `block_sparse` mode.

## Files

- `extract_bigbird_attention.py`: main runner plus sample prep, hooks, stats, and plotting.
- `bigbird_output/tensors/`: saved `.pt` files.
- `bigbird_output/plots/`: PNG heatmaps.
- `bigbird_output/stats/`: CSV and JSON statistics.
- `bigbird_output/logs/`: run config, sample metadata, and verification logs.

## Saved tensor format

### `block_sparse`

Each layer file:

- `bigbird_block_sparse_sample_<sampleid>_layer_<layerid>.pt`

contains a dictionary with:

- metadata
- `rand_attn` block indices
- quantized/dequantized `float16` sparse attention tensors under `tensors`

### `original_full`

Each layer file:

- `bigbird_original_full_sample_<sampleid>_layer_<layerid>.pt`

contains a dictionary with:

- metadata
- one quantized/dequantized `float16` dense tensor of shape `[num_heads, seq_len, seq_len]`

## Stats and plots

The pipeline computes:

- total elements
- zeros
- sparsity ratio
- min / max / mean / std
- threshold sparsity for `1e-4`, `1e-3`, `1e-2`
- block sparsity on a 2D matrix view for block sizes `4`, `8`, `16`, `32`

For `block_sparse`, plots include:

- one plot per real sparse tensor
- one reconstructed dense square map per layer for inspection

For `original_full`, plots are the dense square attention maps.

## Verification

If you pass `--verify-capture` in `block_sparse` mode, the script runs one sample with `output_attentions=True` and compares the pipeline’s dense reconstruction against HuggingFace’s returned dense attention map. The result is written to:

- `bigbird_output/logs/verification.json`

## Example commands

Run the full default block-sparse pipeline:

```bash
conda run -n vit python /home/nizhj/gnn_sparsity/bigbird_attention/extract_bigbird_attention.py --verify-capture
```

Run one minimal sample:

```bash
conda run -n vit python /home/nizhj/gnn_sparsity/bigbird_attention/extract_bigbird_attention.py --samples 1 --verify-capture
```

Run the optional full-attention baseline:

```bash
conda run -n vit python /home/nizhj/gnn_sparsity/bigbird_attention/extract_bigbird_attention.py --attention-type original_full --samples 1
```
