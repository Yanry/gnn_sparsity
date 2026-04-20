import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
import torch.nn.functional as F
from transformers import BigBirdConfig, BigBirdModel
from transformers.models.big_bird.modeling_big_bird import BigBirdBlockSparseAttention, BigBirdSelfAttention


MODEL_PATH = Path("/home/zhaojun/proj26/models/bigbird")
DATASET_PATH = Path("/home/zhaojun/proj26/datasets/wikitext")
OUTPUT_ROOT = Path("/home/nizhj/gnn_sparsity/bigbird_output")

TARGET_LENGTHS = [768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344]
SPARSE_PIECE_NAMES = [
    "first_attn_weights",
    "second_attn_weights",
    "middle_attn_weights",
    "second_last_attn_weights",
    "last_attn_weights",
]


@dataclass
class SampleRecord:
    sample_id: int
    text: str
    source_file: str
    source_rows: List[int]
    target_length: int
    tokenized_length: int


@dataclass
class LayerCapture:
    mode: str
    layer_id: int
    seq_len: int
    block_size: Optional[int]
    num_heads: int
    num_random_blocks: Optional[int]
    sparse_pieces: Dict[str, torch.Tensor]
    rand_attn: Optional[torch.Tensor]
    dense_attention: Optional[torch.Tensor]


class LocalSentencePieceTokenizer:
    def __init__(self, model_path: Path):
        self.processor = spm.SentencePieceProcessor(model_file=str(model_path / "spiece.model"))
        self.pad_token_id = int(self.processor.pad_id())
        self.eos_token_id = int(self.processor.eos_id())
        self.bos_token_id = 2
        self.unk_token_id = int(self.processor.unk_id())
        self.vocab_size = int(self.processor.vocab_size())

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        pieces = self.processor.encode(text, out_type=int)
        if add_special_tokens:
            return [self.bos_token_id] + pieces + [self.eos_token_id]
        return pieces

    def encode_to_tensors(self, text: str, max_length: int, device: torch.device) -> Dict[str, torch.Tensor]:
        ids = self.encode(text, add_special_tokens=True)[:max_length]
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class WikiTextSampleBuilder:
    def __init__(self, dataset_root: Path, tokenizer: LocalSentencePieceTokenizer):
        self.dataset_root = dataset_root
        self.tokenizer = tokenizer
        self.text_rows = self._load_rows()

    def _load_rows(self) -> List[Dict[str, object]]:
        parquet_files = sorted((self.dataset_root / "wikitext-103-raw-v1").glob("train-*.parquet"))
        rows: List[Dict[str, object]] = []
        for file_path in parquet_files:
            frame = pd.read_parquet(file_path)
            for row_idx, text in enumerate(frame["text"].tolist()):
                if not isinstance(text, str):
                    continue
                stripped = text.strip()
                if not stripped or stripped.startswith("="):
                    continue
                rows.append({"file": str(file_path), "row_idx": row_idx, "text": stripped})
        if not rows:
            raise RuntimeError("No usable WikiText rows were found.")
        return rows

    def build_samples(self, sample_count: int, target_lengths: List[int]) -> List[SampleRecord]:
        cursor = 0
        samples: List[SampleRecord] = []
        total_rows = len(self.text_rows)
        for sample_id in range(1, sample_count + 1):
            target_length = target_lengths[sample_id - 1]
            chunks: List[str] = []
            source_rows: List[int] = []
            source_files: List[str] = []
            token_length = 0
            while token_length < target_length and cursor < total_rows:
                row = self.text_rows[cursor]
                chunks.append(str(row["text"]))
                source_rows.append(int(row["row_idx"]))
                source_files.append(str(row["file"]))
                token_length = len(self.tokenizer.encode("\n\n".join(chunks), add_special_tokens=True))
                cursor += 1
            if token_length < target_length:
                raise RuntimeError(f"Ran out of WikiText rows while building sample {sample_id}.")
            samples.append(
                SampleRecord(
                    sample_id=sample_id,
                    text="\n\n".join(chunks),
                    source_file=source_files[0],
                    source_rows=source_rows,
                    target_length=target_length,
                    tokenized_length=token_length,
                )
            )
        return samples


class BigBirdAttentionCaptureManager:
    def __init__(self, model: BigBirdModel, attention_type: str):
        self.model = model
        self.attention_type = attention_type
        self.original_forwards = {}
        self.captures: Dict[int, LayerCapture] = {}

    def attach(self) -> None:
        for layer_idx, layer in enumerate(self.model.encoder.layer):
            module = layer.attention.self
            self.original_forwards[layer_idx] = module.forward
            if self.attention_type == "block_sparse":
                module.forward = MethodType(self._make_block_sparse_forward(layer_idx, module), module)
            else:
                module.forward = MethodType(self._make_full_forward(layer_idx), module)

    def clear(self) -> None:
        self.captures = {}

    def detach(self) -> None:
        for layer_idx, layer in enumerate(self.model.encoder.layer):
            layer.attention.self.forward = self.original_forwards[layer_idx]

    def _make_full_forward(self, layer_idx: int):
        original_forward = self.original_forwards[layer_idx]

        def wrapped(module_self, *args, **kwargs):
            outputs = original_forward(*args, **kwargs)
            dense = outputs[1].detach().cpu() if len(outputs) > 1 and torch.is_tensor(outputs[1]) else None
            hidden_states = args[0]
            self.captures[layer_idx] = LayerCapture(
                mode="original_full",
                layer_id=layer_idx,
                seq_len=int(hidden_states.shape[1]),
                block_size=None,
                num_heads=int(module_self.num_attention_heads),
                num_random_blocks=None,
                sparse_pieces={"attention_probs": dense} if dense is not None else {},
                rand_attn=None,
                dense_attention=dense,
            )
            return outputs

        return wrapped

    def _make_block_sparse_forward(self, layer_idx: int, module: BigBirdBlockSparseAttention):
        original_forward = self.original_forwards[layer_idx]

        def wrapped(module_self, *args, **kwargs):
            hidden_states = args[0]
            seq_len = int(hidden_states.shape[1])
            batch_size = int(hidden_states.shape[0])
            softmax_outputs: List[torch.Tensor] = []
            original_softmax = F.softmax

            def capture_softmax(input_tensor, *softmax_args, **softmax_kwargs):
                output = original_softmax(input_tensor, *softmax_args, **softmax_kwargs)
                softmax_outputs.append(output.detach().cpu())
                return output

            F.softmax = capture_softmax
            try:
                outputs = original_forward(*args, **kwargs)
            finally:
                F.softmax = original_softmax

            dense_attention = outputs[1].detach().cpu() if len(outputs) > 1 and torch.is_tensor(outputs[1]) else None
            if len(softmax_outputs) != 5:
                raise RuntimeError(
                    f"Expected 5 softmax outputs for BigBird block-sparse attention, got {len(softmax_outputs)}"
                )
            sparse_pieces = {
                name: tensor for name, tensor in zip(SPARSE_PIECE_NAMES, softmax_outputs)
            }
            rand_attn = compute_rand_attn_layout(module_self, seq_len=seq_len, batch_size=batch_size).cpu()
            self.captures[layer_idx] = LayerCapture(
                mode="block_sparse",
                layer_id=layer_idx,
                seq_len=seq_len,
                block_size=int(module_self.block_size),
                num_heads=int(module_self.num_attention_heads),
                num_random_blocks=int(module_self.num_random_blocks),
                sparse_pieces=sparse_pieces,
                rand_attn=rand_attn,
                dense_attention=dense_attention,
            )
            return outputs

        return wrapped


def compute_rand_attn_layout(module: BigBirdBlockSparseAttention, seq_len: int, batch_size: int) -> torch.Tensor:
    from_block_size = to_block_size = module.block_size
    n_heads = module.num_attention_heads
    n_rand_blocks = module.num_random_blocks
    np.random.seed(module.seed)
    if seq_len in [1024, 3072, 4096]:
        rand_attn = [
            module._bigbird_block_rand_mask(
                module.max_seqlen,
                module.max_seqlen,
                from_block_size,
                to_block_size,
                n_rand_blocks,
                last_idx=1024,
            )[: (seq_len // from_block_size - 2)]
            for _ in range(n_heads)
        ]
    else:
        plan_from_length, plan_num_rand_blocks = module._get_rand_attn_plan(seq_len, from_block_size, n_rand_blocks)
        rand_attn = module._bigbird_block_rand_mask_with_head(
            from_seq_length=seq_len,
            to_seq_length=seq_len,
            from_block_size=from_block_size,
            to_block_size=to_block_size,
            num_heads=n_heads,
            plan_from_length=plan_from_length,
            plan_num_rand_blocks=plan_num_rand_blocks,
        )
    rand_attn = np.stack(rand_attn, axis=0)
    rand_attn = torch.tensor(rand_attn, dtype=torch.long).unsqueeze(0)
    return rand_attn.repeat(batch_size, 1, 1, 1)


def quantize_dequantize_fp16(tensor: torch.Tensor) -> torch.Tensor:
    q = torch.round(tensor.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    return (q.to(torch.float32) / 255.0).to(torch.float16)


def reconstruct_block_sparse_dense(capture: LayerCapture) -> torch.Tensor:
    block_size = int(capture.block_size)
    seq_len = int(capture.seq_len)
    num_heads = int(capture.num_heads)
    num_blocks = seq_len // block_size
    num_random_blocks = int(capture.num_random_blocks)

    first = capture.sparse_pieces["first_attn_weights"][0]
    second = capture.sparse_pieces["second_attn_weights"][0]
    middle = capture.sparse_pieces["middle_attn_weights"][0]
    second_last = capture.sparse_pieces["second_last_attn_weights"][0]
    last = capture.sparse_pieces["last_attn_weights"][0]
    rand_attn = capture.rand_attn[0]

    dense = torch.zeros((num_heads, seq_len, seq_len), dtype=torch.float32)
    dense[:, :block_size, :] = first
    dense[:, -block_size:, :] = last

    dense[:, block_size : 2 * block_size, : 3 * block_size] = second[:, :, : 3 * block_size]
    dense[:, block_size : 2 * block_size, -block_size:] = second[:, :, 3 * block_size : 4 * block_size]

    dense[:, -2 * block_size : -block_size, :block_size] = second_last[:, :, :block_size]
    dense[:, -2 * block_size : -block_size, -3 * block_size :] = second_last[:, :, block_size : 4 * block_size]

    dense_view = dense.view(num_heads, num_blocks, block_size, num_blocks, block_size)

    for head_idx in range(num_heads):
        second_rand = second[head_idx, :, 4 * block_size :].view(block_size, num_random_blocks, block_size)
        dense_view[head_idx, 1, :, rand_attn[head_idx, 0]] = second_rand

    num_middle = num_blocks - 4
    if num_middle > 0:
        middle_global_first = middle[:, :, :, :block_size]
        middle_sliding = middle[:, :, :, block_size : 4 * block_size]
        middle_random = middle[:, :, :, 4 * block_size : -block_size]
        middle_global_last = middle[:, :, :, -block_size:]

        dense[:, 2 * block_size : -2 * block_size, :block_size] = middle_global_first.reshape(
            num_heads, num_middle * block_size, block_size
        )
        dense[:, 2 * block_size : -2 * block_size, -block_size:] = middle_global_last.reshape(
            num_heads, num_middle * block_size, block_size
        )

        for q_idx in range(num_middle):
            sliding = middle_sliding[:, q_idx].view(num_heads, block_size, 3, block_size)
            dense_view[:, q_idx + 2, :, q_idx + 1 : q_idx + 4, :] = sliding

        for head_idx in range(num_heads):
            for q_idx in range(1, rand_attn.shape[1] - 1):
                random_slice = middle_random[head_idx, q_idx - 1].view(block_size, num_random_blocks, block_size)
                dense_view[head_idx, q_idx + 1, :, rand_attn[head_idx, q_idx]] = random_slice

    for head_idx in range(num_heads):
        second_last_rand = second_last[head_idx, :, 4 * block_size :].view(block_size, num_random_blocks, block_size)
        dense_view[head_idx, -2, :, rand_attn[head_idx, -1]] = second_last_rand

    return dense


def tensor_to_matrix_view(tensor: torch.Tensor) -> Optional[torch.Tensor]:
    if tensor.ndim < 2:
        return None
    return tensor.reshape(-1, tensor.shape[-1])


def compute_block_sparsity(matrix: torch.Tensor, block_size: int) -> Optional[float]:
    rows, cols = matrix.shape
    if rows < block_size or cols < block_size:
        return None
    row_blocks = rows // block_size
    col_blocks = cols // block_size
    if row_blocks == 0 or col_blocks == 0:
        return None
    trimmed = matrix[: row_blocks * block_size, : col_blocks * block_size]
    blocks = trimmed.reshape(row_blocks, block_size, col_blocks, block_size).permute(0, 2, 1, 3)
    zero_blocks = (blocks == 0).all(dim=-1).all(dim=-1)
    return float(zero_blocks.float().mean().item())


def compute_stats_row(
    *,
    sample_id: int,
    layer_id: int,
    mode: str,
    tensor_name: str,
    tensor: torch.Tensor,
    tensor_path: Optional[Path],
    plot_path: Optional[Path],
    is_saved: bool,
) -> Dict[str, object]:
    tensor_f32 = tensor.float()
    total = int(tensor.numel())
    num_zeros = int((tensor == 0).sum().item())
    row = {
        "sample_id": sample_id,
        "layer_id": layer_id,
        "attention_mode": mode,
        "tensor_name": tensor_name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "is_saved": is_saved,
        "tensor_path": str(tensor_path) if tensor_path else "",
        "plot_path": str(plot_path) if plot_path else "",
        "total_elements": total,
        "num_zeros": num_zeros,
        "sparsity_ratio": num_zeros / total,
        "min": float(tensor_f32.min().item()),
        "max": float(tensor_f32.max().item()),
        "mean": float(tensor_f32.mean().item()),
        "std": float(tensor_f32.std(unbiased=False).item()),
        "num_abs_lt_1e_4": int((tensor_f32.abs() < 1e-4).sum().item()),
        "num_abs_lt_1e_3": int((tensor_f32.abs() < 1e-3).sum().item()),
        "num_abs_lt_1e_2": int((tensor_f32.abs() < 1e-2).sum().item()),
    }
    row["ratio_abs_lt_1e_4"] = row["num_abs_lt_1e_4"] / total
    row["ratio_abs_lt_1e_3"] = row["num_abs_lt_1e_3"] / total
    row["ratio_abs_lt_1e_2"] = row["num_abs_lt_1e_2"] / total

    matrix_view = tensor_to_matrix_view(tensor_f32)
    for block_size in [4, 8, 16, 32]:
        key = f"block_sparsity_{block_size}"
        row[key] = compute_block_sparsity(matrix_view, block_size) if matrix_view is not None else None
    return row


def save_heatmap(matrix: torch.Tensor, output_path: Path, title: Optional[str] = None) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
    image = ax.imshow(matrix.numpy(), cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest")
    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Attention", rotation=270, labelpad=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def select_plot_matrix(mode: str, tensor_name: str, tensor: torch.Tensor) -> torch.Tensor:
    if mode == "original_full" or tensor_name == "dense_reconstructed":
        return tensor.float().mean(dim=0)
    if tensor_name == "middle_attn_weights":
        return tensor.float().mean(dim=0).reshape(-1, tensor.shape[-1])
    return tensor.float().mean(dim=0)


def ensure_output_dirs(root: Path) -> Dict[str, Path]:
    dirs = {
        "root": root,
        "tensors": root / "tensors",
        "stats": root / "stats",
        "plots": root / "plots",
        "logs": root / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def save_tensor_payload(payload: Dict[str, object], output_path: Path) -> None:
    torch.save(payload, output_path)


def write_stats_files(stats_rows: List[Dict[str, object]], stats_dir: Path) -> None:
    frame = pd.DataFrame(stats_rows)
    frame.to_csv(stats_dir / "per_tensor_stats.csv", index=False)
    frame.to_json(stats_dir / "per_tensor_stats.json", orient="records", indent=2)

    summary = (
        frame.groupby(["attention_mode", "tensor_name"], dropna=False)
        .agg(
            count=("tensor_name", "size"),
            mean_sparsity_ratio=("sparsity_ratio", "mean"),
            mean_value=("mean", "mean"),
            mean_std=("std", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(stats_dir / "global_summary.csv", index=False)


def load_model(model_path: Path, attention_type: str, device: torch.device) -> BigBirdModel:
    config = BigBirdConfig.from_pretrained(model_path, local_files_only=True)
    config.attention_type = attention_type
    model = BigBirdModel.from_pretrained(model_path, config=config, local_files_only=True)
    model.to(device)
    model.eval()
    return model


def verify_capture_against_dense(
    model: BigBirdModel,
    tokenizer: LocalSentencePieceTokenizer,
    sample: SampleRecord,
    device: torch.device,
    log_path: Path,
) -> None:
    capture_manager = BigBirdAttentionCaptureManager(model, attention_type="block_sparse")
    capture_manager.attach()
    capture_manager.clear()

    model_inputs = tokenizer.encode_to_tensors(sample.text, sample.target_length, device)
    with torch.no_grad():
        outputs = model(**model_inputs, output_attentions=True, return_dict=True)

    layer0 = capture_manager.captures[0]
    reconstructed = reconstruct_block_sparse_dense(layer0)
    dense_return = outputs.attentions[0][0].detach().cpu()
    max_abs_diff = float((reconstructed - dense_return).abs().max().item())
    mean_abs_diff = float((reconstructed - dense_return).abs().mean().item())
    verification = {
        "sample_id": sample.sample_id,
        "target_length": sample.target_length,
        "tokenized_length": sample.tokenized_length,
        "max_abs_diff_layer0": max_abs_diff,
        "mean_abs_diff_layer0": mean_abs_diff,
    }
    with log_path.open("w") as handle:
        json.dump(verification, handle, indent=2)
    capture_manager.detach()


def process_samples(
    *,
    model: BigBirdModel,
    tokenizer: LocalSentencePieceTokenizer,
    samples: List[SampleRecord],
    output_dirs: Dict[str, Path],
    attention_type: str,
    device: torch.device,
) -> None:
    capture_manager = BigBirdAttentionCaptureManager(model, attention_type=attention_type)
    capture_manager.attach()

    stats_rows: List[Dict[str, object]] = []
    sample_manifest: List[Dict[str, object]] = []

    for sample in samples:
        model_inputs = tokenizer.encode_to_tensors(sample.text, sample.target_length, device)
        actual_len = int(model_inputs["input_ids"].shape[1])
        print(f"[run] sample={sample.sample_id:02d} mode={attention_type} seq_len={actual_len}")

        capture_manager.clear()
        with torch.no_grad():
            model(**model_inputs, output_attentions=(attention_type == "original_full"), return_dict=True)

        for layer_id, capture in sorted(capture_manager.captures.items()):
            prefix = f"bigbird_{attention_type}_sample_{sample.sample_id:02d}_layer_{layer_id:02d}"
            tensor_path = output_dirs["tensors"] / f"{prefix}.pt"

            if attention_type == "block_sparse":
                saved_tensors = {}
                for tensor_name, tensor in capture.sparse_pieces.items():
                    tensor_b0 = tensor[0]
                    tensor_fp16 = quantize_dequantize_fp16(tensor_b0)
                    saved_tensors[tensor_name] = tensor_fp16
                    plot_matrix = select_plot_matrix(attention_type, tensor_name, tensor_fp16)
                    plot_path = output_dirs["plots"] / f"{prefix}_{tensor_name}.png"
                    save_heatmap(plot_matrix.cpu(), plot_path, title=f"{tensor_name} avg-head")
                    stats_rows.append(
                        compute_stats_row(
                            sample_id=sample.sample_id,
                            layer_id=layer_id,
                            mode=attention_type,
                            tensor_name=tensor_name,
                            tensor=tensor_fp16,
                            tensor_path=tensor_path,
                            plot_path=plot_path,
                            is_saved=True,
                        )
                    )

                dense_reconstructed = quantize_dequantize_fp16(reconstruct_block_sparse_dense(capture))
                dense_plot_path = output_dirs["plots"] / f"{prefix}_dense_reconstructed.png"
                save_heatmap(
                    select_plot_matrix(attention_type, "dense_reconstructed", dense_reconstructed).cpu(),
                    dense_plot_path,
                    title="dense reconstructed avg-head",
                )
                stats_rows.append(
                    compute_stats_row(
                        sample_id=sample.sample_id,
                        layer_id=layer_id,
                        mode=attention_type,
                        tensor_name="dense_reconstructed",
                        tensor=dense_reconstructed,
                        tensor_path=None,
                        plot_path=dense_plot_path,
                        is_saved=False,
                    )
                )

                payload = {
                    "model_name": "bigbird",
                    "attention_mode": attention_type,
                    "sample_id": sample.sample_id,
                    "layer_id": layer_id,
                    "seq_len": capture.seq_len,
                    "block_size": capture.block_size,
                    "num_heads": capture.num_heads,
                    "num_random_blocks": capture.num_random_blocks,
                    "saved_format": "real_block_sparse_post_softmax_pieces",
                    "tensor_dtypes": {name: str(t.dtype) for name, t in saved_tensors.items()},
                    "tensor_shapes": {name: list(t.shape) for name, t in saved_tensors.items()},
                    "rand_attn": capture.rand_attn[0].clone(),
                    "tensors": saved_tensors,
                }
                save_tensor_payload(payload, tensor_path)
            else:
                dense = capture.dense_attention[0]
                dense_fp16 = quantize_dequantize_fp16(dense)
                plot_path = output_dirs["plots"] / f"{prefix}.png"
                save_heatmap(select_plot_matrix(attention_type, "attention_probs", dense_fp16).cpu(), plot_path)
                payload = {
                    "model_name": "bigbird",
                    "attention_mode": attention_type,
                    "sample_id": sample.sample_id,
                    "layer_id": layer_id,
                    "seq_len": capture.seq_len,
                    "num_heads": capture.num_heads,
                    "saved_format": "full_square_post_softmax_attention",
                    "tensor_dtype": str(dense_fp16.dtype),
                    "tensor_shape": list(dense_fp16.shape),
                    "tensor": dense_fp16,
                }
                save_tensor_payload(payload, tensor_path)
                stats_rows.append(
                    compute_stats_row(
                        sample_id=sample.sample_id,
                        layer_id=layer_id,
                        mode=attention_type,
                        tensor_name="attention_probs",
                        tensor=dense_fp16,
                        tensor_path=tensor_path,
                        plot_path=plot_path,
                        is_saved=True,
                    )
                )

        sample_manifest.append(
            {
                "sample_id": sample.sample_id,
                "text_source": sample.source_file,
                "source_rows": sample.source_rows,
                "target_length": sample.target_length,
                "tokenized_length_before_truncation": sample.tokenized_length,
                "actual_length_used": actual_len,
                "text_preview": sample.text[:500],
            }
        )

    capture_manager.detach()
    write_stats_files(stats_rows, output_dirs["stats"])
    with (output_dirs["logs"] / "sample_manifest.json").open("w") as handle:
        json.dump(sample_manifest, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract BigBird post-softmax attention tensors offline.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--dataset-path", type=Path, default=DATASET_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--attention-type", choices=["block_sparse", "original_full"], default="block_sparse")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--verify-capture", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    output_dirs = ensure_output_dirs(args.output_root)
    device = resolve_device(args.device)

    target_lengths = TARGET_LENGTHS[: args.samples]
    tokenizer = LocalSentencePieceTokenizer(args.model_path)
    sample_builder = WikiTextSampleBuilder(args.dataset_path, tokenizer)
    samples = sample_builder.build_samples(args.samples, target_lengths)

    with (output_dirs["logs"] / "run_config.json").open("w") as handle:
        json.dump(
            {
                "model_path": str(args.model_path),
                "dataset_path": str(args.dataset_path),
                "attention_type": args.attention_type,
                "samples": args.samples,
                "target_lengths": target_lengths,
                "device": str(device),
            },
            handle,
            indent=2,
        )

    print(f"[load] model={args.model_path} attention_type={args.attention_type} device={device}")
    model = load_model(args.model_path, args.attention_type, device)

    if args.verify_capture and args.attention_type == "block_sparse":
        print("[verify] comparing reconstructed dense map with HF returned attention on one sample")
        verify_capture_against_dense(
            model=model,
            tokenizer=tokenizer,
            sample=samples[0],
            device=device,
            log_path=output_dirs["logs"] / "verification.json",
        )
        del model
        model = load_model(args.model_path, args.attention_type, device)

    process_samples(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        output_dirs=output_dirs,
        attention_type=args.attention_type,
        device=device,
    )
    print(f"[done] outputs written to {args.output_root}")


if __name__ == "__main__":
    main()
