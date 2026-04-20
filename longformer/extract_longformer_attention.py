import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import LongformerConfig, LongformerForMaskedLM, RobertaTokenizerFast


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"
MODEL_DIRS = {
    "longformer-base-4096": SCRIPT_DIR / "longformer-base-4096",
    "longformer-large-4096": SCRIPT_DIR / "longformer-large-4096",
}
MODEL_ALIASES = {
    "longformer-base-4096": "base",
    "longformer-large-4096": "large",
}
DEFAULT_TARGET_LENGTHS = [448, 512, 576, 640, 704, 768, 832, 896, 960, 1024]

TEXT_BANK = [
    (
        "Transformers process long documents by repeatedly mixing local context and a small number of global anchors. "
        "When a report grows from a few paragraphs to many pages, the attention pattern stops looking uniform and "
        "starts reflecting topic boundaries, repeated definitions, and the reappearance of key phrases."
    ),
    (
        "Systems work on long sequences often contains recurring structures: headings, bullet-like enumerations, "
        "code fragments, summaries, and transitions between sections. Those structures make neighboring tokens "
        "strongly related while still creating a few tokens that act as routing points for information flow."
    ),
    (
        "A sparse attention model such as Longformer does not attend densely to every token pair. Instead, it keeps "
        "a sliding window for local mixing and supplements that pattern with global tokens that can collect or "
        "broadcast information across the whole sequence."
    ),
    (
        "The objective in this experiment is not text generation. We only need the prefill computation, because that "
        "forward pass already contains the normalized attention probabilities used to weight the value vectors in each "
        "layer and each head."
    ),
    (
        "Repeated natural language paragraphs are useful for instrumentation because they create nontrivial but still "
        "interpretable patterns. Similar phrases appear again after hundreds of tokens, which can reveal how attention "
        "mass is distributed inside the allowed local window and around designated global positions."
    ),
    (
        "Quantization to uint8 followed by dequantization back to float16 is intentionally lossy. Many very small "
        "probabilities will collapse to exact zero after rounding, which makes sparsity statistics more informative "
        "for downstream storage and compression analysis."
    ),
    (
        "Visualization should emphasize structure instead of decoration. A white background and a blue colormap make "
        "it easy to see the banded sliding-window pattern, the vertical stripes created by global attention, and the "
        "differences between early and late transformer layers."
    ),
    (
        "This sample also includes a few domain-specific terms such as tensor hooks, softmax normalization, hidden "
        "states, sequence padding, and head-wise probability maps so that the token stream mixes common language with "
        "repeated technical phrases."
    ),
]


@dataclass
class LayerCapture:
    attn_probs: torch.Tensor
    global_attn_probs: Optional[torch.Tensor]
    one_sided_window: int


class AttentionCaptureManager:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.layer_outputs: Dict[int, LayerCapture] = {}

    def attach(self) -> None:
        for layer_idx, layer in enumerate(self.model.longformer.encoder.layer):
            handle = layer.attention.self.register_forward_hook(self._make_hook(layer_idx))
            self.handles.append(handle)

    def clear(self) -> None:
        self.layer_outputs = {}

    def detach(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _make_hook(self, layer_idx: int):
        def hook(module, _inputs, outputs):
            attn_probs = outputs[1].detach().cpu()
            global_attn_probs = None
            if len(outputs) > 2 and torch.is_tensor(outputs[2]):
                global_attn_probs = outputs[2].detach().cpu()
            self.layer_outputs[layer_idx] = LayerCapture(
                attn_probs=attn_probs,
                global_attn_probs=global_attn_probs,
                one_sided_window=module.one_sided_attn_window_size,
            )

        return hook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Longformer attention probabilities after softmax.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_DIRS.keys()),
        choices=list(MODEL_DIRS.keys()),
        help="Model directories to process.",
    )
    parser.add_argument(
        "--samples-per-model",
        type=int,
        default=10,
        help="Number of samples to run for each model.",
    )
    parser.add_argument(
        "--target-lengths",
        type=int,
        nargs="*",
        default=DEFAULT_TARGET_LENGTHS,
        help="Target token lengths, one per sample. If fewer than samples-per-model, the last value is reused.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for tensors, plots, and statistics.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing tensor and plot files.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_longformer_backbone(model_dir: Path, device: torch.device) -> LongformerForMaskedLM:
    config = LongformerConfig.from_pretrained(model_dir)
    model = LongformerForMaskedLM(config)
    state_dict = torch.load(model_dir / "pytorch_model.bin", map_location="cpu")

    remapped_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("roberta."):
            remapped_state_dict["longformer." + key[len("roberta."):]] = value
        elif key.startswith("pooler."):
            continue
        else:
            remapped_state_dict[key] = value

    missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
    allowed_missing = {"lm_head.decoder.bias"}
    missing_keys = [key for key in missing_keys if key not in allowed_missing]
    if missing_keys:
        raise RuntimeError(f"Missing keys after checkpoint remap for {model_dir.name}: {missing_keys}")
    unexpected_keys = [
        key
        for key in unexpected_keys
        if not key.startswith("pooler.") and not key.startswith("longformer.pooler.")
    ]
    if unexpected_keys:
        raise RuntimeError(f"Unexpected keys after checkpoint remap for {model_dir.name}: {unexpected_keys}")

    model.to(device)
    model.eval()
    return model


def build_sample_text(sample_idx: int) -> str:
    blocks = []
    for section_idx in range(24):
        paragraph = TEXT_BANK[(sample_idx + section_idx) % len(TEXT_BANK)]
        blocks.append(
            f"Section {section_idx + 1}. "
            f"Sample {sample_idx + 1} revisits long-context attention analysis. "
            f"{paragraph} "
            f"In this section, the narrative intentionally repeats anchor terms such as Longformer, softmax, "
            f"attention weights, quantization, dequantization, and visualization so the document contains stable "
            f"motifs across distant regions."
        )
    return "\n\n".join(blocks)


def build_inputs(
    tokenizer: RobertaTokenizerFast,
    sample_idx: int,
    target_length: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    text = build_sample_text(sample_idx)
    while True:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=target_length,
            add_special_tokens=True,
        )
        current_length = int(encoded["input_ids"].shape[1])
        if current_length >= target_length or len(text) > 200000:
            break
        text = text + "\n\n" + build_sample_text(sample_idx + len(text) % len(TEXT_BANK))

    encoded = {key: value.to(device) for key, value in encoded.items()}
    seq_len = int(encoded["input_ids"].shape[1])
    global_attention_mask = torch.zeros_like(encoded["input_ids"])
    global_positions = list(range(0, seq_len, 256))
    for pos in global_positions:
        global_attention_mask[:, pos] = 1
    encoded["global_attention_mask"] = global_attention_mask
    return {
        "text": text,
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "global_attention_mask": encoded["global_attention_mask"],
    }


def reconstruct_dense_attention(
    capture: LayerCapture,
    seq_len: int,
    global_positions: torch.Tensor,
) -> torch.Tensor:
    attn_probs = capture.attn_probs[0, :seq_len]
    num_heads = attn_probs.shape[1]
    num_global = int(global_positions.numel())
    dense = torch.zeros((num_heads, seq_len, seq_len), dtype=torch.float32)

    global_key_probs = attn_probs[:, :, :num_global].permute(1, 0, 2).contiguous()
    local_probs = attn_probs[:, :, num_global:].permute(1, 0, 2).contiguous()

    window = capture.one_sided_window
    local_offsets = torch.arange(-window, window + 1, dtype=torch.long)
    query_positions = torch.arange(seq_len, dtype=torch.long)
    key_positions = query_positions[:, None] + local_offsets[None, :]
    valid = (key_positions >= 0) & (key_positions < seq_len)
    clipped_key_positions = key_positions.clamp(0, seq_len - 1)
    expanded_queries = query_positions[:, None].expand_as(clipped_key_positions)

    for head_idx in range(num_heads):
        dense[head_idx, expanded_queries[valid], clipped_key_positions[valid]] = local_probs[head_idx][valid]

    if num_global > 0:
        dense[:, :, global_positions] = global_key_probs
        if capture.global_attn_probs is not None:
            global_query_probs = capture.global_attn_probs[0, :, :num_global, :seq_len]
            dense[:, global_positions, :] = global_query_probs

    return dense


def quantize_dequantize_to_fp16(attn: torch.Tensor) -> torch.Tensor:
    q = torch.round(attn.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    return (q.to(torch.float32) / 255.0).to(torch.float16)


def compute_sparsity_stats(attn_fp16: torch.Tensor) -> Dict[str, float]:
    attn = attn_fp16.float()
    total = int(attn.numel())
    zeros = int((attn_fp16 == 0).sum().item())
    lt_1e_3 = int((attn.abs() < 1e-3).sum().item())
    lt_1e_2 = int((attn.abs() < 1e-2).sum().item())
    return {
        "total_elements": total,
        "num_zeros": zeros,
        "sparsity_ratio": zeros / total,
        "num_abs_lt_1e_3": lt_1e_3,
        "ratio_abs_lt_1e_3": lt_1e_3 / total,
        "num_abs_lt_1e_2": lt_1e_2,
        "ratio_abs_lt_1e_2": lt_1e_2 / total,
    }


def save_heatmap(attn_fp16: torch.Tensor, output_path: Path) -> None:
    heatmap = attn_fp16.float().mean(dim=0).numpy()
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")
    image = ax.imshow(heatmap, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Attention", rotation=270, labelpad=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def write_stats(output_dir: Path, rows: List[Dict[str, object]]) -> None:
    csv_path = output_dir / "sparsity_stats.csv"
    json_path = output_dir / "sparsity_stats.json"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w") as handle:
        json.dump(rows, handle, indent=2)


def write_manifest(output_dir: Path, manifest_rows: List[Dict[str, object]]) -> None:
    manifest_path = output_dir / "sample_manifest.json"
    with manifest_path.open("w") as handle:
        json.dump(manifest_rows, handle, indent=2)


def ensure_target_lengths(target_lengths: List[int], sample_count: int) -> List[int]:
    if not target_lengths:
        raise ValueError("At least one target length is required.")
    if len(target_lengths) >= sample_count:
        return target_lengths[:sample_count]
    last_value = target_lengths[-1]
    return target_lengths + [last_value] * (sample_count - len(target_lengths))


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    tokenizer.model_max_length = 4096

    stats_rows: List[Dict[str, object]] = []
    manifest_rows: List[Dict[str, object]] = []
    target_lengths = ensure_target_lengths(args.target_lengths, args.samples_per_model)

    for model_name in args.models:
        model_dir = MODEL_DIRS[model_name]
        model_alias = MODEL_ALIASES[model_name]
        print(f"[load] {model_name} on {device}")
        model = load_longformer_backbone(model_dir, device)
        capture_manager = AttentionCaptureManager(model)
        capture_manager.attach()

        for sample_idx in range(args.samples_per_model):
            sample_id = sample_idx + 1
            target_length = target_lengths[sample_idx]
            sample_inputs = build_inputs(tokenizer, sample_idx, target_length, device)
            seq_len = int(sample_inputs["input_ids"].shape[1])
            global_positions = torch.nonzero(sample_inputs["global_attention_mask"][0], as_tuple=False).flatten().cpu()
            print(
                f"[run] model={model_alias} sample={sample_id:02d} target_len={target_length} "
                f"actual_len={seq_len} globals={global_positions.tolist()}"
            )

            capture_manager.clear()
            with torch.no_grad():
                model.longformer(
                    input_ids=sample_inputs["input_ids"],
                    attention_mask=sample_inputs["attention_mask"],
                    global_attention_mask=sample_inputs["global_attention_mask"],
                    output_attentions=True,
                    return_dict=True,
                )

            for layer_idx in sorted(capture_manager.layer_outputs):
                prefix = f"model_{model_alias}_sample_{sample_id:02d}_layer_{layer_idx:02d}"
                tensor_path = output_dir / f"{prefix}.pt"
                image_path = output_dir / f"{prefix}.png"
                if not args.overwrite and tensor_path.exists() and image_path.exists():
                    print(f"[skip] {prefix}")
                    continue

                dense_attn = reconstruct_dense_attention(
                    capture_manager.layer_outputs[layer_idx],
                    seq_len=seq_len,
                    global_positions=global_positions,
                )
                attn_fp16 = quantize_dequantize_to_fp16(dense_attn).cpu()
                torch.save(attn_fp16, tensor_path)
                save_heatmap(attn_fp16, image_path)

                row = {
                    "model_name": model_name,
                    "model_alias": model_alias,
                    "sample_id": sample_id,
                    "layer_id": layer_idx,
                    "seq_len": seq_len,
                    "num_heads": int(attn_fp16.shape[0]),
                    "global_token_count": int(global_positions.numel()),
                    "tensor_path": str(tensor_path),
                    "plot_path": str(image_path),
                }
                row.update(compute_sparsity_stats(attn_fp16))
                stats_rows.append(row)

            manifest_rows.append(
                {
                    "model_name": model_name,
                    "model_alias": model_alias,
                    "sample_id": sample_id,
                    "target_length": target_length,
                    "actual_length": seq_len,
                    "global_positions": global_positions.tolist(),
                    "text_preview": sample_inputs["text"][:400],
                }
            )

        capture_manager.detach()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if stats_rows:
        write_stats(output_dir, stats_rows)
    if manifest_rows:
        write_manifest(output_dir, manifest_rows)

    print(f"[done] wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
