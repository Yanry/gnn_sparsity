from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def _shape_token(shape: list[int]) -> str:
    return "x".join(str(dim) for dim in shape)


def build_report(output_root: Path) -> str:
    summary_paths = sorted(output_root.glob("*/*/stats/summary.json"))
    records = []
    for path in summary_paths:
        with path.open("r", encoding="utf-8") as handle:
            records.extend(json.load(handle))

    if not records:
        return "# GNN Operator Capture Report\n\nNo runs were found."

    models = sorted({record["model_name"] for record in records})
    datasets = sorted({record["dataset_name"] for record in records})
    operators = sorted({record["operator_type"] for record in records})

    by_operator = defaultdict(list)
    by_pair = defaultdict(list)
    notes = []
    for record in records:
        by_operator[record["operator_type"]].append(record)
        by_pair[(record["model_name"], record["dataset_name"])].append(record)
        if record.get("note"):
            notes.append(record["note"])

    lines = [
        "# GNN Operator Capture Report",
        "",
        "## Coverage",
        f"- Models: {', '.join(models)}",
        f"- Datasets: {', '.join(datasets)}",
        f"- Operator types: {', '.join(operators)}",
        f"- Captured invocations: {len(records)}",
        "",
        "## Typical Tensor Shapes",
    ]

    for operator_type, operator_records in sorted(by_operator.items()):
        a_shapes = sorted({_shape_token(record["input_a_stats"]["shape"]) for record in operator_records})
        b_shapes = sorted({_shape_token(record["input_b_stats"]["shape"]) for record in operator_records})
        lines.append(f"- `{operator_type}` input A shapes: {', '.join(a_shapes[:8])}")
        lines.append(f"- `{operator_type}` input B shapes: {', '.join(b_shapes[:8])}")

    lines.extend(["", "## Sparsity Observations"])

    for operator_type, operator_records in sorted(by_operator.items()):
        a_sparsity = sum(record["input_a_stats"]["sparsity_ratio"] for record in operator_records) / len(operator_records)
        b_sparsity = sum(record["input_b_stats"]["sparsity_ratio"] for record in operator_records) / len(operator_records)
        lines.append(
            f"- `{operator_type}` average input sparsity: A={a_sparsity:.4f}, B={b_sparsity:.4f}"
        )

    densest = sorted(
        records,
        key=lambda record: record["input_a_stats"]["sparsity_ratio"],
        reverse=True,
    )[:4]
    lines.extend(["", "## Representative High-Sparsity Captures"])
    for record in densest:
        lines.append(
            f"- `{record['model_name']}` on `{record['dataset_name']}` at `{record['layer_name']}` "
            f"({record['operator_type']}) has input A sparsity {record['input_a_stats']['sparsity_ratio']:.4f} "
            f"with shape {_shape_token(record['input_a_stats']['shape'])}"
        )

    lines.extend(["", "## Per Model-Dataset Notes"])
    for (model_name, dataset_name), pair_records in sorted(by_pair.items()):
        matmul_count = sum(record["operator_type"] == "matmul" for record in pair_records)
        spmm_count = sum(record["operator_type"] == "spmm" for record in pair_records)
        mean_a_sparsity = sum(record["input_a_stats"]["sparsity_ratio"] for record in pair_records) / len(pair_records)
        mean_b_sparsity = sum(record["input_b_stats"]["sparsity_ratio"] for record in pair_records) / len(pair_records)
        lines.append(
            f"- `{model_name}` on `{dataset_name}`: {matmul_count} matmul captures, {spmm_count} spmm captures, "
            f"mean operand sparsity A={mean_a_sparsity:.4f}, B={mean_b_sparsity:.4f}"
        )

    if notes:
        lines.extend(["", "## Limitations and Notes"])
        for note in sorted(set(notes)):
            lines.append(f"- {note}")

    return "\n".join(lines) + "\n"
