import json
from pathlib import Path

import torch

OUTPUT_ROOT = Path("/home/nizhj/gnn_sparsity/torch-rgcn/output")
DATASET_ROOT = Path("/home/nizhj/customized_data/dataset")


def to_tensor(payload):
    if isinstance(payload, torch.Tensor):
        return payload

    if isinstance(payload, dict):
        # Some dumps store sparse matrices as {indices, values, shape}.
        if {"indices", "values", "shape"}.issubset(payload.keys()):
            indices = payload["indices"]
            values = payload["values"]
            shape = payload["shape"]

            if isinstance(shape, torch.Tensor):
                shape = tuple(int(x) for x in shape.tolist())
            else:
                shape = tuple(int(x) for x in shape)

            return torch.sparse_coo_tensor(indices.long(), values, size=shape).coalesce()

        # Fallback: try common container keys.
        for key in ("tensor", "data", "value", "matrix", "mat"):
            if key in payload:
                candidate = payload[key]
                if isinstance(candidate, torch.Tensor):
                    return candidate

        # Fallback: if dict contains a single tensor-like value, use it.
        tensor_values = [v for v in payload.values() if isinstance(v, torch.Tensor)]
        if len(tensor_values) == 1:
            return tensor_values[0]

    raise TypeError(f"Unsupported payload type for tensor extraction: {type(payload)}")


def load_tensor_as_coo(pt_path: Path):
    raw = torch.load(pt_path, map_location="cpu", weights_only=True)
    tensor = to_tensor(raw)

    if tensor.is_sparse:
        sparse = tensor.coalesce()
    else:
        sparse = tensor.float().to_sparse().coalesce()

    indices = sparse.indices()
    values = sparse.values().float()

    row_idx = indices[0].tolist()
    col_idx = indices[1].tolist()
    vals = values.tolist()

    rows, cols = sparse.shape
    return rows, cols, row_idx, col_idx, vals


def collect_sparse_records(records):
    selected = []
    for rec in records:
        op_type = rec.get("op_type")
        if op_type not in ("torch.spmm", "torch.mm"):
            continue

        sparse_input_idx = None
        if rec.get("input1_sparse", False):
            sparse_input_idx = 1
        elif rec.get("input2_sparse", False):
            sparse_input_idx = 2

        if sparse_input_idx is None:
            continue

        selected.append((rec, sparse_input_idx))

    # Stable output order: spmm first, mm second, then by call id.
    op_order = {"torch.spmm": 0, "torch.mm": 1}
    selected.sort(key=lambda x: (op_order.get(x[0]["op_type"], 99), x[0]["call_id"]))
    return selected


def write_mtx(out_path: Path, rows: int, cols: int, row_idx, col_idx, vals):
    nnz = len(row_idx)
    with out_path.open("w") as fout:
        fout.write("%%MatrixMarket matrix coordinate real general\n")
        fout.write(f"{rows} {cols} {nnz}\n")
        for r, c, v in zip(row_idx, col_idx, vals):
            # MatrixMarket uses 1-based indexing.
            fout.write(f"{r + 1} {c + 1} {v:.6g}\n")


def main():
    model_dirs = sorted([p for p in OUTPUT_ROOT.iterdir() if p.is_dir()])

    if not model_dirs:
        raise RuntimeError(f"No model folders found under {OUTPUT_ROOT}")

    print(f"Found {len(model_dirs)} model folders.")

    for model_dir in model_dirs:
        metadata_path = model_dir / "gemm_metadata.json"
        if not metadata_path.exists():
            print(f"[SKIP] {model_dir.name}: missing gemm_metadata.json")
            continue

        with metadata_path.open() as f:
            metadata = json.load(f)
        records = metadata.get("records", [])

        sparse_records = collect_sparse_records(records)
        if not sparse_records:
            print(f"[SKIP] {model_dir.name}: no sparse torch.spmm/torch.mm record")
            continue

        model_prefix = model_dir.name.split("_")[0]
        print(f"\n[{model_dir.name}] exporting {len(sparse_records)} sparse matrices")
        for local_idx, (rec, sparse_input_idx) in enumerate(sparse_records):
            op_type = rec["op_type"]
            call_id = rec["call_id"]
            op_short = op_type.split(".")[-1]

            pt_name = f"{op_type}_call_{call_id:04d}_input{sparse_input_idx}.pt"
            pt_path = model_dir / pt_name

            if not pt_path.exists():
                print(f"  [SKIP] tensor file not found -> {pt_name}")
                continue

            rows, cols, row_idx, col_idx, vals = load_tensor_as_coo(pt_path)
            nnz = len(row_idx)
            density = nnz / (rows * cols)

            # Keep exactly one MTX file in each folder, named as <modelPrefix><index>.
            out_dir = DATASET_ROOT / f"{model_prefix}{local_idx}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "sparse.mtx"

            write_mtx(out_path, rows, cols, row_idx, col_idx, vals)

            print(
                f"  {op_short}_call_{call_id:04d}: shape={rows}x{cols}, nnz={nnz}, density={density:.4%} -> {out_path}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
