from __future__ import annotations

import math
from typing import Any

import torch


BLOCK_SIZES = (1, 4, 8, 16, 32)
THRESHOLDS = (1e-12, 1e-8, 1e-4)


def _matrix_view(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor.reshape(1, 1)
    if tensor.ndim == 1:
        return tensor.reshape(1, -1)
    if tensor.ndim == 2:
        return tensor
    return tensor.reshape(-1, tensor.shape[-1])


def _sparse_layout_name(tensor: torch.Tensor) -> str | None:
    layout_map = {
        torch.sparse_coo: "COO",
        torch.sparse_csr: "CSR",
        torch.sparse_csc: "CSC",
        torch.sparse_bsr: "BSR",
        torch.sparse_bsc: "BSC",
        torch.strided: None,
    }
    return layout_map.get(tensor.layout, str(tensor.layout))


def _nnz(tensor: torch.Tensor) -> int:
    if tensor.layout == torch.sparse_coo:
        return int(tensor.coalesce().values().numel())
    if tensor.layout in {torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc}:
        return int(tensor.values().numel())
    return int(torch.count_nonzero(tensor).item())


def _numel_from_shape(shape: torch.Size) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _sparse_indices_and_values(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    coo = tensor if tensor.layout == torch.sparse_coo else tensor.to_sparse_coo()
    coo = coo.coalesce()
    return coo.indices(), coo.values()


def _block_sparsity(mask2d: torch.Tensor, block_rows: int, block_cols: int) -> dict[str, Any]:
    rows, cols = mask2d.shape
    pad_rows = (block_rows - rows % block_rows) % block_rows
    pad_cols = (block_cols - cols % block_cols) % block_cols
    if pad_rows or pad_cols:
        mask2d = torch.nn.functional.pad(mask2d, (0, pad_cols, 0, pad_rows))
    reshaped = mask2d.reshape(
        mask2d.shape[0] // block_rows,
        block_rows,
        mask2d.shape[1] // block_cols,
        block_cols,
    )
    active_blocks = reshaped.any(dim=1).any(dim=-1)
    nonzero_blocks = int(active_blocks.sum().item())
    total_blocks = int(active_blocks.numel())
    return {
        "block_rows": block_rows,
        "block_cols": block_cols,
        "nonzero_blocks": nonzero_blocks,
        "total_blocks": total_blocks,
        "block_sparsity_ratio": 1.0 - (nonzero_blocks / total_blocks if total_blocks else 0.0),
    }


def _sparse_block_sparsity(indices: torch.Tensor, shape: list[int], block_rows: int, block_cols: int) -> dict[str, Any]:
    rows, cols = shape
    total_blocks = math.ceil(rows / block_rows) * math.ceil(cols / block_cols)
    if indices.numel() == 0:
        nonzero_blocks = 0
    else:
        block_row = torch.div(indices[0], block_rows, rounding_mode="floor")
        block_col = torch.div(indices[1], block_cols, rounding_mode="floor")
        unique_blocks = torch.unique(torch.stack([block_row, block_col], dim=1), dim=0)
        nonzero_blocks = int(unique_blocks.shape[0])
    return {
        "block_rows": block_rows,
        "block_cols": block_cols,
        "nonzero_blocks": nonzero_blocks,
        "total_blocks": total_blocks,
        "block_sparsity_ratio": 1.0 - (nonzero_blocks / total_blocks if total_blocks else 0.0),
    }


def tensor_statistics(tensor: torch.Tensor) -> dict[str, Any]:
    tensor = tensor.detach().cpu()
    if tensor.layout != torch.strided:
        indices, values = _sparse_indices_and_values(tensor)
        shape = list(tensor.shape)
        matrix_shape = shape if len(shape) == 2 else list(_matrix_view(torch.empty(shape)).shape)
        total = _numel_from_shape(tensor.shape)
        nonzero = int(values.numel())
        zero_ratio = 1.0 - (nonzero / total if total else 0.0)
        values64 = values.to(torch.float64)
        sum_values = float(values64.sum().item()) if nonzero else 0.0
        sum_sq = float((values64 * values64).sum().item()) if nonzero else 0.0
        mean = sum_values / total if total else 0.0
        variance = max((sum_sq / total if total else 0.0) - mean * mean, 0.0)
        min_value = min(0.0, float(values64.min().item())) if nonzero else 0.0
        max_value = max(0.0, float(values64.max().item())) if nonzero else 0.0

        stats: dict[str, Any] = {
            "shape": shape,
            "matrix_view_shape": matrix_shape,
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "numel": total,
            "nonzero": nonzero,
            "sparsity_ratio": zero_ratio,
            "zero_ratio": zero_ratio,
            "min": min_value,
            "max": max_value,
            "mean": mean,
            "std": math.sqrt(variance),
            "is_structurally_sparse": True,
            "nnz": _nnz(tensor),
            "sparse_layout": _sparse_layout_name(tensor),
        }
        stats["block_sparsity"] = {
            f"{size}x{size}": _sparse_block_sparsity(indices, matrix_shape, size, size)
            for size in BLOCK_SIZES
        }
        thresholded: dict[str, float] = {}
        if torch.is_floating_point(values):
            abs_values = values.abs()
            for threshold in THRESHOLDS:
                kept = int((abs_values >= threshold).sum().item())
                thresholded[f"abs_lt_{threshold:.0e}"] = 1.0 - (kept / total if total else 0.0)
        stats["thresholded_sparsity"] = thresholded
        return stats

    dense = tensor
    matrix = _matrix_view(dense)
    abs_matrix = matrix.abs()
    nonzero = int(torch.count_nonzero(matrix).item())
    total = int(matrix.numel())
    zero_ratio = 1.0 - (nonzero / total if total else 0.0)
    work = matrix.to(torch.float64)

    stats = {
        "shape": list(tensor.shape),
        "matrix_view_shape": list(matrix.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": total,
        "nonzero": nonzero,
        "sparsity_ratio": zero_ratio,
        "zero_ratio": zero_ratio,
        "min": float(work.min().item()) if total else 0.0,
        "max": float(work.max().item()) if total else 0.0,
        "mean": float(work.mean().item()) if total else 0.0,
        "std": float(work.std(unbiased=False).item()) if total else 0.0,
        "is_structurally_sparse": False,
        "nnz": _nnz(tensor),
        "sparse_layout": _sparse_layout_name(tensor),
    }

    nz_mask = matrix != 0
    stats["block_sparsity"] = {
        f"{size}x{size}": _block_sparsity(nz_mask, size, size) for size in BLOCK_SIZES
    }

    thresholded = {}
    if torch.is_floating_point(matrix):
        for threshold in THRESHOLDS:
            thresholded[f"abs_lt_{threshold:.0e}"] = float((abs_matrix < threshold).sum().item() / total if total else 0.0)
    stats["thresholded_sparsity"] = thresholded
    return stats
