from __future__ import annotations

import contextlib
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .io_utils import ensure_dir, write_csv, write_json
from .plotting import maybe_plot_tensor
from .stats import tensor_statistics


@dataclass
class CaptureScope:
    model_name: str
    dataset_name: str
    layer_name: str
    hook_source: str
    operator_type: str
    input_a_name: str
    input_b_name: str
    note: str = ""


class OperationRecorder:
    def __init__(self, output_root: Path, save_plots: bool = True) -> None:
        self.output_root = output_root
        self.save_plots = save_plots
        self._scope_stack: list[CaptureScope] = []
        self._records: list[dict[str, Any]] = []
        self._counter = itertools.count()
        self._printed_examples: set[str] = set()
        self._patched = False
        self._orig_linear = F.linear
        self._orig_mm = torch.mm
        self._orig_matmul = torch.matmul
        self._orig_sparse_mm = torch.sparse.mm

    @property
    def records(self) -> list[dict[str, Any]]:
        return list(self._records)

    @contextlib.contextmanager
    def scope(self, scope: CaptureScope):
        self._scope_stack.append(scope)
        try:
            yield
        finally:
            self._scope_stack.pop()

    @contextlib.contextmanager
    def patch_ops(self):
        if self._patched:
            yield
            return
        self._patched = True

        def linear_wrapper(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None):
            scope = self._active_scope("matmul")
            if scope is not None:
                self._record(scope, input_tensor, weight)
            return self._orig_linear(input_tensor, weight, bias)

        def mm_wrapper(input_a: torch.Tensor, input_b: torch.Tensor):
            scope = self._active_scope("matmul")
            if scope is not None:
                self._record(scope, input_a, input_b)
            return self._orig_mm(input_a, input_b)

        def matmul_wrapper(input_a: torch.Tensor, input_b: torch.Tensor):
            scope = self._active_scope("matmul")
            if scope is not None:
                self._record(scope, input_a, input_b)
            return self._orig_matmul(input_a, input_b)

        def sparse_mm_wrapper(input_a: torch.Tensor, input_b: torch.Tensor, reduce: str = "sum"):
            scope = self._active_scope("spmm")
            if scope is not None:
                self._record(scope, input_a, input_b)
            if reduce == "sum":
                return self._orig_sparse_mm(input_a, input_b)
            return self._orig_sparse_mm(input_a, input_b, reduce)

        F.linear = linear_wrapper
        torch.mm = mm_wrapper
        torch.matmul = matmul_wrapper
        torch.sparse.mm = sparse_mm_wrapper
        try:
            yield
        finally:
            F.linear = self._orig_linear
            torch.mm = self._orig_mm
            torch.matmul = self._orig_matmul
            torch.sparse.mm = self._orig_sparse_mm
            self._patched = False

    def _active_scope(self, operator_type: str) -> CaptureScope | None:
        if not self._scope_stack:
            return None
        scope = self._scope_stack[-1]
        if scope.operator_type != operator_type:
            return None
        return scope

    def _record(self, scope: CaptureScope, input_a: torch.Tensor, input_b: torch.Tensor) -> None:
        record_id = next(self._counter)
        base_dir = self.output_root / scope.model_name / scope.dataset_name
        tensors_dir = base_dir / "tensors"
        stats_dir = base_dir / "stats"
        plots_dir = base_dir / "plots"
        ensure_dir(tensors_dir)
        ensure_dir(stats_dir)
        if self.save_plots:
            ensure_dir(plots_dir)

        tensor_a_path = tensors_dir / f"{record_id:04d}_{scope.layer_name}_{scope.operator_type}_A.pt"
        tensor_b_path = tensors_dir / f"{record_id:04d}_{scope.layer_name}_{scope.operator_type}_B.pt"
        torch.save(input_a.detach().cpu(), tensor_a_path)
        torch.save(input_b.detach().cpu(), tensor_b_path)

        tensor_a_stats = tensor_statistics(input_a)
        tensor_b_stats = tensor_statistics(input_b)
        record = {
            "record_id": record_id,
            "model_name": scope.model_name,
            "dataset_name": scope.dataset_name,
            "layer_name": scope.layer_name,
            "hook_source": scope.hook_source,
            "operator_type": scope.operator_type,
            "note": scope.note,
            "input_a_name": scope.input_a_name,
            "input_a_path": str(tensor_a_path),
            "input_a_stats": tensor_a_stats,
            "input_b_name": scope.input_b_name,
            "input_b_path": str(tensor_b_path),
            "input_b_stats": tensor_b_stats,
        }
        self._records.append(record)
        write_json(stats_dir / f"{record_id:04d}_{scope.layer_name}_{scope.operator_type}.json", record)

        if self.save_plots:
            maybe_plot_tensor(input_a, plots_dir / f"{record_id:04d}_{scope.layer_name}_{scope.operator_type}_A")
            maybe_plot_tensor(input_b, plots_dir / f"{record_id:04d}_{scope.layer_name}_{scope.operator_type}_B")

        source_key = f"{scope.model_name}:{scope.dataset_name}:{scope.layer_name}:{scope.hook_source}:{scope.operator_type}"
        if source_key not in self._printed_examples:
            print(
                "[capture]",
                f"model={scope.model_name}",
                f"dataset={scope.dataset_name}",
                f"layer={scope.layer_name}",
                f"op={scope.operator_type}",
                f"{scope.input_a_name}={list(input_a.shape)}",
                f"{scope.input_b_name}={list(input_b.shape)}",
            )
            self._printed_examples.add(source_key)

    def flush(self) -> Path | None:
        if not self._records:
            return None
        first = self._records[0]
        stats_dir = self.output_root / first["model_name"] / first["dataset_name"] / "stats"
        write_json(stats_dir / "summary.json", self._records)
        write_csv(stats_dir / "summary.csv", self._records)
        return stats_dir
