from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .datasets import load_dataset
from .io_utils import ensure_dir, write_json
from .models import build_model
from .recorder import OperationRecorder


@dataclass
class ExperimentResult:
    model_name: str
    dataset_name: str
    num_records: int
    output_dir: str
    notes: list[str]


def run_experiment(model_name: str, dataset_name: str, root_dir: Path, output_root: Path, hidden_channels: int = 64, device: str = "cpu", save_plots: bool = True) -> ExperimentResult:
    ensure_dir(root_dir)
    ensure_dir(output_root)
    device_obj = torch.device(device)
    graph = load_dataset(dataset_name, root_dir)
    recorder = OperationRecorder(output_root=output_root, save_plots=save_plots)
    model = build_model(
        model_name=model_name,
        in_channels=graph.num_features,
        hidden_channels=hidden_channels,
        out_channels=graph.num_classes,
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        recorder=recorder,
        dataset_name=dataset_name,
        device=device_obj,
    ).to(device_obj)
    x = graph.x.to(device_obj)
    model.eval()
    with torch.no_grad():
        with recorder.patch_ops():
            _ = model(x)
    stats_dir = recorder.flush()
    result = ExperimentResult(
        model_name=model_name,
        dataset_name=dataset_name,
        num_records=len(recorder.records),
        output_dir=str(output_root / model_name / dataset_name),
        notes=model.notes,
    )
    if stats_dir is not None:
        write_json(stats_dir / "run_metadata.json", asdict(result))
    return result
