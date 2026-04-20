from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch.serialization import safe_globals
from torch_geometric.data.data import Data, DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import BaseStorage, EdgeStorage, GlobalStorage, NodeStorage
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


@dataclass
class GraphData:
    name: str
    x: torch.Tensor
    y: torch.Tensor
    edge_index: torch.Tensor
    num_nodes: int
    num_features: int
    num_classes: int


def load_dataset(name: str, root: Path) -> GraphData:
    name = name.lower()
    if name in {"cora", "citeseer", "pubmed"}:
        dataset = Planetoid(root=str(root / "planetoid"), name={"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}[name], transform=NormalizeFeatures())
        data = dataset[0]
        return GraphData(
            name=name,
            x=data.x,
            y=data.y,
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_features=dataset.num_features,
            num_classes=dataset.num_classes,
        )

    if name == "ogbn-arxiv":
        with safe_globals([Data, DataEdgeAttr, DataTensorAttr, BaseStorage, NodeStorage, EdgeStorage, GlobalStorage]):
            dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=str(root / "ogb"))
        data = dataset[0]
        return GraphData(
            name=name,
            x=data.x,
            y=data.y.view(-1),
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_features=dataset.num_features,
            num_classes=dataset.num_classes,
        )

    raise ValueError(f"Unsupported dataset: {name}")
