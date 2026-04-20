from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax

from .graph_ops import gcn_normalized_adjacency, row_normalized_adjacency, sum_adjacency
from .recorder import CaptureScope, OperationRecorder


@dataclass
class ModelArtifacts:
    model_name: str
    notes: list[str]


class InstrumentedModel(nn.Module):
    def __init__(self, model_name: str, recorder: OperationRecorder, dataset_name: str) -> None:
        super().__init__()
        self.model_name = model_name
        self.recorder = recorder
        self.dataset_name = dataset_name
        self.notes: list[str] = []

    def _scope(self, layer_name: str, hook_source: str, operator_type: str, input_a_name: str, input_b_name: str, note: str = ""):
        return self.recorder.scope(
            CaptureScope(
                model_name=self.model_name,
                dataset_name=self.dataset_name,
                layer_name=layer_name,
                hook_source=hook_source,
                operator_type=operator_type,
                input_a_name=input_a_name,
                input_b_name=input_b_name,
                note=note,
            )
        )


class GCNNet(InstrumentedModel):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, edge_index: torch.Tensor, num_nodes: int, recorder: OperationRecorder, dataset_name: str, device: torch.device) -> None:
        super().__init__("gcn", recorder, dataset_name)
        self.lin1 = nn.Linear(in_channels, hidden_channels, bias=False)
        self.lin2 = nn.Linear(hidden_channels, out_channels, bias=False)
        self.register_buffer("adj", gcn_normalized_adjacency(edge_index, num_nodes, device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with self._scope("layer0_linear", "torch.nn.functional.linear", "matmul", "activation", "weight"):
            x = self.lin1(x)
        with self._scope("layer0_spmm", "torch.sparse.mm", "spmm", "normalized_adj", "projected_features"):
            x = torch.sparse.mm(self.adj, x)
        x = F.relu(x)
        with self._scope("layer1_linear", "torch.nn.functional.linear", "matmul", "activation", "weight"):
            x = self.lin2(x)
        with self._scope("layer1_spmm", "torch.sparse.mm", "spmm", "normalized_adj", "projected_features"):
            x = torch.sparse.mm(self.adj, x)
        return x


class GraphSAGENet(InstrumentedModel):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, edge_index: torch.Tensor, num_nodes: int, recorder: OperationRecorder, dataset_name: str, device: torch.device) -> None:
        super().__init__("graphsage", recorder, dataset_name)
        self.lin_neigh1 = nn.Linear(in_channels, hidden_channels, bias=False)
        self.lin_root1 = nn.Linear(in_channels, hidden_channels, bias=False)
        self.lin_neigh2 = nn.Linear(hidden_channels, out_channels, bias=False)
        self.lin_root2 = nn.Linear(hidden_channels, out_channels, bias=False)
        self.register_buffer("adj", row_normalized_adjacency(edge_index, num_nodes, device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with self._scope("layer0_spmm", "torch.sparse.mm", "spmm", "mean_adj", "node_features"):
            neigh = torch.sparse.mm(self.adj, x)
        with self._scope("layer0_neigh_linear", "torch.nn.functional.linear", "matmul", "neighbor_features", "weight"):
            neigh = self.lin_neigh1(neigh)
        with self._scope("layer0_root_linear", "torch.nn.functional.linear", "matmul", "root_features", "weight"):
            root = self.lin_root1(x)
        x = F.relu(neigh + root)
        with self._scope("layer1_spmm", "torch.sparse.mm", "spmm", "mean_adj", "node_features"):
            neigh = torch.sparse.mm(self.adj, x)
        with self._scope("layer1_neigh_linear", "torch.nn.functional.linear", "matmul", "neighbor_features", "weight"):
            neigh = self.lin_neigh2(neigh)
        with self._scope("layer1_root_linear", "torch.nn.functional.linear", "matmul", "root_features", "weight"):
            root = self.lin_root2(x)
        return neigh + root


class GINNet(InstrumentedModel):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, edge_index: torch.Tensor, num_nodes: int, recorder: OperationRecorder, dataset_name: str, device: torch.device) -> None:
        super().__init__("gin", recorder, dataset_name)
        self.eps = nn.Parameter(torch.tensor(0.1))
        self.mlp1 = nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        self.mlp2 = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, out_channels))
        self.register_buffer("adj", sum_adjacency(edge_index, num_nodes, device))

    def _mlp(self, block: nn.Sequential, x: torch.Tensor, prefix: str) -> torch.Tensor:
        with self._scope(f"{prefix}_linear0", "torch.nn.functional.linear", "matmul", "activation", "weight"):
            x = block[0](x)
        x = block[1](x)
        with self._scope(f"{prefix}_linear1", "torch.nn.functional.linear", "matmul", "activation", "weight"):
            x = block[2](x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with self._scope("layer0_spmm", "torch.sparse.mm", "spmm", "sum_adj", "node_features"):
            agg = torch.sparse.mm(self.adj, x)
        x = self._mlp(self.mlp1, (1 + self.eps) * x + agg, "layer0_mlp")
        with self._scope("layer1_spmm", "torch.sparse.mm", "spmm", "sum_adj", "node_features"):
            agg = torch.sparse.mm(self.adj, x)
        x = self._mlp(self.mlp2, (1 + self.eps) * x + agg, "layer1_mlp")
        return x


class GATNet(InstrumentedModel):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, edge_index: torch.Tensor, num_nodes: int, recorder: OperationRecorder, dataset_name: str, device: torch.device) -> None:
        super().__init__("gat", recorder, dataset_name)
        self.lin1 = nn.Linear(in_channels, hidden_channels, bias=False)
        self.lin2 = nn.Linear(hidden_channels, out_channels, bias=False)
        self.att_src1 = nn.Parameter(torch.empty(hidden_channels))
        self.att_dst1 = nn.Parameter(torch.empty(hidden_channels))
        self.att_src2 = nn.Parameter(torch.empty(out_channels))
        self.att_dst2 = nn.Parameter(torch.empty(out_channels))
        self.edge_index = edge_index.to(device)
        self.num_nodes = num_nodes
        nn.init.xavier_uniform_(self.att_src1.unsqueeze(0))
        nn.init.xavier_uniform_(self.att_dst1.unsqueeze(0))
        nn.init.xavier_uniform_(self.att_src2.unsqueeze(0))
        nn.init.xavier_uniform_(self.att_dst2.unsqueeze(0))
        self.notes.append(
            "GAT attention is represented explicitly as a sparse weighted adjacency built from edge-wise coefficients; "
            "the captured SpMM corresponds to that attention matrix multiplied by projected node features."
        )

    def _attention_spmm(self, edge_index: torch.Tensor, features: torch.Tensor, att_src: torch.Tensor, att_dst: torch.Tensor, layer_name: str) -> torch.Tensor:
        row, col = edge_index
        with self._scope(f"{layer_name}_att_src", "torch.matmul", "matmul", "projected_features", "attention_vector"):
            src_scores = torch.matmul(features, att_src)
        with self._scope(f"{layer_name}_att_dst", "torch.matmul", "matmul", "projected_features", "attention_vector"):
            dst_scores = torch.matmul(features, att_dst)
        edge_scores = F.leaky_relu(src_scores[row] + dst_scores[col], negative_slope=0.2)
        alpha = softmax(edge_scores, row)
        attention = torch.sparse_coo_tensor(edge_index, alpha, (self.num_nodes, self.num_nodes), device=features.device).coalesce().to_sparse_csr()
        with self._scope(f"{layer_name}_spmm", "torch.sparse.mm", "spmm", "attention_adj", "projected_features", note="Edge-wise attention coefficients are materialized as a sparse matrix before multiplication."):
            return torch.sparse.mm(attention, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with self._scope("layer0_linear", "torch.nn.functional.linear", "matmul", "activation", "weight"):
            x = self.lin1(x)
        x = F.elu(self._attention_spmm(self.edge_index, x, self.att_src1, self.att_dst1, "layer0"))
        with self._scope("layer1_linear", "torch.nn.functional.linear", "matmul", "activation", "weight"):
            x = self.lin2(x)
        x = self._attention_spmm(self.edge_index, x, self.att_src2, self.att_dst2, "layer1")
        return x


def build_model(model_name: str, in_channels: int, hidden_channels: int, out_channels: int, edge_index: torch.Tensor, num_nodes: int, recorder: OperationRecorder, dataset_name: str, device: torch.device) -> InstrumentedModel:
    model_name = model_name.lower()
    if model_name == "gcn":
        return GCNNet(in_channels, hidden_channels, out_channels, edge_index, num_nodes, recorder, dataset_name, device)
    if model_name == "graphsage":
        return GraphSAGENet(in_channels, hidden_channels, out_channels, edge_index, num_nodes, recorder, dataset_name, device)
    if model_name == "gin":
        return GINNet(in_channels, hidden_channels, out_channels, edge_index, num_nodes, recorder, dataset_name, device)
    if model_name == "gat":
        return GATNet(in_channels, hidden_channels, out_channels, edge_index, num_nodes, recorder, dataset_name, device)
    raise ValueError(f"Unsupported model: {model_name}")
