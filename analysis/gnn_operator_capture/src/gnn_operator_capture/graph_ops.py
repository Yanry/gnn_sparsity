from __future__ import annotations

import torch


def add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    device = edge_index.device
    loops = torch.arange(num_nodes, device=device)
    loop_edges = torch.stack([loops, loops], dim=0)
    return torch.cat([edge_index, loop_edges], dim=1)


def coalesce_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    values = torch.ones(edge_index.shape[1], device=edge_index.device)
    sparse = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes))
    sparse = sparse.coalesce()
    return sparse.indices()


def gcn_normalized_adjacency(edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
    edge_index = coalesce_edge_index(add_self_loops(edge_index.to(device), num_nodes), num_nodes)
    row, col = edge_index
    degree = torch.bincount(row, minlength=num_nodes).to(torch.float32)
    norm = torch.pow(degree, -0.5)
    norm[torch.isinf(norm)] = 0
    values = norm[row] * norm[col]
    coo = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes), device=device)
    return coo.coalesce().to_sparse_csr()


def row_normalized_adjacency(edge_index: torch.Tensor, num_nodes: int, device: torch.device, add_loops: bool = False) -> torch.Tensor:
    if add_loops:
        edge_index = add_self_loops(edge_index, num_nodes)
    edge_index = coalesce_edge_index(edge_index.to(device), num_nodes)
    row = edge_index[0]
    degree = torch.bincount(row, minlength=num_nodes).to(torch.float32).clamp(min=1)
    values = 1.0 / degree[row]
    coo = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes), device=device)
    return coo.coalesce().to_sparse_csr()


def sum_adjacency(edge_index: torch.Tensor, num_nodes: int, device: torch.device, add_loops: bool = False) -> torch.Tensor:
    if add_loops:
        edge_index = add_self_loops(edge_index, num_nodes)
    edge_index = coalesce_edge_index(edge_index.to(device), num_nodes)
    values = torch.ones(edge_index.shape[1], dtype=torch.float32, device=device)
    coo = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes), device=device)
    return coo.coalesce().to_sparse_csr()
