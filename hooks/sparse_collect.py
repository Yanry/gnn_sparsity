import os
import json
import numpy as np
import torch

from .sparse_metrics import gini, coeff_var, topk_share

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def torch_sparse_to_coo(adj: torch.Tensor):
    """
    adj: torch.sparse_coo_tensor
    return: (row, col, val, shape)
    """
    assert adj.is_sparse
    adj = adj.coalesce()
    idx = adj.indices().cpu().numpy()
    val = adj.values().detach().cpu().numpy()
    row = idx[0].astype(np.int64)
    col = idx[1].astype(np.int64)
    shape = tuple(adj.shape)
    return row, col, val, shape

def rowcol_nnz(row, col, shape):
    nrow, ncol = shape
    row_nnz = np.bincount(row, minlength=nrow)
    col_nnz = np.bincount(col, minlength=ncol)
    return row_nnz, col_nnz

def summarize_degree(nnz_arr: np.ndarray, name: str, topks=(1,10,100,1000)):
    nnz_arr = nnz_arr.astype(np.int64)
    out = {
        f"{name}_mean": float(nnz_arr.mean()),
        f"{name}_std": float(nnz_arr.std(ddof=0)),
        f"{name}_min": int(nnz_arr.min()) if nnz_arr.size else 0,
        f"{name}_max": int(nnz_arr.max()) if nnz_arr.size else 0,
        f"{name}_median": float(np.median(nnz_arr)) if nnz_arr.size else 0.0,
        f"{name}_p90": float(np.percentile(nnz_arr, 90)) if nnz_arr.size else 0.0,
        f"{name}_p99": float(np.percentile(nnz_arr, 99)) if nnz_arr.size else 0.0,
        f"{name}_gini": gini(nnz_arr.astype(np.float64)),
        f"{name}_cv": coeff_var(nnz_arr.astype(np.float64)),
    }
    out.update({f"{name}_{k}": v for k, v in topk_share(nnz_arr, ks=topks).items()})
    # hub 比例：degree >= 10x mean（论文里常用一种粗定义）
    mean = nnz_arr.mean() if nnz_arr.size else 0.0
    thr = 10 * mean
    out[f"{name}_hub_thr_10x_mean"] = float(thr)
    out[f"{name}_hub_frac_10x_mean"] = float((nnz_arr >= thr).mean()) if mean > 0 else 0.0
    return out

def collect_sparse_graph_stats(adj: torch.Tensor, out_dir: str, tag: str):
    """
    针对 norm_adj 做一次论文级统计，并保存：
    - json: 汇总指标
    - npz: row_nnz / col_nnz / shape / nnz
    """
    ensure_dir(out_dir)
    row, col, val, shape = torch_sparse_to_coo(adj)
    nnz = int(row.size)
    nrow, ncol = shape
    density = nnz / (nrow * ncol)

    row_nnz, col_nnz = rowcol_nnz(row, col, shape)

    summary = {
        "tag": tag,
        "shape": [int(nrow), int(ncol)],
        "nnz": nnz,
        "density": float(density),
        # 额外：值统计（norm_adj 通常有归一化权重）
        "value_abs_mean": float(np.mean(np.abs(val))) if val.size else 0.0,
        "value_abs_p99": float(np.percentile(np.abs(val), 99)) if val.size else 0.0,
        "value_min": float(val.min()) if val.size else 0.0,
        "value_max": float(val.max()) if val.size else 0.0,
    }
    summary.update(summarize_degree(row_nnz, "row_nnz"))
    summary.update(summarize_degree(col_nnz, "col_nnz"))

    # 保存
    with open(os.path.join(out_dir, f"{tag}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    np.savez_compressed(
        os.path.join(out_dir, f"{tag}_degree.npz"),
        row_nnz=row_nnz.astype(np.int64),
        col_nnz=col_nnz.astype(np.int64),
        shape=np.array([nrow, ncol], dtype=np.int64),
        nnz=np.array([nnz], dtype=np.int64),
    )

    return summary