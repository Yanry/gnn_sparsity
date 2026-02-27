import os
import json
import numpy as np
import scipy.sparse as sp
import torch
import os, sys
# import tensorflow as tf
import tensorflow as _tf2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 关键：让 LightGCN.py 里 `import tensorflow as tf` 也拿到 v1
sys.modules["tensorflow"] = tf

THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../project/LightGCN
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))   # .../project
sys.path.insert(0, PROJECT_ROOT)                                # 让 hooks 可被 import
sys.path.insert(0, THIS_DIR)                                    # 让 utility/LightGCN.py 可被 import（稳）

from utility.batch_test import args, data_generator  # parses args + builds data_generator
from LightGCN import LightGCN

from hooks.sparse_collect import collect_sparse_graph_stats
from hooks.sparse_viz import render_all_degree_plots


def scipy_to_torch_sparse(adj: sp.spmatrix) -> torch.Tensor:
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    idx = torch.tensor([adj.row, adj.col], dtype=torch.int64)
    val = torch.tensor(adj.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, size=adj.shape).coalesce()


def pick_adj():
    plain_adj, norm_adj, mean_adj, pre_adj = data_generator.get_adj_mat()
    if args.adj_type == "plain":
        return plain_adj, "plain_adj"
    if args.adj_type == "norm":
        return norm_adj, "norm_adj"
    if args.adj_type == "gcmc":
        return mean_adj, "mean_adj_gcmc"
    if args.adj_type == "pre":
        return pre_adj, "pre_adj"
    return mean_adj + sp.eye(mean_adj.shape[0]), "mean_plus_I"

def save_raw_adjacency(adj, out_dir, tag, sample_size=2000):
    """
    保存原始稀疏邻接矩阵（scipy sparse）
    """

    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()

    row = adj.row.astype(np.int64)
    col = adj.col.astype(np.int64)
    val = adj.data.astype(np.float32)

    # 1️⃣ 保存完整COO格式
    np.savez_compressed(
        os.path.join(out_dir, f"{tag}_adj_coo.npz"),
        row=row,
        col=col,
        val=val,
        shape=np.array(adj.shape, dtype=np.int64)
    )

    # 2️⃣ 保存元信息
    meta = {
        "shape": adj.shape,
        "nnz": int(adj.nnz),
        "density": float(adj.nnz / (adj.shape[0] * adj.shape[1]))
    }

    with open(os.path.join(out_dir, f"{tag}_adj_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # 3️⃣ 保存一个小dense采样（方便画heatmap）
    csr = adj.tocsr()
    n = min(sample_size, csr.shape[0])
    sample_dense = csr[:n, :n].toarray()
    np.save(os.path.join(out_dir, f"{tag}_adj_dense_sample.npy"), sample_dense)

    print("Saved raw adjacency.")

def save_all_layer_spmm(adj, u_emb, i_emb, n_layers, out_dir, tag):
    csr = adj.tocsr()

    # 初始 embedding
    E = np.vstack([u_emb, i_emb])

    np.save(os.path.join(out_dir, f"{tag}_layer0.npy"), E.astype(np.float32))

    print("Saved layer 0")

    for k in range(1, n_layers + 1):
        E = csr.dot(E)   # SpMM
        np.save(
            os.path.join(out_dir, f"{tag}_layer{k}.npy"),
            E.astype(np.float32)
        )
        print(f"Saved layer {k}")


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    out_dir = os.path.abspath(f"./output/sparse_out_{args.dataset}")
    os.makedirs(out_dir, exist_ok=True)

    # 1) sparse stats + plots (paper-level)
    adj, adj_tag = pick_adj()
    tag = f"{args.dataset}_{adj_tag}"
    torch_adj = scipy_to_torch_sparse(adj)
    save_raw_adjacency(adj, out_dir, tag)

    collect_sparse_graph_stats(torch_adj, out_dir=out_dir, tag=tag)
    deg = np.load(os.path.join(out_dir, f"{tag}_degree.npz"))
    render_all_degree_plots(deg["row_nnz"], deg["col_nnz"], out_dir=os.path.join(out_dir, "figs"), tag=tag)

    # bipartite split plots (amazon-book特别有用)
    n_users, n_items = data_generator.n_users, data_generator.n_items
    row_nnz = deg["row_nnz"]
    user_deg = row_nnz[:n_users]
    item_deg = row_nnz[n_users:n_users + n_items]
    render_all_degree_plots(user_deg, item_deg, out_dir=os.path.join(out_dir, "figs_bipartite"),
                            tag=f"{tag}_user_vs_item")

    # 2) inference demo: compute embeddings once, output topK for 10 users
    config = {"n_users": n_users, "n_items": n_items, "norm_adj": adj}
    model = LightGCN(data_config=config, pretrain_data=None)

    tf_cfg = tf.ConfigProto()
    tf_cfg.gpu_options.allow_growth = True
    with tf.Session(config=tf_cfg) as sess:
        sess.run(tf.global_variables_initializer())
        u_emb, i_emb = sess.run([model.ua_embeddings, model.ia_embeddings])  # (U,D), (I,D)

    n_layers = len(eval(args.layer_size))   # 或直接 args.n_layers
    save_all_layer_spmm(adj,u_emb,i_emb,n_layers,out_dir,tag)

    topk = int(eval(args.Ks)[0]) if hasattr(args, "Ks") else 20
    users = list(getattr(data_generator, "test_set", {}).keys())[:10] or list(range(10))

    recs = {}
    iT = i_emb.T
    for u in users:
        scores = u_emb[u] @ iT
        # filter training items if available
        train_items = getattr(data_generator, "train_items", {}).get(u, [])
        if len(train_items) > 0:
            scores[np.array(train_items, dtype=np.int64)] = -np.inf
        top = np.argpartition(-scores, kth=min(topk, scores.size - 1))[:topk]
        top = top[np.argsort(-scores[top])]
        recs[int(u)] = [int(x) for x in top]

    with open(os.path.join(out_dir, f"{args.dataset}_top{topk}_demo.json"), "w", encoding="utf-8") as f:
        json.dump(recs, f, indent=2, ensure_ascii=False)

    print("DONE. outputs:", out_dir)


if __name__ == "__main__":
    main()