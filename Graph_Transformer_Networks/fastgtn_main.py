import torch
import torch.nn.functional as F
import numpy as np
import argparse
import copy
import pickle
import scipy.sparse as sp
from model_fastgtn import FastGTNs, FastGTN, FastGTLayer  # 导入FastGTN模型
from utils import init_seed, add_self_loops, _norm  # 导入工具函数
import os, sys
import json

THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../project/LightGCN
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))   # .../project
sys.path.insert(0, PROJECT_ROOT)                                # 让 hooks 可被 import
sys.path.insert(0, THIS_DIR)                                    # 让 utility/LightGCN.py 可被 import（稳）

def print_model_params(model):
    # 打印每个模块的名称和它的参数
    for name, module in model.named_modules():
        print(f"Layer: {name}")
        for param_name, param in module.named_parameters():
            print(f"  Param: {param_name} - Shape: {param.shape}")

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
    n = min(sample_size, adj.shape[0])
    sample_dense = adj[:n, :n].toarray()
    np.save(os.path.join(out_dir, f"{tag}_adj_dense_sample.npy"), sample_dense)

    print("Saved raw adjacency.")

def register_non_local_x_hooks(model, out_dir, run_tag="run0", save_fp16=True):
    """
    在不修改 FastGTN 的情况下，hook 每个 FastGTN.feat_trans_layers[i]
    保存 generate_non_local_graph 里的 x = relu(feat_trans(H)).
    """
    os.makedirs(out_dir, exist_ok=True)
    hooks = []

    # model 是 FastGTNs，里面有 model.fastGTNs: ModuleList[FastGTN]
    for gtn_idx, fastgtn in enumerate(model.fastGTNs):
        # fastgtn.feat_trans_layers: ModuleList，长度 = num_layers + 1（代码里这么建的）
        for layer_i, feat_trans in enumerate(fastgtn.feat_trans_layers):

            def _make_hook(gtn_idx_, layer_i_):
                def _hook(mod, inputs, output):
                    # inputs[0] = H（传给 feat_trans 的输入）
                    # output = feat_trans(H)（还没 relu）
                    x = F.relu(output)

                    x_save = x.detach()
                    if save_fp16:
                        x_save = x_save.to(torch.float16)

                    # 文件名里明确标注：第几个 FastGTN block + 第几层 i
                    path = os.path.join(
                        out_dir,
                        f"{run_tag}_fastgtn{gtn_idx_}_layer{layer_i_}_x.npy"
                    )
                    np.save(path, x_save.cpu().numpy())
                return _hook

            hooks.append(feat_trans.register_forward_hook(_make_hook(gtn_idx, layer_i)))

    return hooks

# Hook函数：捕获每层的meta-path邻接矩阵
def capture_x_hook(module, input, output):
    """
    捕获每一层 `generate_non_local_graph` 中的 `x` 和 `H`
    """
    print(f"Inside hook for layer {module.layer_id}")
    print(input)
    H = input[1]  # 假设 H 是第二个输入参数：节点特征矩阵
    x = F.relu(module.feat_trans_layers[0](H))  # 假设 feat_trans_layers 是每层的特征变换

    # 打印或保存 `x`
    print(f"Captured x (feature transformation) in layer {module.layer_id}, shape: {x.shape}")
    
    # 保存 `x`，保存为 .npy 文件
    np.save(f"output_directory/layer_{module.layer_id}_x.npy", x.cpu().detach().numpy())

    return output

# 注册Hook函数
def register_hooks(model):
    hooks = []
    for name, module in model.named_modules():
        # 确保在包含 `feat_trans` 的层（即 `FastGTN`）注册 hook
        if isinstance(module, FastGTN):  # FastGTN 层中有 `feat_trans`
            hook = module.register_forward_hook(capture_x_hook)
            hooks.append(hook)
    return hooks

def main():
    init_seed(seed=777)
    parser = argparse.ArgumentParser()
    # 添加FastGTN的相关参数
    parser.add_argument('--dataset', type=str, required=True, help='Dataset')
    parser.add_argument('--node_dim', type=int, default=64, help='hidden dimensions')
    parser.add_argument('--num_channels', type=int, default=2, help='number of channels')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--num_layers', type=int, default=3, help='number of layers in FastGTN')
    parser.add_argument('--runs', type=int, default=10, help='number of runs')
    parser.add_argument('--pre_train', action='store_true', help='whether to pre-train the FastGT layers')
    parser.add_argument('--num_FastGTN_layers', type=int, default=1, help='number of FastGTN layers')
    parser.add_argument("--channel_agg", type=str, default='concat')
    # Configurations for FastGTNs
    parser.add_argument("--non_local", action='store_true', help="use non local operations")
    parser.add_argument("--non_local_weight", type=float, default=0, help="weight initialization for non local operations")
    parser.add_argument("--beta", type=float, default=0, help="beta (Identity matrix)")
    parser.add_argument('--K', type=int, default=1,
                        help='number of non-local negibors')
    # parser.add_argument("--pre_train", action='store_true', help="pre-training FastGT layers")

    args = parser.parse_args()
    print(args)

    # 数据加载部分
    with open(f'./data/{args.dataset}/node_features.pkl', 'rb') as f:
        node_features = pickle.load(f)
    with open(f'./data/{args.dataset}/edges.pkl', 'rb') as f:
        edges = pickle.load(f)
        edges = [sp.csr_matrix(edge) if not isinstance(edge, sp.csr_matrix) else edge for edge in edges]
    with open(f'./data/{args.dataset}/labels.pkl', 'rb') as f:
        labels = pickle.load(f)

    num_nodes = edges[0].shape[0]
    args.num_nodes = num_nodes

    # 构建邻接矩阵
    A = []
    for i, edge in enumerate(edges):
        edge_tmp = torch.from_numpy(np.vstack((edge.nonzero()[1], edge.nonzero()[0]))).type(torch.LongTensor)
        value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.FloatTensor)
        
        # 处理FastGTN中的邻接矩阵
        edge_tmp, value_tmp = add_self_loops(edge_tmp, edge_attr=value_tmp, fill_value=1e-20, num_nodes=num_nodes)
        deg_inv_sqrt, deg_row, deg_col = _norm(edge_tmp.detach(), num_nodes, value_tmp.detach())
        value_tmp = deg_inv_sqrt[deg_row] * value_tmp
        A.append((edge_tmp, value_tmp))
    
    # 添加自环
    edge_tmp = torch.stack((torch.arange(0, num_nodes), torch.arange(0, num_nodes))).type(torch.LongTensor)
    value_tmp = torch.ones(num_nodes).type(torch.FloatTensor)
    A.append((edge_tmp, value_tmp))

    num_edge_type = len(A)
    node_features = torch.from_numpy(node_features).type(torch.FloatTensor)

    # 根据任务加载训练集/验证集/测试集
    train_node = torch.from_numpy(np.array(labels[0])[:, 0]).type(torch.LongTensor)
    train_target = torch.from_numpy(np.array(labels[0])[:, 1]).type(torch.LongTensor)
    valid_node = torch.from_numpy(np.array(labels[1])[:, 0]).type(torch.LongTensor)
    valid_target = torch.from_numpy(np.array(labels[1])[:, 1]).type(torch.LongTensor)
    test_node = torch.from_numpy(np.array(labels[2])[:, 0]).type(torch.LongTensor)
    test_target = torch.from_numpy(np.array(labels[2])[:, 1]).type(torch.LongTensor)
    
    num_classes = np.max([torch.max(train_target).item(), torch.max(valid_target).item(), torch.max(test_target).item()]) + 1
    
    # 初始化FastGTN模型
    model = FastGTNs(num_edge_type=num_edge_type, 
                     w_in=node_features.shape[1], 
                     num_class=num_classes, 
                     num_nodes=num_nodes, 
                     args=args)
    # print_model_params(model)
    
    # 注册 hook 来捕获每层的邻接矩阵
    # hooks = register_hooks(model)
    out_dir = f"./output/{args.dataset}_fastgtn_hooks"
    hooks = register_non_local_x_hooks(model, out_dir, run_tag="infer", save_fp16=True)

    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)

    # 训练和推理过程，只做推理
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 将模型和数据移动到相同设备上
    model = model.to(device)
    node_features = node_features.to(device)
    A = [(edge.to(device), value.to(device)) for edge, value in A]
    test_node = test_node.to(device)
    test_target = test_target.to(device)
    model.eval()
    
    with torch.no_grad():
        output = model(A, node_features, test_node, test_target)
        print("Inference completed. Saved meta-path adjacency matrices for each layer.")

    for h in hooks:
            h.remove()

    print("Done.")
    

if __name__ == "__main__":
    main()