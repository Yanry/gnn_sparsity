import os
import numpy as np
import matplotlib.pyplot as plt

from .sparse_metrics import ccdf, lorenz_curve

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def log_hist(x, bins=50):
    x = np.asarray(x).astype(np.int64)
    x = x[x > 0]
    if x.size == 0:
        return np.array([1,2]), np.array([0.0])
    xmin, xmax = x.min(), x.max()
    # log spaced bins
    edges = np.unique(np.logspace(np.log10(xmin), np.log10(xmax), bins).astype(np.int64))
    if edges.size < 2:
        edges = np.array([xmin, xmax+1])
    hist, edges = np.histogram(x, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, hist

def plot_degree_hist_logbins(x, title, out_path):
    centers, hist = log_hist(x, bins=80)
    plt.figure()
    plt.title(title)
    plt.xlabel("degree (log bins)")
    plt.ylabel("count")
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(centers, hist, marker="o", linewidth=1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_ccdf(x, title, out_path):
    sx, y = ccdf(x)
    if sx.size == 0:
        return
    plt.figure()
    plt.title(title)
    plt.xlabel("degree")
    plt.ylabel("P(X >= degree)")
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(sx, y, linewidth=1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_lorenz(x, title, out_path):
    p, cum = lorenz_curve(x)
    plt.figure()
    plt.title(title)
    plt.xlabel("cumulative fraction of nodes")
    plt.ylabel("cumulative fraction of edges")
    plt.plot(p, cum, linewidth=2, label="Lorenz")
    plt.plot([0,1],[0,1], linestyle="--", linewidth=1, label="Equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_topk_cumshare(x, title, out_path, max_k=5000):
    x = np.asarray(x).astype(np.float64)
    x = np.clip(x, 0, None)
    if x.sum() <= 0:
        return
    xs = np.sort(x)[::-1]
    k = min(max_k, xs.size)
    cum = np.cumsum(xs[:k]) / xs.sum()
    plt.figure()
    plt.title(title)
    plt.xlabel("top-k nodes")
    plt.ylabel("cumulative share of edges")
    plt.plot(np.arange(1, k+1), cum, linewidth=2)
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def render_all_degree_plots(row_nnz, col_nnz, out_dir, tag):
    ensure_dir(out_dir)
    plot_degree_hist_logbins(row_nnz, f"{tag} row degree hist (log-log)", os.path.join(out_dir, f"{tag}_row_hist.png"))
    plot_degree_hist_logbins(col_nnz, f"{tag} col degree hist (log-log)", os.path.join(out_dir, f"{tag}_col_hist.png"))

    plot_ccdf(row_nnz, f"{tag} row degree CCDF", os.path.join(out_dir, f"{tag}_row_ccdf.png"))
    plot_ccdf(col_nnz, f"{tag} col degree CCDF", os.path.join(out_dir, f"{tag}_col_ccdf.png"))

    plot_lorenz(row_nnz, f"{tag} row degree Lorenz", os.path.join(out_dir, f"{tag}_row_lorenz.png"))
    plot_lorenz(col_nnz, f"{tag} col degree Lorenz", os.path.join(out_dir, f"{tag}_col_lorenz.png"))

    plot_topk_cumshare(row_nnz, f"{tag} row top-k cumulative share", os.path.join(out_dir, f"{tag}_row_topk.png"))
    plot_topk_cumshare(col_nnz, f"{tag} col top-k cumulative share", os.path.join(out_dir, f"{tag}_col_topk.png"))