import numpy as np

def safe_div(a, b, eps=1e-12):
    return float(a) / float(b + eps)

def gini(x: np.ndarray) -> float:
    """Gini 系数：0均匀，1极度不均匀"""
    x = x.astype(np.float64)
    if x.size == 0:
        return 0.0
    x = np.clip(x, 0, None)
    s = x.sum()
    if s <= 0:
        return 0.0
    x_sorted = np.sort(x)
    n = x_sorted.size
    cum = np.cumsum(x_sorted)
    # 标准离散 Gini
    return float((n + 1 - 2 * np.sum(cum) / s) / n)

def coeff_var(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    m = x.mean() if x.size else 0.0
    if m <= 0:
        return 0.0
    return float(x.std(ddof=0) / m)

def topk_share(x: np.ndarray, ks=(1, 10, 100, 1000), assume_sorted=False):
    """Top-k 贡献占比：sum(topk)/sum(all)"""
    x = x.astype(np.float64)
    s = x.sum()
    if s <= 0:
        return {f"top{k}_share": 0.0 for k in ks}
    if not assume_sorted:
        x = np.sort(x)[::-1]
    out = {}
    for k in ks:
        kk = min(int(k), x.size)
        out[f"top{int(k)}_share"] = float(x[:kk].sum() / s)
    return out

def ccdf(x: np.ndarray):
    """返回 (sorted_x, ccdf_y)"""
    x = np.asarray(x).astype(np.int64)
    x = x[x >= 0]
    if x.size == 0:
        return np.array([]), np.array([])
    sx = np.sort(x)
    # ccdf(y) = P(X >= value)
    # 对每个排序位置 i，sx[i] 的 ccdf = (n-i)/n
    n = sx.size
    y = (n - np.arange(n)) / n
    return sx, y

def lorenz_curve(x: np.ndarray):
    """Lorenz 曲线：用于可视化不均匀性"""
    x = np.asarray(x).astype(np.float64)
    x = np.clip(x, 0, None)
    s = x.sum()
    if s <= 0 or x.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    xs = np.sort(x)
    cum = np.cumsum(xs) / s
    p = np.arange(1, xs.size + 1) / xs.size
    # 加上 (0,0)
    p = np.concatenate([[0.0], p])
    cum = np.concatenate([[0.0], cum])
    return p, cum