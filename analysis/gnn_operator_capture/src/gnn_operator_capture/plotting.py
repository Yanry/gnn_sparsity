from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .io_utils import ensure_dir


def maybe_plot_tensor(tensor: torch.Tensor, base_path: Path) -> None:
    tensor = tensor.detach().cpu()
    ensure_dir(base_path.parent)
    dense = tensor.to_dense() if tensor.layout != torch.strided else tensor
    values = dense.reshape(-1).to(torch.float32)

    fig = plt.figure(figsize=(6, 4))
    plt.hist(values.numpy(), bins=64)
    plt.title("Value Histogram")
    plt.tight_layout()
    fig.savefig(base_path.with_suffix(".hist.png"))
    plt.close(fig)

    matrix = dense
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    elif matrix.ndim > 2:
        matrix = matrix.reshape(-1, matrix.shape[-1])

    if matrix.numel() <= 4096:
        fig = plt.figure(figsize=(5, 4))
        plt.imshow((matrix != 0).numpy(), aspect="auto", cmap="Greys")
        plt.title("Sparsity Pattern")
        plt.tight_layout()
        fig.savefig(base_path.with_suffix(".heatmap.png"))
        plt.close(fig)
