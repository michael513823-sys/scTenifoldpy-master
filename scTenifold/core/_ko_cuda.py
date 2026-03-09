"""
CUDA-accelerated virtual knock-out.

Two improvements over the original CPU implementation
------------------------------------------------------
1. **ko_propagation_cuda** – replaces the per-step ``adj_mat @ perturbs[d]``
   NumPy matmul with a CuPy / PyTorch matmul so the propagation loop runs
   entirely on the GPU.  For large gene panels the matrix-vector product is
   the hot inner loop.

2. **reconstruct_pcnets_cuda** – processes the list of ``nets`` in *batches*:
   all ``ko_propagation`` calls happen sequentially on GPU (each is fast), and
   the downstream ``make_networks`` call uses :func:`make_networks_cuda` so
   that PCNet reconstruction also runs on GPU.

Requires
--------
* cupy or torch (same rules as ``_networks_cuda.py``)
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from scTenifold.core._networks_cuda import make_networks_cuda, CUDA_AVAILABLE, _check_cuda

__all__ = ["ko_propagation_cuda", "reconstruct_pcnets_cuda"]

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

_cupy_available = False
try:
    import cupy as cp
    _cupy_available = True
except ImportError:
    cp = None  # type: ignore

_torch_available = False
try:
    import torch
    _torch_available = torch.cuda.is_available()
except ImportError:
    torch = None  # type: ignore


# ---------------------------------------------------------------------------
# GPU ko_propagation
# ---------------------------------------------------------------------------

def ko_propagation_cuda(
    B: np.ndarray,
    x: np.ndarray,
    ko_gene_id: List[int],
    degree: int,
    device: str = "cuda:0",
) -> np.ndarray:
    """
    GPU-accelerated perturbation propagation.

    Identical semantics to :func:`scTenifold.core._ko.ko_propagation` but
    the adjacency-matrix × perturbation-vector products are executed on the
    GPU.

    Parameters
    ----------
    B : np.ndarray  shape (n_genes, n_genes)
        Adjacency matrix (host).
    x : np.ndarray  shape (n_genes, n_cells)  or  (n_genes,)
        Expression matrix / vector (host).
    ko_gene_id : list of int
        Indices of the knocked-out genes.
    degree : int
        Number of propagation hops.
    device : str
        CUDA device string (used only for the PyTorch path).

    Returns
    -------
    x_ko : np.ndarray  shape == x.shape  (host)
    """
    _check_cuda()

    if _cupy_available:
        xp = cp
        adj_mat = cp.asarray(B.copy(), dtype=cp.float64)
        x_ko_gpu = cp.asarray(x.copy(), dtype=cp.float64)
        p = cp.zeros_like(x_ko_gpu)
        p[ko_gene_id, :] = x_ko_gpu[ko_gene_id, :]

        perturbs = [p]
        n_genes = x_ko_gpu.shape[0]
        is_visited = cp.zeros(n_genes, dtype=bool)
        cp.fill_diagonal(adj_mat, 0.0)

        for d in range(degree):
            if is_visited.all():
                break
            next_p = adj_mat @ perturbs[d]
            perturbs.append(next_p)
            new_visited = (perturbs[d + 1] != 0).any(axis=1) if perturbs[d + 1].ndim > 1 \
                          else (perturbs[d + 1] != 0)
            adj_mat[is_visited, :] = 0.0
            adj_mat[:, is_visited] = 0.0
            is_visited = is_visited | new_visited

        for p in perturbs:
            x_ko_gpu -= p

        result_gpu = cp.where(x_ko_gpu >= 0, x_ko_gpu, 0)
        return cp.asnumpy(result_gpu)

    else:
        # PyTorch path
        dev = torch.device(device)
        adj_mat = torch.as_tensor(B.copy(), dtype=torch.float64, device=dev)
        x_ko_gpu = torch.as_tensor(x.copy(), dtype=torch.float64, device=dev)

        p = torch.zeros_like(x_ko_gpu)
        p[ko_gene_id, :] = x_ko_gpu[ko_gene_id, :]

        perturbs = [p]
        n_genes = x_ko_gpu.shape[0]
        is_visited = torch.zeros(n_genes, dtype=torch.bool, device=dev)
        adj_mat.fill_diagonal_(0.0)

        for d in range(degree):
            if is_visited.all():
                break
            next_p = adj_mat @ perturbs[d]
            perturbs.append(next_p)
            new_visited = (perturbs[d + 1] != 0).any(dim=1) if perturbs[d + 1].ndim > 1 \
                          else (perturbs[d + 1] != 0)
            adj_mat[is_visited, :] = 0.0
            adj_mat[:, is_visited] = 0.0
            is_visited = is_visited | new_visited

        for p in perturbs:
            x_ko_gpu -= p

        result_gpu = torch.where(x_ko_gpu >= 0, x_ko_gpu,
                                 torch.zeros_like(x_ko_gpu))
        return result_gpu.cpu().numpy()


# ---------------------------------------------------------------------------
# reconstruct_pcnets_cuda
# ---------------------------------------------------------------------------

def reconstruct_pcnets_cuda(
    nets: List[coo_matrix],
    X_df: pd.DataFrame,
    ko_gene_id: List[int],
    degree: int,
    device: str = "cuda:0",
    **kwargs,
) -> List[coo_matrix]:
    """
    GPU-accelerated KO-network reconstruction.

    For each network in ``nets``:

    1. Run :func:`ko_propagation_cuda` on GPU (matmul hot-path on device).
    2. Call :func:`make_networks_cuda` for the post-KO PCNet construction
       (SVD + beta-matrix also on GPU).

    Parameters
    ----------
    nets : list of coo_matrix
        Original (WT) PCNets.
    X_df : pd.DataFrame
        Gene-expression matrix used for PCNet reconstruction.
    ko_gene_id : list of int
        Indices of knocked-out genes.
    degree : int
        Propagation depth.
    device : str
        CUDA device identifier.
    **kwargs
        Forwarded to :func:`make_networks_cuda`.

    Returns
    -------
    ko_nets : list of coo_matrix
    """
    _check_cuda()
    ko_nets = []
    x_vals = X_df.values  # keep host copy; propagation uploads per-net

    for net in nets:
        # --- GPU propagation ---
        x_ko = ko_propagation_cuda(
            net.toarray(), x_vals, ko_gene_id, degree, device=device
        )
        # --- Rebuild expression DataFrame ---
        data_ko = pd.DataFrame(x_ko, index=X_df.index, columns=X_df.columns)
        # --- GPU PCNet construction ---
        ko_net = make_networks_cuda(data_ko, n_nets=1, device=device, **kwargs)[0]
        ko_nets.append(ko_net)

    return ko_nets
