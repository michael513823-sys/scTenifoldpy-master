from typing import List
from warnings import warn

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from scTenifold.core._networks import make_networks

# Optional CUDA path
try:
    from scTenifold.core._ko_cuda import (
        ko_propagation_cuda,
        reconstruct_pcnets_cuda,
        CUDA_AVAILABLE as _CUDA_AVAILABLE,
    )
except ImportError:  # pragma: no cover
    ko_propagation_cuda = None    # type: ignore
    reconstruct_pcnets_cuda = None  # type: ignore
    _CUDA_AVAILABLE = False


def ko_propagation(B, x, ko_gene_id, degree: int) -> np.ndarray:
    adj_mat = B.copy()
    np.fill_diagonal(adj_mat, 0)
    x_ko = x.copy()
    p = np.zeros(shape=x.shape)
    p[ko_gene_id, :] = x[ko_gene_id, :]
    perturbs = [p]
    is_visited = np.array([False for _ in range(x_ko.shape[0])])
    for d in range(degree):
        if not is_visited.all():
            perturbs.append(adj_mat @ perturbs[d])
            new_visited = (perturbs[d+1] != 0).any(axis=1)
            adj_mat[is_visited, :] = 0
            adj_mat[:, is_visited] = 0
            is_visited = is_visited | new_visited

    for p in perturbs:
        x_ko = x_ko - p
    return np.where(x_ko >= 0, x_ko, 0)


def reconstruct_pcnets(nets: List[coo_matrix],
                       X_df,
                       ko_gene_id,
                       degree,
                       use_cuda: bool = False,
                       device: str = "cuda:0",
                       **kwargs):
    """Reconstruct PCNets after virtual knock-out.

    Parameters
    ----------
    nets : list of coo_matrix
        Original WT PCNets.
    X_df : pd.DataFrame
        Expression matrix (genes x cells).
    ko_gene_id : list of int
        Indices of knocked-out genes.
    degree : int
        Propagation hops.
    use_cuda : bool, default False
        If True and a CUDA backend is available, delegate propagation and
        PCNet reconstruction to GPU via :func:`reconstruct_pcnets_cuda`.
    device : str, default "cuda:0"
        CUDA device string (PyTorch backend only).
    **kwargs
        Forwarded to :func:`make_networks` / :func:`make_networks_cuda`.
    """
    # ------------------------------------------------------------------ #
    # CUDA fast-path                                                       #
    # ------------------------------------------------------------------ #
    if use_cuda:
        if reconstruct_pcnets_cuda is None or not _CUDA_AVAILABLE:
            warn(
                "use_cuda=True requested but no CUDA backend available. "
                "Falling back to CPU reconstruction.",
                RuntimeWarning,
            )
        else:
            return reconstruct_pcnets_cuda(
                nets, X_df, ko_gene_id, degree,
                device=device, **kwargs
            )

    # ------------------------------------------------------------------ #
    # CPU path (original implementation)                                  #
    # ------------------------------------------------------------------ #
    ko_nets = []
    for net in nets:
        data = ko_propagation(net.toarray(), X_df.values, ko_gene_id, degree)
        data = pd.DataFrame(data, index=X_df.index, columns=X_df.columns)
        ko_net = make_networks(data, n_nets=1, **kwargs)[0]
        ko_nets.append(ko_net)
    return ko_nets