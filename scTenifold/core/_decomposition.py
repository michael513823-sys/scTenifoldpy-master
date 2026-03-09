from typing import Sequence
from warnings import warn

import numpy as np
import pandas as pd
import scipy
from scTenifold.core._utils import timer
from tensorly.decomposition import parafac, parafac2, parafac_power_iteration
from tensorly import decomposition
import tensorly as tl

# Optional CUDA path
try:
    from scTenifold.core._decomposition_cuda import (
        tensor_decomp_cuda,
        CUDA_AVAILABLE as _CUDA_AVAILABLE,
    )
except ImportError:  # pragma: no cover
    tensor_decomp_cuda = None  # type: ignore
    _CUDA_AVAILABLE = False

__all__ = ["tensor_decomp"]


@timer
def tensor_decomp(networks: np.ndarray,
                  gene_names: Sequence[str],
                  method: str = "parafac",
                  n_decimal: int = 1,
                  K: int = 5,
                  tol: float = 1e-6,
                  max_iter: int = 1000,
                  random_state: int = 42,
                  use_cuda: bool = False,
                  device: str = "cuda:0",
                  **kwargs) -> pd.DataFrame:
    """
    Perform tensor decomposition on pc networks

    Parameters
    ----------
    networks: np.ndarray
        Concatenated network, expected shape = (n_genes, n_genes, n_pcnets)
    gene_names: sequence of str
        The name of each gene in the network (order matters)
    method: str, default = 'parafac'
        Tensor decomposition method, tensorly's decomposition method was used:
        http://tensorly.org/stable/modules/api.html#module-tensorly.decomposition
    n_decimal: int
        Number of decimal in the final df
    K: int
        Rank in parafac function
    tol: float
        Tolerance in the iteration
    max_iter: int
        Number of interation
    random_state: int
        Random seed used to reproduce the same result
    use_cuda: bool, default False
        If True, run the decomposition on GPU via :func:`tensor_decomp_cuda`.
        Requires cupy or a CUDA-enabled torch installation.
    device : str, default "cuda:0"
        CUDA device string (PyTorch backend only).
    **kwargs:
        Keyword arguments used in the decomposition function

    Returns
    -------
    tensor_decomp_df: pd.DataFrame
        The result of tensor decomposition, expected shape = (n_genes, n_genes)

    References
    ----------
    http://tensorly.org/stable/modules/api.html#module-tensorly.decomposition

    """
    # ------------------------------------------------------------------ #
    # CUDA fast-path                                                       #
    # ------------------------------------------------------------------ #
    if use_cuda:
        if tensor_decomp_cuda is None or not _CUDA_AVAILABLE:
            warn(
                "use_cuda=True requested but no CUDA backend is available. "
                "Falling back to CPU tensor decomposition.",
                RuntimeWarning,
            )
        else:
            return tensor_decomp_cuda(
                networks, gene_names,
                method=method, n_decimal=n_decimal, K=K,
                tol=tol, max_iter=max_iter,
                random_state=random_state,
                device=device,
                **kwargs,
            )
    # Us, est, res_hist = cp_als(networks, n_components=K, max_iter=max_iter, tol=tol)
    # print(est.shape, len(Us), res_hist)
    print("Using tensorly")
    factors = getattr(decomposition, method)(networks, rank=K, n_iter_max=max_iter, tol=tol,
                                             random_state=random_state, **kwargs)
    estimate = tl.cp_to_tensor(factors)
    print(estimate.shape)
    out = np.sum(estimate, axis=-1) / len(networks)
    out = np.round(out / np.max(abs(out)), n_decimal)
    return pd.DataFrame(out, index=gene_names, columns=gene_names)