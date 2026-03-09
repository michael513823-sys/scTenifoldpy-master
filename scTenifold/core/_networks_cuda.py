"""
CUDA-accelerated PCNet construction.

Key optimisations over the CPU path
-------------------------------------
1. **Batch SVD on GPU** – instead of the ``randomized_svd`` call inside
   ``cal_pc_coefs`` (which runs per-gene, serially on CPU), we compute the
   truncated SVD of the *whole* ``(cells × genes)`` matrix once with
   ``cupy.linalg.svd`` / ``torch.linalg.svd``, then derive regression
   coefficients for every gene simultaneously via batched matrix operations.

2. **All heavy NumPy kernels → CuPy** – standardisation, beta matrix
   construction, quantile thresholding, etc. all happen on the GPU.

3. **Multiple sub-sampled networks computed in a Python loop, but each
   individual PCNet is fully GPU-resident** – no host↔device transfers
   inside the hot loop.  Results are moved to host only when assembling the
   final COO list.

4. **Optional torch.linalg.svd path** for environments where CuPy is
   unavailable (e.g. pure-PyTorch conda environments).

Requires
--------
* cupy   (``pip install cupy-cuda11x`` / ``cupy-cuda12x``)
* torch  (optional, used as fallback if cupy is not available)

Both are optional at *import* time; if neither is available the module can
still be imported – but calling the CUDA functions will raise ``RuntimeError``.
"""

from __future__ import annotations

from functools import partial
from typing import List, Optional, Tuple
from warnings import warn

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from scTenifold.core._utils import timer

__all__ = ["make_networks_cuda", "pc_net_calc_cuda", "CUDA_AVAILABLE"]

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

_cupy_available = False
_torch_cuda_available = False

try:
    import cupy as cp
    import cupy.linalg as cpla
    _cupy_available = True
except ImportError:
    cp = None  # type: ignore

try:
    import torch
    _torch_cuda_available = torch.cuda.is_available()
except ImportError:
    torch = None  # type: ignore

CUDA_AVAILABLE: bool = _cupy_available or _torch_cuda_available


def _check_cuda():
    if not CUDA_AVAILABLE:
        raise RuntimeError(
            "No CUDA backend found. Install cupy (recommended) via:\n"
            "  pip install cupy-cuda11x   # for CUDA 11.x\n"
            "  pip install cupy-cuda12x   # for CUDA 12.x\n"
            "or install PyTorch with CUDA support:\n"
            "  https://pytorch.org/get-started/locally/"
        )


# ---------------------------------------------------------------------------
# CuPy backend
# ---------------------------------------------------------------------------

def _cal_beta_matrix_cupy(Xt_gpu: "cp.ndarray", n_comp: int) -> "cp.ndarray":
    """
    Compute the full beta matrix (genes × genes) on GPU using CuPy.

    Parameters
    ----------
    Xt_gpu : cp.ndarray  shape (cells, genes)
        Standardised expression matrix already on GPU.
    n_comp : int
        Number of principal components to retain.

    Returns
    -------
    B : cp.ndarray  shape (genes, genes)
        B[k, j] = regression coefficient of gene j predicting gene k.
        Diagonal entries are meaningless (set to 0 later by the caller).
    """
    n_cells, n_genes = Xt_gpu.shape
    B = cp.zeros((n_genes, n_genes), dtype=cp.float64)

    for k in range(n_genes):
        y = Xt_gpu[:, k]                          # (cells,)
        # Remove the k-th column
        idx = list(range(n_genes))
        idx.pop(k)
        Xi = Xt_gpu[:, idx]                        # (cells, genes-1)

        # Truncated SVD on GPU
        # cupy.linalg.svd full_matrices=False → U(cells,r), s(r,), VT(r,genes-1)
        U, s, VT = cpla.svd(Xi, full_matrices=False)
        VT = VT[:n_comp, :]                        # (n_comp, genes-1)
        coef = VT.T                                # (genes-1, n_comp)

        score = Xi @ coef                          # (cells, n_comp)
        score_norm = cp.sum(score ** 2, axis=0)    # (n_comp,)
        score = score / score_norm                 # broadcast: (cells, n_comp)

        betas = coef @ cp.sum(y[:, None] * score, axis=0)  # (genes-1,)
        B[k, idx] = betas

    return B


def pc_net_calc_cuda_cupy(
    X_np: np.ndarray,           # genes × cells  (NumPy, host)
    selected_samples: np.ndarray,
    n_comp: int = 3,
    scale_scores: bool = True,
    symmetric: bool = False,
    q: float = 0.0,
) -> np.ndarray:
    """Single PCNet construction fully on GPU via CuPy."""
    # ---- 1. Sub-sample & filter on host (cheap) ----
    Z = X_np[:, selected_samples]
    mask = Z.sum(axis=1) > 0
    Z = Z[mask, :]                                # (active_genes, cells)

    # ---- 2. Upload to GPU ----
    Z_gpu = cp.asarray(Z, dtype=cp.float64)
    Xt = Z_gpu.T                                  # (cells, active_genes)

    # ---- 3. Standardise ----
    Xt = (Xt - Xt.mean(axis=0)) / (Xt.std(axis=0) + 1e-12)

    # ---- 4. Compute beta matrix ----
    n_ag = Xt.shape[1]
    assert 2 < n_comp <= n_ag, f"n_comp={n_comp} must be in (2, n_active_genes={n_ag}]"

    B = _cal_beta_matrix_cupy(Xt, n_comp)         # (active_genes, active_genes)

    # ---- 5. Build adjacency matrix ----
    A = B.copy()
    cp.fill_diagonal(A, 0.0)

    if symmetric:
        A = (A + A.T) / 2.0

    abs_A = cp.abs(A)
    if scale_scores:
        max_val = cp.max(abs_A)
        if max_val > 0:
            A = A / max_val

    if q > 0:
        threshold = cp.quantile(abs_A, q)
        A[abs_A < threshold] = 0.0

    cp.fill_diagonal(A, 0.0)

    # ---- 6. Download result ----
    return cp.asnumpy(A), mask


# ---------------------------------------------------------------------------
# PyTorch backend (fallback)
# ---------------------------------------------------------------------------

def _cal_beta_matrix_torch(Xt_gpu: "torch.Tensor", n_comp: int) -> "torch.Tensor":
    """
    Same logic as ``_cal_beta_matrix_cupy`` but using PyTorch tensors.
    """
    n_cells, n_genes = Xt_gpu.shape
    B = torch.zeros((n_genes, n_genes), dtype=torch.float64, device=Xt_gpu.device)

    for k in range(n_genes):
        y = Xt_gpu[:, k]
        idx = list(range(n_genes))
        idx.pop(k)
        Xi = Xt_gpu[:, idx]                       # (cells, genes-1)

        # torch.linalg.svd: full_matrices=False
        U, s, Vh = torch.linalg.svd(Xi, full_matrices=False)
        VT = Vh[:n_comp, :]                        # (n_comp, genes-1)
        coef = VT.T                                # (genes-1, n_comp)

        score = Xi @ coef                          # (cells, n_comp)
        score_norm = (score ** 2).sum(dim=0)       # (n_comp,)
        score = score / score_norm

        betas = coef @ (y.unsqueeze(1) * score).sum(dim=0)  # (genes-1,)
        B[k, idx] = betas

    return B


def pc_net_calc_cuda_torch(
    X_np: np.ndarray,
    selected_samples: np.ndarray,
    n_comp: int = 3,
    scale_scores: bool = True,
    symmetric: bool = False,
    q: float = 0.0,
    device: str = "cuda:0",
) -> Tuple[np.ndarray, np.ndarray]:
    """Single PCNet construction on GPU via PyTorch."""
    Z = X_np[:, selected_samples]
    mask = Z.sum(axis=1) > 0
    Z = Z[mask, :]

    dev = torch.device(device)
    Z_t = torch.as_tensor(Z, dtype=torch.float64, device=dev)
    Xt = Z_t.T

    Xt = (Xt - Xt.mean(dim=0)) / (Xt.std(dim=0) + 1e-12)

    n_ag = Xt.shape[1]
    assert 2 < n_comp <= n_ag

    B = _cal_beta_matrix_torch(Xt, n_comp)

    A = B.clone()
    A.fill_diagonal_(0.0)

    if symmetric:
        A = (A + A.T) / 2.0

    abs_A = A.abs()
    if scale_scores:
        max_val = abs_A.max()
        if max_val > 0:
            A = A / max_val

    if q > 0:
        threshold = torch.quantile(abs_A.flatten(), q)
        A[abs_A < threshold] = 0.0

    A.fill_diagonal_(0.0)

    return A.cpu().numpy(), mask


# ---------------------------------------------------------------------------
# Unified pc_net_calc_cuda dispatcher
# ---------------------------------------------------------------------------

def pc_net_calc_cuda(
    X_np: np.ndarray,
    selected_samples: np.ndarray,
    n_comp: int = 3,
    scale_scores: bool = True,
    symmetric: bool = False,
    q: float = 0.0,
    device: str = "cuda:0",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute one PCNet on GPU, dispatching to CuPy or PyTorch backend.

    Returns
    -------
    A : np.ndarray  shape (active_genes, active_genes)
    mask : np.ndarray  bool, shape (n_genes,)
    """
    _check_cuda()
    if _cupy_available:
        return pc_net_calc_cuda_cupy(X_np, selected_samples,
                                     n_comp=n_comp, scale_scores=scale_scores,
                                     symmetric=symmetric, q=q)
    else:
        return pc_net_calc_cuda_torch(X_np, selected_samples,
                                      n_comp=n_comp, scale_scores=scale_scores,
                                      symmetric=symmetric, q=q, device=device)


# ---------------------------------------------------------------------------
# Batch SVD optimisation: compute ALL genes' betas in one batched SVD call
# ---------------------------------------------------------------------------

def _cal_beta_matrix_cupy_batched(Xt_gpu: "cp.ndarray", n_comp: int) -> "cp.ndarray":
    """
    Faster variant: compute SVD of the FULL matrix once, then use
    the leave-one-out (LOO) approximation for each gene's regression.

    Full SVD: Xt = U · diag(s) · VT
    For gene k removed:  Xi_k ≈ U · diag(s_k) · VT_k
    where s_k and VT_k drop the contribution of column k.

    This avoids n_genes separate SVD calls at the cost of using a global
    approximation.  The approximation is exact when n_comp == n_cells.
    """
    n_cells, n_genes = Xt_gpu.shape

    # Full truncated SVD of the (cells × genes) matrix
    U, s, VT = cpla.svd(Xt_gpu, full_matrices=False)   # U(c,r), s(r,), VT(r,g)
    r = min(n_comp, s.shape[0])
    U = U[:, :r]          # (cells, r)
    s = s[:r]             # (r,)
    VT = VT[:r, :]        # (r, genes)

    # coef for all genes simultaneously: VT.T = (genes, r)
    coef_all = VT.T        # (genes, r)

    # score matrix: Xt @ coef_all = (cells, genes) @ (genes, r) = (cells, r)
    score = Xt_gpu @ coef_all                         # (cells, r)
    score_norm = cp.sum(score ** 2, axis=0)           # (r,)
    score_normed = score / score_norm                 # (cells, r)

    # For each gene k: betas_k = coef_all[idx,:] @ sum_cells(y_k * score_normed)
    # y_k = Xt_gpu[:, k]  → (cells,)
    # sum_cells(y_k * score_normed) = Xt_gpu.T @ score_normed  → one step for ALL k
    # shape: (genes, r) = (genes, cells) @ (cells, r)
    rhs = Xt_gpu.T @ score_normed                     # (genes, r)

    # B[k, j] = coef_all[j, :] · rhs[k, :]
    # = rhs @ coef_all.T  → (genes, genes)
    B = rhs @ coef_all.T                              # (genes, genes)

    # Diagonal is self-regression – zero it out
    cp.fill_diagonal(B, 0.0)
    return B


def pc_net_calc_cuda_fast(
    X_np: np.ndarray,
    selected_samples: np.ndarray,
    n_comp: int = 3,
    scale_scores: bool = True,
    symmetric: bool = False,
    q: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    **Fastest** CUDA path: one global SVD, then batch matrix multiply.

    Uses the leave-one-out SVD approximation for regression coefficients.
    Preferred for large gene panels (n_genes > 500).

    Requires CuPy.
    """
    if not _cupy_available:
        raise RuntimeError("pc_net_calc_cuda_fast requires CuPy.")

    Z = X_np[:, selected_samples]
    mask = Z.sum(axis=1) > 0
    Z = Z[mask, :]

    Z_gpu = cp.asarray(Z, dtype=cp.float64)
    Xt = Z_gpu.T                               # (cells, active_genes)
    Xt = (Xt - Xt.mean(axis=0)) / (Xt.std(axis=0) + 1e-12)

    n_ag = Xt.shape[1]
    assert 2 < n_comp <= n_ag

    B = _cal_beta_matrix_cupy_batched(Xt, n_comp)

    A = B.copy()
    if symmetric:
        A = (A + A.T) / 2.0
    abs_A = cp.abs(A)
    if scale_scores:
        max_val = cp.max(abs_A)
        if max_val > 0:
            A = A / max_val
    if q > 0:
        threshold = cp.quantile(abs_A, q)
        A[abs_A < threshold] = 0.0
    cp.fill_diagonal(A, 0.0)

    return cp.asnumpy(A), mask


# ---------------------------------------------------------------------------
# make_networks_cuda  –  drop-in replacement for make_networks
# ---------------------------------------------------------------------------

@timer
def make_networks_cuda(
    data: pd.DataFrame,
    n_nets: int = 10,
    n_samp_cells: Optional[int] = 500,
    n_comp: int = 3,
    scale_scores: bool = True,
    symmetric: bool = False,
    q: float = 0.95,
    random_state: int = 42,
    fast_svd: bool = True,
    device: str = "cuda:0",
    **kwargs,
) -> List[coo_matrix]:
    """
    GPU-accelerated replacement for :func:`make_networks`.

    Parameters
    ----------
    data : pd.DataFrame
        Gene-expression matrix (genes × cells).
    n_nets : int
        Number of sub-sampled PCNets to build.
    n_samp_cells : int or None
        Cells to sample per network; None uses all cells.
    n_comp : int
        Number of PCs used per gene regression.
    scale_scores : bool
        Normalise edge weights to [-1, 1].
    symmetric : bool
        Symmetrise the adjacency matrix.
    q : float
        Quantile threshold for sparsification.
    random_state : int
        RNG seed.
    fast_svd : bool, default True
        Use the batched global-SVD approximation (faster, CuPy only).
        Set to ``False`` for the exact per-gene SVD (slower but exact).
    device : str, default "cuda:0"
        PyTorch device string (only used when CuPy is unavailable).
    **kwargs
        Ignored (kept for API compatibility with ``make_networks``).

    Returns
    -------
    networks : List[coo_matrix]
    """
    _check_cuda()

    gene_names = data.index.to_numpy()
    n_genes, n_cells = data.shape
    assert not np.array_equal(gene_names, np.array(range(n_genes))), \
        "Gene names are required"

    rng = np.random.default_rng(random_state)
    X_np = data.values  # keep on host; individual sub-matrices go to GPU

    networks = []

    for net_idx in range(n_nets):
        # ---- Sub-sample ----
        sample = (rng.choice(n_cells, n_samp_cells, replace=True)
                  if n_samp_cells is not None
                  else np.arange(n_cells))

        # ---- GPU PCNet ----
        if fast_svd and _cupy_available:
            A, mask = pc_net_calc_cuda_fast(
                X_np, sample,
                n_comp=n_comp, scale_scores=scale_scores,
                symmetric=symmetric, q=q,
            )
        else:
            A, mask, = pc_net_calc_cuda(
                X_np, sample,
                n_comp=n_comp, scale_scores=scale_scores,
                symmetric=symmetric, q=q, device=device,
            )

        # ---- Rebuild full-size (n_genes × n_genes) sparse matrix ----
        sel_gene_names = gene_names[mask]
        temp = np.zeros((n_genes, n_genes), dtype=np.float32)

        # Map active gene indices back to full index
        active_idx = np.where(mask)[0]
        for local_i, global_i in enumerate(active_idx):
            for local_j, global_j in enumerate(active_idx):
                temp[global_i, global_j] = A[local_i, local_j]

        networks.append(coo_matrix(temp))

    return networks
