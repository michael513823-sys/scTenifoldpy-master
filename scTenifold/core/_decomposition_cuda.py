"""
CUDA-accelerated tensor decomposition.

Strategy
--------
TensorLy supports multiple backends including **PyTorch** and **CuPy**.
We switch the global TensorLy backend to one of these before calling
``parafac``, so the entire CP decomposition (including ALS iterations)
runs on GPU with no code duplication.

Requires
--------
* tensorly >= 0.6
* torch (GPU build) **or** cupy
"""

from __future__ import annotations

from typing import Sequence
from warnings import warn

import numpy as np
import pandas as pd

import tensorly as tl
from tensorly import decomposition

from scTenifold.core._utils import timer

__all__ = ["tensor_decomp_cuda"]

# ---------------------------------------------------------------------------
# Backend detection (mirrors _networks_cuda.py)
# ---------------------------------------------------------------------------

_cupy_available = False
_torch_cuda_available = False

try:
    import cupy as cp
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
            "No CUDA backend available for tensor decomposition.\n"
            "Install cupy (pip install cupy-cuda11x / cupy-cuda12x) or\n"
            "a CUDA-enabled PyTorch build."
        )


# ---------------------------------------------------------------------------
# Helper: send numpy tensor to GPU via the chosen backend
# ---------------------------------------------------------------------------

def _to_gpu_tensor(arr: np.ndarray, device: str = "cuda:0"):
    """Move a NumPy ndarray to the GPU and return a backend tensor."""
    if _cupy_available:
        return cp.asarray(arr, dtype=cp.float64)
    else:
        return torch.as_tensor(arr, dtype=torch.float64,
                               device=torch.device(device))


def _from_gpu_tensor(tensor) -> np.ndarray:
    """Copy a GPU tensor back to a NumPy array."""
    if _cupy_available:
        return cp.asnumpy(tensor)
    else:
        return tensor.cpu().numpy()


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

@timer
def tensor_decomp_cuda(
    networks: np.ndarray,
    gene_names: Sequence[str],
    method: str = "parafac",
    n_decimal: int = 1,
    K: int = 5,
    tol: float = 1e-6,
    max_iter: int = 1000,
    random_state: int = 42,
    device: str = "cuda:0",
    **kwargs,
) -> pd.DataFrame:
    """
    GPU-accelerated CP tensor decomposition.

    Identical interface to :func:`scTenifold.core._decomposition.tensor_decomp`
    but switches TensorLy's backend to **PyTorch (GPU)** or **CuPy** before
    running the decomposition, so all inner ALS iterations execute on the GPU.

    Parameters
    ----------
    networks : np.ndarray
        Concatenated networks, shape ``(n_genes, n_genes, n_pcnets)``.
    gene_names : sequence of str
        Gene labels (length must match first two dims of ``networks``).
    method : str, default ``'parafac'``
        Any ``tensorly.decomposition`` method name.
    n_decimal : int
        Rounding precision for output values.
    K : int
        CP rank.
    tol : float
        ALS convergence tolerance.
    max_iter : int
        Maximum ALS iterations.
    random_state : int
        Seed for reproducibility.
    device : str, default ``"cuda:0"``
        CUDA device string.  Used only when PyTorch backend is active.
    **kwargs
        Extra keyword arguments forwarded to the TensorLy decomposition.

    Returns
    -------
    tensor_decomp_df : pd.DataFrame  shape (n_genes, n_genes)
    """
    _check_cuda()

    # ---- 1. Select TensorLy backend ----
    _prev_backend = tl.get_backend()

    if _cupy_available:
        tl.set_backend("cupy")
        print(f"[tensor_decomp_cuda] Using TensorLy CuPy backend")
    elif _torch_cuda_available:
        tl.set_backend("pytorch")
        print(f"[tensor_decomp_cuda] Using TensorLy PyTorch (GPU) backend, device={device}")
    else:
        # Should never reach here due to _check_cuda(), but be safe
        raise RuntimeError("No CUDA backend available.")

    try:
        # ---- 2. Move tensor to GPU ----
        tensor_gpu = _to_gpu_tensor(networks, device=device)

        # ---- 3. Decompose on GPU ----
        factors = getattr(decomposition, method)(
            tensor_gpu,
            rank=K,
            n_iter_max=max_iter,
            tol=tol,
            random_state=random_state,
            **kwargs,
        )
        estimate_gpu = tl.cp_to_tensor(factors)
        print(f"[tensor_decomp_cuda] Decomposed tensor shape: {estimate_gpu.shape}")

        # ---- 4. Post-process on GPU, then move to host ----
        # Sum across the slice dimension (axis=-1) and normalise
        if _cupy_available:
            out_gpu = cp.sum(estimate_gpu, axis=-1) / networks.shape[-1]
            max_abs = cp.max(cp.abs(out_gpu))
            out_gpu = cp.round(out_gpu / max_abs, n_decimal)
            out_np = cp.asnumpy(out_gpu)
        else:
            out_gpu = torch.sum(estimate_gpu, dim=-1) / networks.shape[-1]
            max_abs = torch.max(torch.abs(out_gpu))
            out_gpu = torch.round(out_gpu / max_abs * (10 ** n_decimal)) / (10 ** n_decimal)
            out_np = out_gpu.cpu().numpy()

    finally:
        # Always restore the original backend
        tl.set_backend(_prev_backend)

    return pd.DataFrame(out_np, index=gene_names, columns=gene_names)
