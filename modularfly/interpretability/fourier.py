"""Fourier analysis of a modular-addition network (Nanda et al. 2023). Pure tensor math.

Shapes: p = modulus (odd), d = feature dim, K = p // 2 frequencies, w_k = 2*pi*k/p.
Basis F [p, p] is orthonormal over x = 0..p-1 with rows:
    row 0      = 1/sqrt(p)                      (constant)
    row 2k - 1 = sqrt(2/p) * cos(w_k x),  k = 1..K
    row 2k     = sqrt(2/p) * sin(w_k x),  k = 1..K
Per-frequency arrays have length K+1 with index 0 = the constant component.

For a 2D grid over (a, b), coefficient C[i, j] = component i of a, component j of b.
With cc = cos(wa)cos(wb), ss = sin(wa)sin(wb), sc = sin(wa)cos(wb), cs = cos(wa)sin(wb):
    cos(w(a+b)) = cc - ss      cos(w(a-b)) = cc + ss
    sin(w(a+b)) = sc + cs      sin(w(a-b)) = sc - cs
"""
import math

import torch


def fourier_basis(p: int, dtype=torch.float32, device=None) -> torch.Tensor:
    """[p, p] orthonormal real DFT basis (row layout in module docstring)."""
    if p % 2 == 0:
        raise ValueError(f"fourier_basis needs odd p so every frequency has a cos/sin pair; got p={p}")
    x = torch.arange(p, dtype=torch.float64)
    k = torch.arange(1, p // 2 + 1, dtype=torch.float64)
    angle = 2 * math.pi * k[:, None] * x[None, :] / p  # [K, p]
    basis = torch.empty(p, p, dtype=torch.float64)
    basis[0] = 1 / math.sqrt(p)
    basis[1::2] = math.sqrt(2 / p) * torch.cos(angle)
    basis[2::2] = math.sqrt(2 / p) * torch.sin(angle)
    return basis.to(dtype=dtype, device=device)


def pair_sum(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Along `dim` of size p: [x0, x1+x2, x3+x4, ...] -> size K+1 (groups cos/sin of one frequency)."""
    x = x.movedim(dim, -1)
    out = torch.cat([x[..., :1], x[..., 1::2] + x[..., 2::2]], dim=-1)
    return out.movedim(-1, dim)


def fourier_1d(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """[p, d] -> [p, d] Fourier coefficients over the input axis."""
    return basis @ x


def fourier_2d(grid: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """[..., p, p] -> [..., p, p]: F @ grid @ F.T; (i, j) = component i of a, component j of b."""
    return basis @ grid @ basis.T


def frequency_norms(coeffs: torch.Tensor) -> torch.Tensor:
    """[p, d] -> [K+1]: L2 norm of each frequency's (cos, sin) rows."""
    return pair_sum((coeffs ** 2).sum(dim=1), dim=0).sqrt()


def block_energy(coeffs: torch.Tensor) -> torch.Tensor:
    """[..., p, p] -> [..., K+1, K+1]: squared coefficients summed within frequency blocks."""
    return pair_sum(pair_sum(coeffs ** 2, dim=-1), dim=-2)


def frequency_energy(energy: torch.Tensor) -> torch.Tensor:
    """[..., K+1, K+1] -> [..., K]: E[k,k] + E[0,k] + E[k,0] for k = 1..K
    (Nanda's 3x3 block for frequency k without the (0,0) constant)."""
    diag = energy[..., 1:, 1:].diagonal(dim1=-2, dim2=-1)
    return diag + energy[..., 0, 1:] + energy[..., 1:, 0]


def sum_difference_energy(coeffs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[..., p, p] -> (sum [..., K], diff [..., K]): energy of the (k,k) 2x2 block on
    cos/sin(w(a+b)) vs cos/sin(w(a-b)); sum + diff equals the block energy exactly."""
    cc = coeffs[..., 1::2, 1::2].diagonal(dim1=-2, dim2=-1)  # [..., K]
    ss = coeffs[..., 2::2, 2::2].diagonal(dim1=-2, dim2=-1)
    sc = coeffs[..., 2::2, 1::2].diagonal(dim1=-2, dim2=-1)
    cs = coeffs[..., 1::2, 2::2].diagonal(dim1=-2, dim2=-1)
    total = ((cc - ss) ** 2 + (sc + cs) ** 2) / 2
    diff = ((cc + ss) ** 2 + (sc - cs) ** 2) / 2
    return total, diff


def frequency_component(coeffs: torch.Tensor, basis: torch.Tensor, k: int) -> torch.Tensor:
    """[p, d] -> [p, d]: the part of the signal carried by frequency k, reconstructed over x."""
    n_freq = basis.shape[0] // 2
    if not 0 <= k <= n_freq:
        raise ValueError(f"frequency k={k} outside 0..{n_freq}")
    rows = [0] if k == 0 else [2 * k - 1, 2 * k]
    return basis[rows].T @ coeffs[rows]
