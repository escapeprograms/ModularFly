import math

import pytest
import torch

from modularfly.interpretability.fourier import (
    block_energy, fourier_1d, fourier_2d, fourier_basis, frequency_component,
    frequency_energy, frequency_norms, pair_sum, sum_difference_energy,
)

P = 13
F64 = torch.float64


def _grid(p, k, sign):
    a = torch.arange(p, dtype=F64)
    return torch.cos(2 * math.pi * k * (a[:, None] + sign * a[None, :]) / p)


@pytest.mark.parametrize("p", [7, 113])
def test_basis_orthonormal(p):
    f = fourier_basis(p, dtype=F64)
    assert torch.allclose(f @ f.T, torch.eye(p, dtype=F64), atol=1e-10)


def test_basis_rejects_even_p():
    with pytest.raises(ValueError):
        fourier_basis(8)


def test_pair_sum():
    assert torch.equal(pair_sum(torch.tensor([1., 2, 3, 4, 5]), 0), torch.tensor([1., 5, 9]))
    x = torch.arange(10.).reshape(2, 5)
    expected = torch.tensor([[0., 3, 7], [5, 13, 17]])
    assert torch.equal(pair_sum(x, 1), expected)


def test_cosine_weight_has_single_frequency():
    f = fourier_basis(P, dtype=F64)
    x = torch.arange(P, dtype=F64)
    r = torch.randn(6, dtype=F64, generator=torch.Generator().manual_seed(0))
    w = torch.cos(2 * math.pi * 2 * x / P)[:, None] * r[None, :]
    norms = frequency_norms(fourier_1d(w, f))
    assert norms[2] > 1e-3
    mask = torch.ones_like(norms, dtype=torch.bool)
    mask[2] = False
    assert torch.allclose(norms[mask], torch.zeros_like(norms[mask]), atol=1e-10)


def test_frequency_component_reconstructs():
    f = fourier_basis(P, dtype=F64)
    w = torch.randn(P, 5, dtype=F64, generator=torch.Generator().manual_seed(1))
    coeffs = fourier_1d(w, f)
    total = sum(frequency_component(coeffs, f, k) for k in range(P // 2 + 1))
    assert torch.allclose(total, w, atol=1e-10)
    with pytest.raises(ValueError):
        frequency_component(coeffs, f, P // 2 + 1)


def test_sum_grid_is_sum_dominated():
    f = fourier_basis(P, dtype=F64)
    k = 3
    c = fourier_2d(_grid(P, k, +1), f)
    energy = frequency_energy(block_energy(c))
    total = (c ** 2).sum()
    assert torch.isclose(energy[k - 1], total)
    others = torch.cat([energy[:k - 1], energy[k:]])
    assert torch.allclose(others, torch.zeros_like(others), atol=1e-10)
    s, d = sum_difference_energy(c)
    assert torch.allclose(d, torch.zeros_like(d), atol=1e-10)
    assert torch.isclose(s[k - 1], total)
    s, d = sum_difference_energy(fourier_2d(_grid(P, k, -1), f))
    assert torch.allclose(s, torch.zeros_like(s), atol=1e-10)
    assert d[k - 1] > 1e-3


def test_sum_plus_diff_equals_block_energy():
    f = fourier_basis(P, dtype=F64)
    c = fourier_2d(torch.randn(3, P, P, dtype=F64, generator=torch.Generator().manual_seed(2)), f)
    s, d = sum_difference_energy(c)
    expected = block_energy(c)[..., 1:, 1:].diagonal(dim1=-2, dim2=-1)
    assert torch.allclose(s + d, expected, atol=1e-10)


def test_block_energy_total():
    f = fourier_basis(P, dtype=F64)
    c = fourier_2d(torch.randn(P, P, dtype=F64, generator=torch.Generator().manual_seed(3)), f)
    assert torch.isclose(block_energy(c).sum(), (c ** 2).sum())


def test_fourier_2d_batched_matches_loop():
    f = fourier_basis(P, dtype=F64)
    grids = torch.randn(3, P, P, dtype=F64, generator=torch.Generator().manual_seed(4))
    batched = fourier_2d(grids, f)
    for i in range(3):
        assert torch.allclose(batched[i], fourier_2d(grids[i], f), atol=1e-12)
