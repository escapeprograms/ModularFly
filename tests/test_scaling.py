import torch

from modularfly.model.fly.scaling import spectral_radius


def test_spectral_radius_matches_eigvals():
    gen = torch.Generator().manual_seed(1)
    a = torch.randn(60, 60, generator=gen, dtype=torch.float64) / 60 ** 0.5
    expected = torch.linalg.eigvals(a).abs().max().item()
    estimate = spectral_radius(lambda v: a @ v, 60, 2000, gen, dtype=torch.float64)
    assert abs(estimate - expected) / expected < 0.05


def test_spectral_radius_complex_dominant_pair():
    rotation = torch.tensor([[0.0, -2.0], [2.0, 0.0]], dtype=torch.float64)  # eigenvalues +-2i
    a = torch.block_diag(rotation, torch.diag(torch.tensor([0.5, -0.3], dtype=torch.float64)))
    gen = torch.Generator().manual_seed(0)
    estimate = spectral_radius(lambda v: a @ v, 4, 200, gen, dtype=torch.float64)
    assert abs(estimate - 2.0) < 1e-6
