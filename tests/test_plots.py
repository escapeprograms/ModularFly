import torch
from matplotlib.figure import Figure

from modularfly.config import InterpretabilityConfig
from modularfly.interpretability.plots import (
    plot_activation_grid, plot_fourier_grid, plot_frequency_norms, plot_fve_histogram,
    plot_input_profile, save_figure,
)

P = 7
K = P // 2
CFG = InterpretabilityConfig()


def _gen() -> torch.Generator:
    return torch.Generator().manual_seed(0)


def test_frequency_norms_has_one_bar_per_frequency():
    fig = plot_frequency_norms(torch.rand(K + 1, generator=_gen()), CFG)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    assert len(fig.axes[0].patches) == K + 1 == 4


def test_input_profile_has_one_bar_per_input():
    fig = plot_input_profile(torch.randn(P, generator=_gen()), CFG)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    assert len(fig.axes[0].patches) == P


def test_activation_grid_heatmap_and_colorbar():
    fig = plot_activation_grid(torch.randn(P, P, generator=_gen()), CFG)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    assert fig.axes[0].images[0].get_array().shape == (P, P)


def test_fourier_grid_heatmap_and_colorbar():
    coeffs = torch.randn(P, P, generator=_gen())
    fig = plot_fourier_grid(coeffs, CFG)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    array = fig.axes[0].images[0].get_array()
    assert array.shape == (P, P)
    assert (array >= 0).all()


def test_fve_histogram_uses_configured_bins():
    fig = plot_fve_histogram(torch.rand(50, generator=_gen()), CFG)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    assert len(fig.axes[0].patches) == CFG.histogram_bins


def test_save_figure_writes_png(tmp_path):
    fig = plot_input_profile(torch.randn(P, generator=_gen()), CFG)
    path = save_figure(fig, tmp_path / "sub" / "profile.png", CFG)
    assert path == tmp_path / "sub" / "profile.png"
    assert path.exists() and path.stat().st_size > 0
