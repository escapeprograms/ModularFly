"""Matplotlib figures for the Fourier analysis of a modular-addition network.

Object-oriented API only (no pyplot): no backend or global figure state is touched.
p = modulus, K = p // 2 frequencies. Every plot function returns its Figure;
save_figure is the only file I/O in the package.
"""
from pathlib import Path

import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from modularfly.config import InterpretabilityConfig


def _numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _bar_figure(values: torch.Tensor, size: tuple[float, float], title: str,
                xlabel: str, ylabel: str) -> Figure:
    """Bar chart of a 1-D tensor over its indices 0..len-1."""
    fig = Figure(figsize=size)
    ax = fig.add_subplot()
    y = _numpy(values)
    ax.bar(np.arange(len(y)), y)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    return fig


def _heatmap(fig: Figure, ax: Axes, matrix: np.ndarray, xlabel: str, ylabel: str) -> None:
    """imshow with rows on y, columns on x, (0, 0) bottom-left, plus a colorbar."""
    image = ax.imshow(matrix, origin="lower", aspect="auto")
    fig.colorbar(image, ax=ax)
    ax.set(xlabel=xlabel, ylabel=ylabel)


def plot_frequency_norms(freq_norms: torch.Tensor, cfg: InterpretabilityConfig) -> Figure:
    """[K+1] Fourier norms per frequency k = 0..K; index 0 is the constant term (Nanda 2023 Fig. 3)."""
    return _bar_figure(freq_norms, cfg.figure_size, "Fourier norm per frequency",
                       "frequency k", "Fourier norm")


def plot_input_profile(values: torch.Tensor, cfg: InterpretabilityConfig) -> Figure:
    """[p] activation for each input x = 0..p-1."""
    return _bar_figure(values, cfg.figure_size, "Activation for each input", "input x", "value")


def plot_activation_grid(grid: torch.Tensor, cfg: InterpretabilityConfig) -> Figure:
    """[p, p] activation over a (rows) x b (cols) (Nanda 2023 Fig. 4 centre)."""
    fig = Figure(figsize=cfg.heatmap_size)
    ax = fig.add_subplot()
    _heatmap(fig, ax, _numpy(grid), "b", "a")
    ax.set_title("Activation over (a, b)")
    return fig


def plot_fourier_grid(coeffs: torch.Tensor, cfg: InterpretabilityConfig) -> Figure:
    """[p, p] |2-D Fourier coefficients| of an activation grid (Nanda 2023 Fig. 4 right)."""
    fig = Figure(figsize=cfg.heatmap_size)
    ax = fig.add_subplot()
    _heatmap(fig, ax, np.abs(_numpy(coeffs)), "Fourier component of b", "Fourier component of a")
    ax.set_title("2-D Fourier coefficient magnitudes")
    return fig


def plot_fve_histogram(fve: torch.Tensor, cfg: InterpretabilityConfig) -> Figure:
    """[n] per-neuron fraction of variance explained by its best frequency (Nanda 2023 Fig. 5 left)."""
    fig = Figure(figsize=cfg.figure_size)
    ax = fig.add_subplot()
    ax.hist(_numpy(fve), bins=cfg.histogram_bins, range=(0.0, 1.0))
    ax.set(title="Best-frequency variance explained",
           xlabel="fraction of variance explained by best frequency", ylabel="neurons")
    return fig


def save_figure(fig: Figure, path: Path, cfg: InterpretabilityConfig) -> Path:
    """Write fig to path (creating parent directories) at cfg.dpi; return path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
    return path
