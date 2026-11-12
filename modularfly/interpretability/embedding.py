"""1D Fourier spectrum of a token weight matrix over the input-number axis.

Shapes: p = modulus, d = feature dim, K = p // 2.
Following Nanda et al. 2023 Sec. 4.1, a [p, d] weight (row x = the vector for number x)
is transformed with the real DFT along the input axis (basis @ weight), then each
frequency's (cos, sin) rows are reduced with an L2 norm along the feature axis. A model
that has grokked modular addition concentrates this spectrum on a few key frequencies,
so the Gini coefficient of the non-constant norms is a sparsity progress measure.

SplitInputModel.unembedding.weight is an nn.Linear weight already shaped [p, n_out]
(the opposite of Nanda's W_U [d, p] convention), so the same function applies to both
the embedding [p, n_in // 2] and the unembedding without a transpose.
"""
from dataclasses import dataclass

import torch

from modularfly.config import InterpretabilityConfig
from modularfly.interpretability.fourier import fourier_1d, fourier_basis, frequency_norms
from modularfly.model.end_to_end.split_input import SplitInputModel


@dataclass(frozen=True)
class EmbeddingSpectrum:
    coeffs: torch.Tensor  # [p, d] Fourier coefficients (basis @ weight)
    freq_norms: torch.Tensor  # [K+1] per-frequency norm, index 0 = constant
    dominant: tuple[int, ...]  # k >= 1 with norm >= fraction * max over k >= 1, ascending
    gini: float  # Gini coefficient of freq_norms[1:]


def gini(x: torch.Tensor) -> float:
    """Gini coefficient of a non-negative 1D tensor (Hurley & Rickard 2009).

    G = 1 - 2 * sum_{k=1..n} (x_(k) / ||x||_1) * (n - k + 1/2) / n with x sorted ascending.
    One-hot gives 1 - 1/n, uniform gives 0. An all-zero x has no spread, so it is defined
    to return 0.0.
    """
    total = x.sum()
    if total == 0:
        return 0.0
    n = x.numel()
    ranks = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)
    weights = (n - ranks + 0.5) / n
    return float(1 - 2 * (x.sort().values / total * weights).sum())


def dominant_frequencies(freq_norms: torch.Tensor, fraction: float) -> tuple[int, ...]:
    """Frequencies k >= 1 with freq_norms[k] >= fraction * max(freq_norms[1:]), ascending.
    The constant component (index 0) is never returned."""
    nonconstant = freq_norms[1:]
    keep = nonconstant >= fraction * nonconstant.max()
    return tuple(int(k) + 1 for k in torch.nonzero(keep).flatten())


def embedding_spectrum(weight: torch.Tensor, cfg: InterpretabilityConfig) -> EmbeddingSpectrum:
    """[p, d] weight -> its Fourier spectrum over the input axis (detached, pure)."""
    weight = weight.detach()
    basis = fourier_basis(weight.shape[0], dtype=weight.dtype, device=weight.device)
    coeffs = fourier_1d(weight, basis)
    norms = frequency_norms(coeffs)
    return EmbeddingSpectrum(
        coeffs=coeffs,
        freq_norms=norms,
        dominant=dominant_frequencies(norms, cfg.dominant_fraction),
        gini=gini(norms[1:]),
    )


def model_token_spectra(
    model: SplitInputModel, cfg: InterpretabilityConfig
) -> tuple[EmbeddingSpectrum, EmbeddingSpectrum]:
    """(embedding spectrum, unembedding spectrum) of a SplitInputModel."""
    return (
        embedding_spectrum(model.embedding.weight, cfg),
        embedding_spectrum(model.unembedding.weight, cfg),
    )
