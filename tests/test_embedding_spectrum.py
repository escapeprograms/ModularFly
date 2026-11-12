import math

import pytest
import torch

from modularfly.config import InterpretabilityConfig
from modularfly.interpretability.embedding import (
    dominant_frequencies,
    embedding_spectrum,
    gini,
    model_token_spectra,
)
from tests.test_split_input import P, make_model, trainable_fly


def test_gini_one_hot_and_uniform():
    n = 6
    one_hot = torch.zeros(n)
    one_hot[2] = 3.0
    assert gini(one_hot) == pytest.approx(1 - 1 / n, abs=1e-6)
    assert gini(torch.full((n,), 2.5)) == pytest.approx(0.0, abs=1e-6)
    assert gini(torch.zeros(n)) == 0.0


def test_dominant_frequencies():
    norms = torch.tensor([100.0, 1.0, 9.0, 0.5, 10.0])
    assert dominant_frequencies(norms, 0.5) == (2, 4)


def test_embedding_spectrum_synthetic():
    p, d, k = 13, 5, 3
    x = torch.arange(p, dtype=torch.float64)
    r = torch.randn(d, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    weight = torch.cos(2 * math.pi * k * x / p)[:, None] * r[None, :]
    spec = embedding_spectrum(weight, InterpretabilityConfig())
    assert spec.coeffs.shape == (p, d)
    assert spec.dominant == (k,)
    others = torch.ones(p // 2 + 1, dtype=torch.bool)
    others[k] = False
    assert torch.allclose(spec.freq_norms[others], torch.zeros(p // 2, dtype=torch.float64), atol=1e-9)
    assert spec.freq_norms[k] > 1.0
    assert spec.gini == pytest.approx(1 - 1 / (p // 2), abs=1e-6)


def test_model_token_spectra_shapes(small_graph):
    model = make_model(small_graph, trainable_fly(small_graph))
    emb, unemb = model_token_spectra(model, InterpretabilityConfig())
    n_out = model.rnn.output_idx.numel()
    assert emb.coeffs.shape == (P, 7)
    assert unemb.coeffs.shape == (P, n_out)
    for spec in (emb, unemb):
        assert spec.freq_norms.shape == (P // 2 + 1,)
        assert not spec.coeffs.requires_grad
        assert not spec.freq_norms.requires_grad
