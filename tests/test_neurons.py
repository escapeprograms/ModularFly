import math
from functools import partial

import pytest
import torch

from modularfly.config import InterpretabilityConfig
from modularfly.interpretability.neurons import (
    activation_grid, all_pairs, cluster_by_frequency, neuron_spectra,
    readout_window_mean, state_at_step,
)
from tests.conftest import SMALL_DYN, SMALL_P as P, make_model as make_group_model, trainable_fly

IDX = torch.tensor([3, 7, 11])


def make_model(graph):
    return make_group_model(graph, trainable_fly(graph))


def readout_grid(model, neuron_idx, pair_chunk):
    probe = partial(readout_window_mean, neuron_idx=neuron_idx)
    return activation_grid(model, InterpretabilityConfig(pair_chunk=pair_chunk), probe)


def test_all_pairs_order():
    expected = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
    assert all_pairs(3).tolist() == expected


def test_activation_grid_matches_iter_states(small_graph):
    model = make_model(small_graph)
    grid = readout_grid(model, IDX, pair_chunk=8)  # 49 pairs -> uneven last chunk
    assert grid.shape == (3, P, P)
    for a, b in [(0, 0), (2, 5), (6, 6)]:
        with torch.no_grad():
            states = list(model.rnn.iter_states(model.drive(torch.tensor([[a, b]]))))
        direct = torch.stack([h[IDX, 0] for h in states[-SMALL_DYN.readout_steps:]]).mean(0)
        assert torch.allclose(grid[:, a, b], direct, atol=1e-6)


def test_readout_grid_matches_forward(small_graph):
    model = make_model(small_graph)
    idx = model.rnn.output_idx
    grid = readout_grid(model, idx, pair_chunk=8)
    with torch.no_grad():
        expected = model.rnn(model.drive(all_pairs(P))).t()
    assert torch.allclose(grid.reshape(idx.numel(), -1), expected, atol=1e-6)


def test_chunking_is_invariant(small_graph):
    model = make_model(small_graph)
    assert torch.allclose(readout_grid(model, IDX, 49), readout_grid(model, IDX, 5), atol=1e-6)


def test_state_at_step(small_graph):
    model = make_model(small_graph)
    tokens = torch.tensor([[1, 2], [3, 4]])
    with torch.no_grad():
        first = next(model.rnn.iter_states(model.drive(tokens)))[IDX]
        assert torch.allclose(state_at_step(model, tokens, IDX, 0), first)
    with pytest.raises(ValueError):
        state_at_step(model, tokens, IDX, SMALL_DYN.n_steps)
    with pytest.raises(ValueError):
        state_at_step(model, tokens, IDX, -1)


def test_grid_has_no_grad(small_graph):
    model = make_model(small_graph)
    assert readout_grid(model, IDX, 8).requires_grad is False


def synthetic_spectra(p=13, k=4):
    x = torch.arange(p, dtype=torch.float32)
    a, b = x[:, None], x[None, :]
    w = 2 * math.pi * k / p
    grid = torch.stack([torch.cos(w * (a + b)), torch.cos(w * (a - b)), torch.full((p, p), 0.3)])
    return neuron_spectra(torch.tensor([10, 20, 30]), grid, InterpretabilityConfig())


def test_neuron_spectra_synthetic():
    spectra = synthetic_spectra()
    assert spectra.coeffs.shape == (3, 13, 13)
    assert spectra.dominant_freq[:2].tolist() == [4, 4]
    assert torch.allclose(spectra.fve, torch.tensor([1.0, 1.0, 0.0]), atol=1e-4)
    assert torch.allclose(spectra.sum_share, torch.tensor([1.0, 0.0, 0.0]), atol=1e-4)
    for t in (spectra.coeffs, spectra.fve, spectra.sum_share):
        assert not torch.isnan(t).any()


def test_operand_only_neuron_has_zero_sum_share():
    p, k = 13, 4
    a = torch.arange(p, dtype=torch.float32)[:, None].expand(p, p)
    grid = torch.cos(2 * math.pi * k * a / p).unsqueeze(0)
    spectra = neuron_spectra(torch.tensor([5]), grid, InterpretabilityConfig())
    assert spectra.dominant_freq.tolist() == [k]
    assert torch.allclose(spectra.fve, torch.tensor([1.0]), atol=1e-4)
    assert spectra.sum_share.tolist() == [0.0]


def test_cluster_by_frequency():
    clusters = cluster_by_frequency(synthetic_spectra(), InterpretabilityConfig(fve_threshold=0.85))
    assert list(clusters) == [4]
    assert torch.equal(clusters[4], torch.tensor([10, 20]))
