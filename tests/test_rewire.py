import torch

from modularfly.config import RewireConfig
from modularfly.data.graph import edge_keys
from modularfly.data.rewire import rewire
from tests.conftest import random_graph


def out_strength(graph):
    return torch.zeros(graph.n_neurons).index_add(0, graph.pre, graph.syn_count)


def test_rewire_preserves_degrees_and_simplicity():
    graph = random_graph(n_neurons=80, n_edges=600, seed=3)
    result = rewire(graph, RewireConfig(swaps_per_edge=5, max_rounds=500, seed=1))
    new = result.graph
    assert result.accepted == 5 * graph.n_edges
    assert torch.equal(new.in_degree(), graph.in_degree())
    assert torch.equal(new.out_degree(), graph.out_degree())
    assert not bool((new.pre == new.post).any())
    keys = edge_keys(new.pre, new.post, new.n_neurons)
    assert keys.unique().numel() == new.n_edges
    old = set(edge_keys(graph.pre, graph.post, graph.n_neurons).tolist())
    assert len(old - set(keys.tolist())) > 0


def test_rewire_keeps_out_strength():
    graph = random_graph(n_neurons=50, n_edges=300, seed=4)
    new = rewire(graph, RewireConfig(swaps_per_edge=2, max_rounds=500, seed=2)).graph
    torch.testing.assert_close(out_strength(new), out_strength(graph))
