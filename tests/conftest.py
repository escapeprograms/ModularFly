"""Tiny synthetic graphs; no FlyWire data needed."""
import pytest
import torch

from modularfly.config import DynamicsConfig
from modularfly.data.graph import ConnectomeGraph
from modularfly.data.groups import select_neurons
from modularfly.model.fly.connectome import ConnectomeRNN

LEVELS = ("flow", "super_class", "cell_class", "cell_sub_class", "supertype", "cell_type")
SMALL_DYN = DynamicsConfig(n_steps=6, readout_steps=2, propagate_chunk_edges=32)


def random_graph(n_neurons: int, n_edges: int, seed: int) -> ConnectomeGraph:
    gen = torch.Generator().manual_seed(seed)
    keys = torch.randperm(n_neurons * n_neurons, generator=gen)
    keys = keys[(keys // n_neurons) != (keys % n_neurons)][:n_edges]  # no self-edges
    annotations = {level: torch.randint(-1, 3, (n_neurons,), generator=gen) for level in LEVELS}
    annotations["flow"] = torch.randint(0, 3, (n_neurons,), generator=gen)
    vocab = {level: [f"{level}_{k}" for k in range(3)] for level in LEVELS}
    vocab["flow"] = ["afferent", "efferent", "intrinsic"]
    sign = torch.where(torch.rand(n_neurons, generator=gen) < 0.3, -1.0, 1.0)
    return ConnectomeGraph(
        n_neurons=n_neurons, pre=keys // n_neurons, post=keys % n_neurons,
        syn_count=torch.randint(5, 50, (n_edges,), generator=gen).float(),
        neuron_sign=sign, nt_confidence=torch.rand(n_neurons, generator=gen),
        annotations=annotations, vocab=vocab, root_ids=torch.arange(n_neurons) + 10**17,
    )


def dense_matrix(graph: ConnectomeGraph, weights: torch.Tensor) -> torch.Tensor:
    w = torch.zeros(graph.n_neurons, graph.n_neurons, dtype=weights.dtype)
    w[graph.post, graph.pre] = weights
    return w


def make_rnn(graph: ConnectomeGraph, weights: torch.nn.Module):
    """ConnectomeRNN driving the afferent and reading the efferent neurons.
    Returns (rnn, n_in, n_out)."""
    inputs = select_neurons(graph, "flow", ["afferent"])
    outputs = select_neurons(graph, "flow", ["efferent"])
    return ConnectomeRNN(graph, weights, inputs, outputs, SMALL_DYN), inputs.numel(), outputs.numel()


@pytest.fixture
def small_graph() -> ConnectomeGraph:
    return random_graph(n_neurons=40, n_edges=300, seed=0)
