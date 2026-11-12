"""Scale estimates used to choose w0 and the inhibitory init scale."""
import math
from collections.abc import Callable

import torch

from modularfly.data.graph import ConnectomeGraph
from modularfly.model.fly.propagate import SparsePropagator


def spectral_radius(matvec: Callable[[torch.Tensor], torch.Tensor], n: int, n_iter: int,
                    generator: torch.Generator, device=None, dtype=torch.float32) -> float:
    """|lambda_max| from norm growth ||A^k v||^(1/k): valid when the dominant eigenvalue
    is complex (no Rayleigh quotient). Averages log-growth over the second half of the run."""
    v = torch.randn(n, 1, generator=generator, dtype=dtype).to(device)
    v = v / v.norm()
    log_growth = []
    for _ in range(n_iter):
        v = matvec(v)
        norm = v.norm()
        log_growth.append(math.log(float(norm)))
        v = v / norm
    tail = log_growth[n_iter // 2:]
    return math.exp(sum(tail) / len(tail))


def w0_for_radius(radius: float, target_radius: float) -> float:
    return target_radius / radius


def signed_count_weights(graph: ConnectomeGraph) -> torch.Tensor:
    """sign(pre) * syn_count per edge: the unscaled condition-1 weights."""
    return graph.neuron_sign[graph.pre] * graph.syn_count


def ei_balance_scale(graph: ConnectomeGraph) -> float:
    """Inhibitory init scale that equalises total |weight| mass of E and I edges under
    iid |theta|: (# excitatory edges) / (# inhibitory edges)."""
    excitatory = graph.neuron_sign[graph.pre] > 0
    return float(excitatory.sum()) / float((~excitatory).sum())


def graph_spectral_radius(graph: ConnectomeGraph, weights: torch.Tensor, n_iter: int,
                          generator: torch.Generator, device=None) -> float:
    """Spectral radius of the sparse matrix W[post, pre] = weights."""
    propagator = SparsePropagator(graph, chunk_edges=graph.n_edges).to(device)
    w = weights.to(device)
    return spectral_radius(lambda v: propagator(w, v), graph.n_neurons, n_iter, generator, device)
