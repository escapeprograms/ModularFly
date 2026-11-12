"""Degree-preserving rewired control graph (Cho 2026 App. D).

Swap (a, b), (c, d) -> (a, d), (c, b) in batched rounds. Each edge slot keeps its pre
neuron and syn_count; only post endpoints move, so in- and out-degree (and out-strength)
are preserved exactly. Duplicates are rejected conservatively against the pre-round key
set and against collisions inside the round's batch.
"""
from dataclasses import dataclass

import torch

from modularfly.config import RewireConfig
from modularfly.data.graph import ConnectomeGraph, edge_keys


@dataclass
class RewireResult:
    graph: ConnectomeGraph
    rounds: int
    proposed: int
    accepted: int


def propose_pairs(n_edges: int, generator: torch.Generator, device) -> tuple[torch.Tensor, torch.Tensor]:
    perm = torch.randperm(n_edges, generator=generator).to(device)
    half = n_edges // 2
    return perm[:half], perm[half:2 * half]


def contains(sorted_keys: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    pos = torch.searchsorted(sorted_keys, query).clamp(max=sorted_keys.numel() - 1)
    return sorted_keys[pos] == query


def endpoint_mask(a, b, c, d) -> torch.Tensor:
    """Reject self-edge slots, swaps that create self-edges, and no-op swaps."""
    return (a != b) & (c != d) & (a != c) & (b != d) & (a != d) & (c != b)


def collision_free(keys_ad: torch.Tensor, keys_cb: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    """Reject candidates whose new key is produced more than once within the batch."""
    pooled = torch.cat([keys_ad[candidate], keys_cb[candidate]])
    _, inverse, counts = torch.unique(pooled, return_inverse=True, return_counts=True)
    once = counts[inverse] == 1
    n = int(candidate.sum())
    ok = candidate.clone()
    ok[candidate] = once[:n] & once[n:]
    return ok


def accepted_swaps(pre, post, i, j, n_neurons) -> torch.Tensor:
    a, b, c, d = pre[i], post[i], pre[j], post[j]
    existing = torch.sort(edge_keys(pre, post, n_neurons)).values
    keys_ad, keys_cb = edge_keys(a, d, n_neurons), edge_keys(c, b, n_neurons)
    candidate = endpoint_mask(a, b, c, d) & ~contains(existing, keys_ad) & ~contains(existing, keys_cb)
    return collision_free(keys_ad, keys_cb, candidate)


def rewire_round(pre, post, n_neurons, remaining, generator) -> int:
    """One batched round, modifying `post` in place. Returns swaps applied."""
    i, j = propose_pairs(pre.numel(), generator, pre.device)
    accept = accepted_swaps(pre, post, i, j, n_neurons)
    i, j = i[accept][:remaining], j[accept][:remaining]
    post_i, post_j = post[i].clone(), post[j].clone()
    post[i], post[j] = post_j, post_i
    return i.numel()


def rewire(graph: ConnectomeGraph, cfg: RewireConfig, device=None) -> RewireResult:
    """Accept swaps_per_edge * E swaps; raises if max_rounds is not enough."""
    generator = torch.Generator().manual_seed(cfg.seed)
    pre, post = graph.pre.to(device), graph.post.clone().to(device)
    target, accepted, rounds = cfg.swaps_per_edge * graph.n_edges, 0, 0
    while accepted < target:
        if rounds == cfg.max_rounds:
            raise RuntimeError(f"rewiring stopped after {rounds} rounds with {accepted}/{target} swaps")
        accepted += rewire_round(pre, post, graph.n_neurons, target - accepted, generator)
        rounds += 1
    new_graph = graph.with_edges(graph.pre, post.cpu(), graph.syn_count)
    return RewireResult(new_graph, rounds, rounds * (graph.n_edges // 2), accepted)
