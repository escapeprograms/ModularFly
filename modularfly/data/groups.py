"""Neuron selection, neuron grouping, and edge -> gain-index builders."""
from collections.abc import Sequence

import torch

from modularfly.config import DataConfig
from modularfly.data.graph import ConnectomeGraph, edge_keys


def select_neurons(graph: ConnectomeGraph, level: str, labels: Sequence[str]) -> torch.Tensor:
    """Sorted indices of neurons whose `level` label is one of `labels`."""
    vocab = graph.vocab[level]
    missing = [label for label in labels if label not in vocab]
    if missing:
        raise KeyError(f"labels {missing} not in vocabulary of {level}")
    wanted = torch.tensor([vocab.index(label) for label in labels])
    return torch.nonzero(torch.isin(graph.annotations[level], wanted)).flatten()


def group_neurons(graph: ConnectomeGraph, level: str,
                  hierarchy: Sequence[str] = DataConfig.annotation_levels) -> torch.Tensor:
    """Contiguous group id per neuron at `level`. A neuron without a label at `level`
    takes its label at the next coarser level (as its own group), and so on."""
    depth = list(hierarchy).index(level)
    level_rank = torch.full((graph.n_neurons,), -1, dtype=torch.long)
    code = torch.full((graph.n_neurons,), -1, dtype=torch.long)
    for rank in range(depth, -1, -1):
        level_codes = graph.annotations[hierarchy[rank]]
        fill = (level_rank < 0) & (level_codes >= 0)
        level_rank[fill], code[fill] = rank, level_codes[fill]
    if (level_rank < 0).any():
        raise ValueError(f"{int((level_rank < 0).sum())} neurons unlabeled at every level")
    _, groups = torch.unique(level_rank * (code.max() + 1) + code, return_inverse=True)
    return groups


def pair_gain_index(graph: ConnectomeGraph, groups: torch.Tensor):
    """One gain per occupied (post group, pre group) pair.
    Returns edge_group[E] and pair_keys[P, 2] = (post group, pre group)."""
    n_groups = int(groups.max()) + 1
    keys = edge_keys(groups[graph.post], groups[graph.pre], n_groups)
    unique_keys, edge_group = torch.unique(keys, return_inverse=True)
    pair_keys = torch.stack([unique_keys // n_groups, unique_keys % n_groups], dim=1)
    return edge_group, pair_keys


def global_gain_index(graph: ConnectomeGraph):
    """Duan's global-gain baseline: every edge shares one gain. Key 0."""
    return torch.zeros(graph.n_edges, dtype=torch.long), torch.zeros(1, dtype=torch.long)


def sign_gain_index(graph: ConnectomeGraph):
    """Duan's E/I-gain baseline: one gain per presynaptic sign. Keys (-1, +1)."""
    edge_group = (graph.neuron_sign[graph.pre] > 0).long()
    return edge_group, torch.tensor([-1, 1])
