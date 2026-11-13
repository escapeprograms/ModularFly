"""Condition 2 (Duan 2025) edge weights: a base weight module times learned gains
shared by groups of edges.

Every forward() returns FloatTensor[E] aligned with graph.pre / graph.post.
"""
import torch
from torch import nn


class GainModulatedWeights(nn.Module):
    """Condition 2 (Duan 2025): base() * gain()."""

    def __init__(self, base: nn.Module, gain: nn.Module):
        super().__init__()
        self.base = base
        self.gain = gain

    def forward(self) -> torch.Tensor:
        return self.base() * self.gain()

    def drift_penalty(self) -> torch.Tensor:
        """Sum of the base and gain drift penalties."""
        return self.base.drift_penalty() + self.gain.drift_penalty()


class GroupGain(nn.Module):
    """forward() -> 1 + Z[edge_group]; Z starts at exactly 0. Not clamped: 1 + Z < 0
    flips an edge's sign, so training should log min_gain()."""

    def __init__(self, edge_group: torch.Tensor, n_groups: int):
        super().__init__()
        self.register_buffer("edge_group", edge_group.clone())
        self.z = nn.Parameter(torch.zeros(n_groups))

    def forward(self) -> torch.Tensor:
        return 1.0 + self.z[self.edge_group]

    def min_gain(self) -> torch.Tensor:
        return 1.0 + self.z.detach().min()

    def drift_penalty(self) -> torch.Tensor:
        """sum(Z^2): Z starts at exactly 0, so this is the squared drift from init."""
        return (self.z ** 2).sum()
