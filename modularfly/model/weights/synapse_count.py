"""Condition 1 (Shiu 2024) edge weights: the fly's synapse counts, signed and frozen.

forward() returns FloatTensor[E] aligned with graph.pre / graph.post.
"""
import torch
from torch import nn

from modularfly.data.graph import ConnectomeGraph
from modularfly.model.fly.scaling import signed_count_weights


class SynapseCountWeights(nn.Module):
    """Condition 1 (Shiu 2024): frozen w0 * sign(pre) * syn_count."""

    def __init__(self, graph: ConnectomeGraph, w0: float):
        super().__init__()
        self.register_buffer("weights", w0 * signed_count_weights(graph))

    def forward(self) -> torch.Tensor:
        return self.weights

    def drift_penalty(self) -> torch.Tensor:
        """Zero (0-dim, no grad): there are no trainable parameters to drift."""
        return self.weights.new_zeros(())
