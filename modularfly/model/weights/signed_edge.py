"""Condition 3 (Cho 2026 with FlyWire signs) edge weights: one random trainable
magnitude per edge, with the presynaptic neuron's sign fixed.

forward() returns FloatTensor[E] aligned with graph.pre / graph.post. drift_penalty()
returns sum((theta - theta_init)^2), the L2 pull toward the initial magnitudes.
"""
import torch
from torch import nn

from modularfly.data.graph import ConnectomeGraph


class SignedEdgeWeights(nn.Module):
    """Condition 3 (Cho 2026 with FlyWire signs): sign(pre) * |theta| / sqrt(max(in_deg, 1)),
    theta ~ N(0, 1); inhibitory thetas are multiplied by inhibitory_init_scale at init."""

    def __init__(self, graph: ConnectomeGraph, inhibitory_init_scale: float,
                 generator: torch.Generator):
        super().__init__()
        sign = graph.neuron_sign[graph.pre]
        theta = torch.randn(graph.n_edges, generator=generator)
        theta = torch.where(sign < 0, theta * inhibitory_init_scale, theta)
        self.theta = nn.Parameter(theta)
        self.register_buffer("theta_init", theta.clone())
        in_degree = graph.in_degree().clamp(min=1).float()
        self.register_buffer("sign_over_sqrt_degree", sign / in_degree.sqrt()[graph.post])

    def forward(self) -> torch.Tensor:
        return self.sign_over_sqrt_degree * self.theta.abs()

    def drift_penalty(self) -> torch.Tensor:
        """sum((theta - theta_init)^2), 0-dim, with grad."""
        return ((self.theta - self.theta_init) ** 2).sum()
