"""ConnectomeRNN: weight-agnostic leaky tanh recurrence on the connectome.

h_{t+1} = (1 - alpha) h_t + alpha tanh(u + W h_t + b), state [N, B], h_0 = 0,
constant drive u on the input neurons for all T steps.
Public interface: forward(drive [B, n_in]) -> [B, n_out];
window_mean(drive [B, n_in], neuron_idx [n]) -> [n, B].
"""
from collections.abc import Iterator

import torch
from torch import nn

from modularfly.config import DynamicsConfig
from modularfly.data.graph import ConnectomeGraph
from modularfly.model.fly.propagate import SparsePropagator


class ConnectomeRNN(nn.Module):
    def __init__(self, graph: ConnectomeGraph, edge_weights: nn.Module,
                 input_idx: torch.Tensor, output_idx: torch.Tensor, cfg: DynamicsConfig):
        super().__init__()
        self.cfg = cfg
        self.n_neurons = graph.n_neurons
        self.edge_weights = edge_weights
        self.propagator = SparsePropagator(graph, cfg.propagate_chunk_edges)
        self.register_buffer("input_idx", input_idx.clone())
        self.register_buffer("output_idx", output_idx.clone())
        self.register_buffer("bias", torch.zeros(graph.n_neurons, 1))

    def expand_drive(self, drive: torch.Tensor) -> torch.Tensor:
        """[B, n_in] -> [N, B] with zeros outside the input neurons."""
        full = drive.new_zeros(self.n_neurons, drive.shape[0])
        return full.index_copy(0, self.input_idx, drive.t())

    def step(self, h: torch.Tensor, drive: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """One leaky tanh update. h, drive: [N, B]; weights: [E]."""
        recurrent = self.propagator(weights, h)
        alpha = self.cfg.alpha
        return (1 - alpha) * h + alpha * torch.tanh(drive + recurrent + self.bias)

    def iter_states(self, drive: torch.Tensor) -> Iterator[torch.Tensor]:
        """Yields the T states h_1..h_T, each [N, B]."""
        weights = self.edge_weights()
        full_drive = self.expand_drive(drive)
        h = full_drive.new_zeros(self.n_neurons, drive.shape[0])
        for _ in range(self.cfg.n_steps):
            h = self.step(h, full_drive, weights)
            yield h

    def window_mean(self, drive: torch.Tensor, neuron_idx: torch.Tensor) -> torch.Tensor:
        """Mean of h[neuron_idx] over the last readout_steps steps: [n, B]."""
        first_read = self.cfg.n_steps - self.cfg.readout_steps
        readout = [h[neuron_idx] for t, h in enumerate(self.iter_states(drive)) if t >= first_read]
        return torch.stack(readout).mean(dim=0)

    def forward(self, drive: torch.Tensor) -> torch.Tensor:
        """Mean output-neuron activity over the last K steps: [B, n_out]."""
        return self.window_mean(drive, self.output_idx).t()
